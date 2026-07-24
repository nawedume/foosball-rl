from rsl_rl.env import VecEnv
from dataclasses import dataclass
from tensordict import TensorDict
import torch
import time
import warp as wp

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer

TERMINATION_HEIGHT = 0.15

GOAL_CENTER_BLUE = [0.563284, 0.0, 0.289419]
GOAL_CENTER_RED = [-0.550284, 0.0, 0.289419]

class FoosballEnv(VecEnv):
    """An environment for foosball training with corrected perspective symmetries."""

    def __init__(self, num_envs: int = 1, dt: float = 1.0 / 60.0, device: str = "cuda:0", model="model.xml", sync_with_viewer=False, always_blue=False) -> None:
        super().__init__()

        self.num_envs = num_envs
        self.sync_with_viewer = sync_with_viewer
        self.always_blue = always_blue

        self.num_actions = 8
        self.num_obs = 46
        self.num_privileged_obs = 46
        self.max_episode_length = int(60 / dt) // 2
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.decimation = 1

        self.device = device
        self.cfg = {}

        wp.set_device(device)
        with open("./model.xml") as f:
            model_str = f.read()

        self.mjm = mujoco.MjModel.from_xml_string(model_str)
        self.mjd = mujoco.MjData(self.mjm)
        self.mjm.opt.timestep = dt

        self.model_d = mjw.put_model(self.mjm)
        self.data_d = mjw.make_data(self.mjm, nworld=num_envs)

        self.ball_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "ball")

        blue_goal_sensor_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SENSOR, "blue_goal_reached")
        self.blue_goal_sensor_adr = self.mjm.sensor_adr[blue_goal_sensor_id]

        red_goal_sensor_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SENSOR, "red_goal_reached")
        self.red_goal_sensor_adr = self.mjm.sensor_adr[red_goal_sensor_id]

        self.blue_goal_center = torch.tensor(GOAL_CENTER_BLUE, device=self.device).unsqueeze(0)
        self.red_goal_center = torch.tensor(GOAL_CENTER_RED, device=self.device).unsqueeze(0)

        self.side = torch.zeros((self.num_envs), dtype=torch.int8, device=self.device)
        self.opp_side = 1 - self.side

        self.goal_reward = 100.0
        self.opponent_policy = None

        self._reset(None)

    def get_observations(self) -> TensorDict:
        q_pos = wp.to_torch(self.data_d.qpos).clone()
        q_vel = wp.to_torch(self.data_d.qvel).clone()

        is_red = self.side == 1

        # Relative ball positions and velocities
        ball_pos_rel = q_pos[:, 16:].clone()
        ball_pos_rel[:, :3] = torch.where(
            is_red.unsqueeze(1), 
            ball_pos_rel[:, :3] - self.red_goal_center, 
            ball_pos_rel[:, :3] - self.blue_goal_center
        )
        
        # Mirror BOTH X and Y axes for Red so left/right and forward/backward match perspective
        ball_pos_rel[is_red, 0] = -ball_pos_rel[is_red, 0]
        ball_pos_rel[is_red, 1] = -ball_pos_rel[is_red, 1]

        ball_vel_rel = q_vel[:, 16:].clone()
        ball_vel_rel[is_red, 0] = -ball_vel_rel[is_red, 0]
        ball_vel_rel[is_red, 1] = -ball_vel_rel[is_red, 1]

        # Rod states
        blue_pos = q_pos[:, :8]
        blue_vel = q_vel[:, :8]
        blue_state = torch.cat([blue_pos, blue_vel], dim=-1)

        red_pos = q_pos[:, 8:16]
        red_vel = q_vel[:, 8:16]
        # Negate Red's rod state so positive values represent forward tilt/movement from Red's view
        red_state = torch.cat([-red_pos, -red_vel], dim=-1)

        player_rod_state = torch.where(is_red.unsqueeze(1), red_state, blue_state)
        op_rod_state = torch.where(is_red.unsqueeze(1), blue_state, red_state)

        data = torch.cat([player_rod_state, op_rod_state, ball_pos_rel, ball_vel_rel, self.side.to(torch.float).unsqueeze(1)], dim=1)

        ret = {"policy": data}
        return TensorDict(ret, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # 1. Action Clamping & Scaling
        raw_actions = torch.clamp(actions, min=-1.0, max=1.0)
        scaled_actions = raw_actions * 40.0
        
        control = wp.to_torch(self.data_d.ctrl)
        assert control.shape[1] == 16

        if scaled_actions.shape == 16 or scaled_actions.shape[1] == 16:
            control[:] = scaled_actions
        else:
            control.zero_()
            is_red = self.side == 1

            if self.opponent_policy is not None:
                # Evaluate opponent
                self.side = 1 - self.side
                op_obs = self.get_observations()
                with torch.no_grad():
                    op_actions = self.opponent_policy(op_obs)
                    if isinstance(op_actions, TensorDict):
                        op_actions = op_actions["policy"]
                self.side = 1 - self.side

                op_scaled_actions = torch.clamp(op_actions, min=-1.0, max=1.0) * 40.0

                # Blue envs: Player is Blue (:8), Opponent is Red (8: negated)
                control[~is_red, :8] = scaled_actions[~is_red]
                control[~is_red, 8:] = -op_scaled_actions[~is_red]

                # Red envs: Player is Red (8: negated), Opponent is Blue (:8)
                control[is_red, 8:] = -scaled_actions[is_red]
                control[is_red, :8] = op_scaled_actions[is_red]

            else:
                op_actions = torch.sin(self.episode_length_buf.float() * 0.1).unsqueeze(1).repeat(1, 8) * 20.0

                control[~is_red, :8] = scaled_actions[~is_red]
                control[~is_red, 8:] = op_actions[~is_red]

                control[is_red, 8:] = -scaled_actions[is_red]
                control[is_red, :8] = op_actions[is_red]

        for i in range(self.decimation):
            mjw.step(self.model_d, self.data_d)

        wp.synchronize()
        self.episode_length_buf += 1

        # Environment State & Resets
        ball_pos = wp.to_torch(self.data_d.xpos)[:, self.ball_id]
        out_of_bounds = ball_pos[:, 2] <= TERMINATION_HEIGHT

        sensor_data = wp.to_torch(self.data_d.sensordata)
        blue_goals = sensor_data[:, self.blue_goal_sensor_adr] > 0.5
        red_goals = sensor_data[:, self.red_goal_sensor_adr] > 0.5

        dones = (self.episode_length_buf > self.max_episode_length).bool() | out_of_bounds | blue_goals | red_goals
        self.episode_length_buf[dones] = 0

        obs = self.get_observations()

        # Goal Rewards
        in_goal = blue_goals | red_goals
        in_right_goal = ((self.side == 0) & red_goals) | ((self.side == 1) & blue_goals)
        rewards = torch.zeros(self.num_envs, device=self.device)
        rewards[in_goal] = torch.where(in_right_goal[in_goal], self.goal_reward, -self.goal_reward)

        # Distance Penalty
        blue_side_not_in_goal = ~in_goal & (self.side == 0)
        red_side_not_in_goal = ~in_goal & (self.side == 1)

        blue_dist = torch.clamp(torch.linalg.vector_norm(ball_pos[blue_side_not_in_goal, :] - self.red_goal_center, dim=1), max=2.0)
        red_dist = torch.clamp(torch.linalg.vector_norm(ball_pos[red_side_not_in_goal, :] - self.blue_goal_center, dim=1), max=2.0)

        rewards[blue_side_not_in_goal] = -blue_dist * 0.01
        rewards[red_side_not_in_goal] = -red_dist * 0.01

        # Out of Bounds Penalty
        rewards[out_of_bounds] = -10.0

        # Velocity Bonus (Restored)
        ball_vel = wp.to_torch(self.data_d.qvel)[:, 16:]
        world_ball_vel_x = ball_vel[:, 0]

        blue_forward_vel = torch.clamp(-world_ball_vel_x[blue_side_not_in_goal], min=0.0)
        red_forward_vel = torch.clamp(world_ball_vel_x[red_side_not_in_goal], min=0.0)

        velocity_scale = 0.2
        rewards[blue_side_not_in_goal] += blue_forward_vel * velocity_scale
        rewards[red_side_not_in_goal] += red_forward_vel * velocity_scale

        # Control Penalty (Calculated on raw [-1, 1] actions)
        penalty_scale = 0.05
        control_penalty = torch.sum(torch.square(raw_actions), dim=1) * penalty_scale
        rewards -= control_penalty

        if dones.any():
            self._reset(dones)

        if self.sync_with_viewer:
            self.get_sim_data()

        return obs, rewards, dones, {}

    def _reset(self, dones: torch.Tensor | None = None):
        if dones is None:
            env_idx = slice(None)
            self.side[env_idx] = torch.randint(0, 2, size=(self.num_envs,), dtype=torch.int8, device=self.device)
            size = self.num_envs
        else:
            size = int(dones.sum().item())
            self.side[dones] = torch.randint(0, 2, size=(size,), dtype=torch.int8, device=self.device)
            mjw.reset_data(self.model_d, self.data_d, reset=wp.from_torch(dones))
            env_idx = dones

        if self.always_blue:
            self.side[:] = 0

        ball_vel = wp.to_torch(self.data_d.qvel)[:, 16:]
        ball_vel[env_idx, 0] = (torch.rand(size, device=self.device) - 0.5) * 0.1
        ball_vel[env_idx, 1] = -torch.rand(size, device=self.device) * 2.0
        mjw.forward(self.model_d, self.data_d)

    def get_sim_data(self):
        mjw.get_data_into(result=self.mjd, mjm=self.mjm, d=self.data_d)