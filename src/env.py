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

from torch.profiler import record_function

TERMINATION_HEIGHT = 0.15

GOAL_CENTER_BLUE = [0.563284, 0.0, 0.289419]
GOAL_CENTER_RED = [-0.550284, 0.0, 0.289419]

class FoosballEnv(VecEnv):
    """An environment for foosball training with corrected perspective symmetries."""

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1.0 / 60.0,
        device: str = "cuda:0",
        model="model.xml",
        sync_with_viewer=False,
        always_blue=False,
        op_policy=None,
        bias_to_blue=False,
    ) -> None:

        super().__init__()

        self.num_envs = num_envs
        self.sync_with_viewer = sync_with_viewer
        self.always_blue = always_blue
        self.bias_to_blue = bias_to_blue

        self.num_actions = 8
        self.max_episode_length = int(60 / dt) // 4
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.decimation = 1

        self.device = device
        self.cfg = {}

        wp.set_device(device)
        with open(model) as f:
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

        self.goal_reward = 300.0

        if op_policy is None:
            self.op_policy = NullPolicy(self.num_envs, device=self.device)
        else:
            self.op_policy = op_policy

        self._reset(None)

    def get_observations(self) -> TensorDict:
        is_red = self.side == 1
        return self._get_obs(is_red)

    @record_function("observations")
    def _get_obs(self, is_red: torch.Tensor) -> TensorDict:
        q_pos = wp.to_torch(self.data_d.qpos).clone()
        q_vel = wp.to_torch(self.data_d.qvel).clone()

        # Relative ball positions and velocities
        ball_pos_rel = q_pos[:, 16:19].clone()
        ball_pos_rel[:, :3] = torch.where(
            is_red.unsqueeze(1),
            ball_pos_rel[:, :3] - self.red_goal_center,
            ball_pos_rel[:, :3] - self.blue_goal_center
        )

        # Mirror BOTH X and Y axes for Red so left/right and forward/backward match perspective
        ball_pos_rel[is_red, 0] = -ball_pos_rel[is_red, 0]
        ball_pos_rel[is_red, 1] = -ball_pos_rel[is_red, 1]

        ball_vel_rel = q_vel[:, 16:19].clone()
        ball_vel_rel[is_red, 0] = -ball_vel_rel[is_red, 0]
        ball_vel_rel[is_red, 1] = -ball_vel_rel[is_red, 1]

        # Rod states
        blue_pos = q_pos[:, :8]
        blue_vel = q_vel[:, :8]
        blue_state = torch.cat([blue_pos, blue_vel], dim=-1)

        red_pos = q_pos[:, 8:16]
        red_vel = q_vel[:, 8:16]
        # Negate Red's rod state so positive values represent forward tilt/movement from Red's view
        red_state = torch.cat([red_pos, red_vel], dim=-1)

        player_rod_state = torch.where(is_red.unsqueeze(1), red_state, blue_state)
        op_rod_state = torch.where(is_red.unsqueeze(1), blue_state, red_state)

        data = torch.cat([player_rod_state, op_rod_state, ball_pos_rel, ball_vel_rel, is_red.to(torch.float).unsqueeze(1)], dim=1)

        ret = {"policy": data}
        td = TensorDict(ret, batch_size=[self.num_envs], device=self.device)
        return td

    @record_function("step")
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:

        # 1. Action Clamping & Scaling
        with record_function("action_clamp and control"):
            raw_actions = torch.clamp(actions, min=-1.0, max=1.0)
            scaled_actions = raw_actions * 40.0

            control = wp.to_torch(self.data_d.ctrl)

            is_red = self.side == 1
            control.zero_()

            control[is_red, 8:16] = scaled_actions[is_red]
            control[~is_red, :8] = scaled_actions[~is_red]

            op_actions = self.op_policy(self._get_obs(self.side == 0)) * 40.0
            control[is_red, :8] =  op_actions[is_red]
            control[~is_red, 8:16] = op_actions[~is_red]

        with record_function("MJC step"):
            for _ in range(self.decimation):
                mjw.step(self.model_d, self.data_d)

        with record_function("env state and resets"):
            # wp.synchronize()
            self.episode_length_buf += 1

            # Environment State & Resets
            ball_pos = wp.to_torch(self.data_d.xpos)[:, self.ball_id]
            out_of_bounds = ball_pos[:, 2] <= TERMINATION_HEIGHT

            sensor_data = wp.to_torch(self.data_d.sensordata)
            blue_goals = sensor_data[:, self.blue_goal_sensor_adr] > 0.5
            red_goals = sensor_data[:, self.red_goal_sensor_adr] > 0.5

            ball_vel = wp.to_torch(self.data_d.qvel)[:, 16:19]

        rewards = compute_rewards(
            blue_goals,
            red_goals,
            self.side,
            ball_pos,
            ball_vel,
            self.blue_goal_center,
            self.red_goal_center,
            out_of_bounds,
            self.goal_reward,
            self.num_envs,
        )

        with record_function("post reward"):
            time_outs = (self.episode_length_buf > self.max_episode_length).bool()
            dones =  time_outs | out_of_bounds | blue_goals | red_goals
            self.episode_length_buf[dones] = 0
            if dones.any():
                self._reset(dones)

            obs = self.get_observations()

            if self.sync_with_viewer:
                self.get_sim_data()

        return obs, rewards, dones, { 'time_outs': time_outs }
    @record_function("reset")
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
            self.side[env_idx] = 0

        qvel = wp.to_torch(self.data_d.qvel)
        ball_vel = qvel[:, 16:19]

        # always throw to blue
        bias = 0.0 if self.bias_to_blue else -0.5
        scale = 0.3 if self.bias_to_blue else 0.1

        ball_vel[env_idx, 0] = (torch.rand(size, device=self.device) + bias) * scale
        ball_vel[env_idx, 1] = -torch.rand(size, device=self.device) * 2.0

        # randomize ball starting position so the fooseball learns to kick right
        qpos = wp.to_torch(self.data_d.qpos)
        qpos[env_idx, 16:17] += ((torch.rand((size, 1), device=self.device) -  0.5) * 2.0) * 0.4 # from +- 0.4
        qpos[env_idx, 17:18] += ((torch.rand((size, 1), device=self.device) -  0.5) * 2.0) * 0.3 # from +- 0.3

        joint_ranges = wp.to_torch(self.model_d.jnt_range)[0, :16, :]
        min_range = joint_ranges[:, 0]
        max_range = joint_ranges[:, 1]

        joint_positions = torch.rand((size, 16), device=self.device) * (max_range - min_range) + min_range

        qpos[env_idx, :16] = joint_positions

        mjw.forward(self.model_d, self.data_d)

    def get_sim_data(self):
        wp.synchronize()
        mjw.get_data_into(result=self.mjd, mjm=self.mjm, d=self.data_d)


class NullPolicy:
    def __init__(self, num_envs, device='cuda:0'):
        self.device =  device
        self.num_envs = num_envs

    def __call__(self, obs):
        return torch.zeros((self.num_envs, 8), device=self.device)

@record_function("reward")
@torch.compile
def compute_rewards(
    blue_goals: torch.Tensor,
    red_goals: torch.Tensor,
    side: torch.Tensor,
    ball_pos,
    ball_vel,
    blue_goal_center,
    red_goal_center,
    out_of_bounds,
    goal_reward: float,
    num_envs: int,
):
    ## Goal Rewards
    in_goal = blue_goals | red_goals
    in_right_goal = ((side == 0) & red_goals) | ((side == 1) & blue_goals)
    rewards = torch.zeros(num_envs, device=ball_pos.device) # pyright: ignore
    rewards[in_goal] = torch.where(in_right_goal[in_goal], 1.5*goal_reward, -goal_reward)

    # Distance Penalty
    # blue_side_not_in_goal = ~in_goal & (side == 0)
    # red_side_not_in_goal = ~in_goal & (side == 1)
    is_blue = (side == 0).unsqueeze(1)

    # blue_dist = torch.clamp(torch.linalg.vector_norm(ball_pos[blue_side_not_in_goal, :] - red_goal_center, dim=1), max=2.0)
    # red_dist = torch.clamp(torch.linalg.vector_norm(ball_pos[red_side_not_in_goal, :] - blue_goal_center, dim=1), max=2.0)

    dist_vec = torch.where(is_blue, ball_pos - red_goal_center, ball_pos - blue_goal_center)
    dist = torch.linalg.vector_norm(dist_vec, dim=1)

    target_dir = dist_vec / dist.unsqueeze(1).clamp_min(1e-6)
    target_dir = target_dir[:, :2]
    vel_reward = torch.linalg.vecdot(ball_vel[:, :2], -target_dir[:, :2], dim=1) * 0.28

    # rewards[blue_side_not_in_goal] = -(blue_dist*blue_dist)
    # rewards[red_side_not_in_goal] = -(red_dist*red_dist)
    # dist_reward = -torch.clamp(dist, max=2.0).square()

    rewards += vel_reward # dist_reward + vel_reward

    # Out of Bounds Penalty
    rewards[out_of_bounds] += -300.0
    return rewards
