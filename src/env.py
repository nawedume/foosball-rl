from rsl_rl.env import VecEnv
from dataclasses import dataclass
from tensordict import TensorDict
import torch
import time
import warp as wp
# from typing import tuple

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer

TERMINATION_HEIGHT = 0.15

GOAL_CENTER_BLUE = [0.563284, 0.0, 0.289419]
GOAL_CENTER_RED = [-0.550284, 0.0, 0.289419]

class FoosballEnv(VecEnv):
    """An environment for the foosball training."""

    def __init__(self, num_envs: int = 1, dt: float = 1.0 / 60.0, device: str = "cuda:0", model="model.xml", sync_with_viewer=False) -> None:
        super().__init__()

        self.num_envs = num_envs
        self.sync_with_viewer = sync_with_viewer

        # for each 4 rods, we can rotate and slide, 4x2 = 8 actions.
        # should output torques and forces for the motoros. See model.xml for definitions of actuators.
        self.num_actions = 8
        self.num_obs = 46
        self.num_privileged_obs = 46
        # 1 minute worth of steps
        self.max_episode_length = int(60 / dt) // 2
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.decimation = 8

        self.device = device
        self.cfg = {}

        wp.set_device(device)
        # mujcoco init
        with open("./model.xml") as f:
            model_str = f.read()

        self.mjm = mujoco.MjModel.from_xml_string(model_str)  # pyright: ignore[reportAttributeAccessIssue]
        self.mjd = mujoco.MjData(self.mjm)  # pyright: ignore[reportAttributeAccessIssue]
        self.mjm.opt.timestep = dt

        # Might not need CPU buffer, can keep it on the GPU?
        # self.mjd = mujoco.MjData(mjm)  # pyright: ignore[reportAttributeAccessIssue]

        dt = self.mjm.opt.timestep

        # on device data
        self.model_d = mjw.put_model(self.mjm)
        self.data_d = mjw.make_data(self.mjm, nworld=num_envs)

        # Retrieve IDs for all items
        self.ball_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "ball"  # pyright: ignore[reportAttributeAccessIssue]
        )

        self.blue_rod_11 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "blue_11"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.blue_rod_12 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "blue_12"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.blue_rod_2 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "blue_2"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.blue_rod_3 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "blue_3"  # pyright: ignore[reportAttributeAccessIssue]
        )

        self.red_rod_11 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "red_11"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.red_rod_12 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "red_12"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.red_rod_2 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "red_2"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.red_rod_3 = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "red_3"  # pyright: ignore[reportAttributeAccessIssue]
        )

        blue_goal_sensor_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_SENSOR,  # pyright: ignore[reportAttributeAccessIssue]
            "blue_goal_reached"
        )
        self.blue_goal_sensor_adr = self.mjm.sensor_adr[blue_goal_sensor_id]

        red_goal_sensor_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.mjm, mujoco.mjtObj.mjOBJ_SENSOR,  # pyright: ignore[reportAttributeAccessIssue]
            "red_goal_reached"
        )
        self.red_goal_sensor_adr = self.mjm.sensor_adr[red_goal_sensor_id]

        # unfortunately due to a bad table asset, this isn't symmetrical :(
        # should update the asset later to be much better
        self.blue_goal_center = torch.tensor(GOAL_CENTER_BLUE, device=self.device).unsqueeze(0)
        self.red_goal_center = torch.tensor(GOAL_CENTER_RED, device=self.device).unsqueeze(0)

        self.side = torch.zeros((self.num_envs), dtype=torch.int8, device=self.device)
        self.opp_side = 1 - self.side

        self.goal_reward = 100.0

        self.opponent_policy = None

        self._reset(None)


    def get_observations(self) -> TensorDict:
        # All this is pretty bad so we need to optimize. The idea here is that we want
        # to ensure that the view for the policy is symmetrical regardless of what side its on.
        # So we make the position relative to its own goal, and negate the velocity if its player 2 (red)
        # We also ensure that the player state is first, and add a single bit of information stating the player
        # so the policy can deal with asymmetry on the table, which exists because its a bad table asset.
        # In the future, we should really optimize this, perhaps using Warp, and also use a better table asset.
        q_pos = wp.to_torch(self.data_d.qpos).clone()
        q_vel = wp.to_torch(self.data_d.qvel).clone()

        # qpos and qvel should just be the cartesian coordinates
        is_red = self.side == 1

        ball_pos_rel = q_pos[:, 16:]
        ball_pos_rel[:, :3] = torch.where(is_red.unsqueeze(1), ball_pos_rel[:, :3] - self.red_goal_center, ball_pos_rel[:, :3] - self.blue_goal_center)

        # reverse the direction so its always from the perspective of the current player
        ball_vel_rel = q_vel[:, 16:]
        ball_vel_rel[is_red, 0:2] = -ball_vel_rel[is_red, 0:2]

        blue_pos = q_pos[:, :8]
        blue_vel = q_vel[:, :8]
        blue_state = torch.cat([blue_pos, blue_vel], dim=-1)

        red_pos = q_pos[:, 8:16]
        red_vel = q_vel[:, 8:16]
        red_state = torch.cat([red_pos, red_vel], dim=-1)

        player_rod_state = torch.where(is_red.unsqueeze(1), red_state, blue_state)
        op_rod_state = torch.where(is_red.unsqueeze(1), blue_state, red_state)

        data = torch.cat([player_rod_state, op_rod_state, ball_pos_rel, ball_vel_rel, self.side.to(torch.float).unsqueeze(1)], dim=1)

        ret = { "policy": data }
        return TensorDict(ret, batch_size=[self.num_envs], device=self.device)


    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:

        actions = actions*40
        control = wp.to_torch(self.data_d.ctrl)
        assert control.shape[1] == 16

        if actions.shape == 16:
            control[:] = actions
        else:
            control.zero_()
            is_red = self.side == 1

            if self.opponent_policy is not None:
                self.side = 1 - self.side
                op_obs = self.get_observations()
                with torch.no_grad():
                    op_actions = self.opponent_policy(op_obs)
                self.side = 1 - self.side

                control[~is_red, :8] = actions[~is_red]
                control[~is_red, 8:] = op_actions[~is_red]

                control[is_red, 8:] = actions[is_red]
                control[is_red, :8] = op_actions[is_red]

            else:
                op_actions = torch.sin(self.episode_length_buf.float()*0.1).unsqueeze(1).repeat(1,8)
                
                control[~is_red, :8] = actions[~is_red]
                control[~is_red, 8:] = op_actions[~is_red]

                control[is_red, 8:] = actions[is_red]
                control[is_red, :8] = op_actions[is_red]
        
        control = control
        for i in range(self.decimation):
            mjw.step(self.model_d, self.data_d)

        # Do I need this here if we just launch GPU kernels? Probably not remove later.
        wp.synchronize()

        self.episode_length_buf += 1

        # used for termination and rewards
        ball_pos = wp.to_torch(self.data_d.xpos)[:, self.ball_id]
        out_of_bounds = ball_pos[:, 2] <= TERMINATION_HEIGHT

        sensor_data = wp.to_torch(self.data_d.sensordata)
        blue_goals = sensor_data[:, self.blue_goal_sensor_adr] > 0.5
        red_goals = sensor_data[:, self.red_goal_sensor_adr] > 0.5

        dones = (self.episode_length_buf > self.max_episode_length).bool() | out_of_bounds | blue_goals | red_goals
        self.episode_length_buf[dones] = 0

        # TODO: reset environments
        if dones.any():
            self._reset(dones)

        obs = self.get_observations()

        # include goal scoring here later.
        in_goal = blue_goals | red_goals
        goal_side = self.side[in_goal]

        in_right_goal = ((self.side == 0) & blue_goals) | ((self.side == 1) & red_goals)
        rewards = torch.zeros(self.num_envs,  device=self.device)
        rewards[in_goal] = torch.where(in_right_goal[in_goal], self.goal_reward, -self.goal_reward)

        # add distance reward
        # goal pivot is is (1 x 3), and the ball_pos is of sice (N, 3).
        blue_side_not_in_goal = ~in_goal & (self.side == 0)
        rewards[blue_side_not_in_goal] = 2.0 - torch.linalg.vector_norm(ball_pos[blue_side_not_in_goal, :] - self.red_goal_center, dim=1)

        red_side_not_in_goal = ~in_goal & (self.side == 1)
        rewards[red_side_not_in_goal] = 2.0 - torch.linalg.vector_norm(ball_pos[red_side_not_in_goal, :] - self.blue_goal_center, dim=1)

        if self.sync_with_viewer:
            self.get_sim_data()

        return obs, rewards, dones, {}


    def _reset(self, dones: torch.Tensor | None = None):
        if dones is None:
            env_idx = slice(None)
            self.side[env_idx] = torch.randint(0, 2, size=(self.num_envs, ), dtype=torch.int8, device=self.device)
            size = self.num_envs

        else:
            size = int(dones.sum().item())
            self.side[dones] = torch.randint(0, 2, size=(size, ), dtype=torch.int8, device=self.device)
            mjw.reset_data(self.model_d, self.data_d, reset=wp.from_torch(dones))
            env_idx = dones

        ball_vel = wp.to_torch(self.data_d.qvel)[:, 16:]
        ball_vel[env_idx, 0] = (torch.rand(size, device=self.device) - 0.5) * 0.1
        ball_vel[env_idx, 1] = -torch.rand(size, device=self.device) * 2.0
        # should not advance time
        mjw.forward(self.model_d, self.data_d)

    def get_sim_data(self):
        mjw.get_data_into(result=self.mjd, mjm=self.mjm, d=self.data_d)
