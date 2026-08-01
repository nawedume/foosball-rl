from warp import record_event
from rsl_rl.env import VecEnv
from dataclasses import dataclass
from tensordict import TensorDict
import torch
import time
import warp as wp

wp.config.enable_mathdx_solver = True

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

        self.qpos_tensor = wp.to_torch(self.data_d.qpos)
        self.qvel_tensor = wp.to_torch(self.data_d.qvel)
        
        # A single fixed block of memory we overwrite on every step
        self.obs_buf = torch.zeros((self.num_envs, 39), dtype=torch.float32, device=self.device)

        if op_policy is None:
            self.op_policy = NullPolicy(self.num_envs, device=self.device)
        else:
            self.op_policy = op_policy

        self._reset(None)

        mjw.step(self.model_d, self.data_d)
        wp.synchronize()

        # 2. Record the exact decimation sequence into a hardware graph
        wp.capture_begin()
        for _ in range(self.decimation):
            mjw.step(self.model_d, self.data_d)
        self.step_graph = wp.capture_end()

    def get_observations(self) -> TensorDict:
        is_red = self.side == 1
        return self._get_obs(is_red)

    @record_function("observations")
    def _get_obs(self, is_red: torch.Tensor) -> TensorDict:
        self.obs_buf = compute_observations(
                    self.obs_buf,
                    self.qpos_tensor,
                    self.qvel_tensor,
                    is_red,
                    self.red_goal_center,
                    self.blue_goal_center
                )

        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device)

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

        with record_function("Decimation loop"):
            for _ in range(self.decimation):
                with record_function("MJW step"):
                    wp.capture_launch(self.step_graph)

        with record_function("env state and resets"):
            self.episode_length_buf += 1

            # Environment State & Resets
            ball_pos = wp.to_torch(self.data_d.xpos)[:, self.ball_id]
            out_of_bounds = ball_pos[:, 2] <= TERMINATION_HEIGHT

            sensor_data = wp.to_torch(self.data_d.sensordata)
            blue_goals = sensor_data[:, self.blue_goal_sensor_adr] > 0.5
            red_goals = sensor_data[:, self.red_goal_sensor_adr] > 0.5

            ball_vel = wp.to_torch(self.data_d.qvel)[:, 16:19]

        with record_function("reward_func"):
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
    def _reset(self, env_ids: torch.Tensor | None = None):
    

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero(as_tuple=False).squeeze(-1)
            
        size = len(env_ids)
        
        # If no environments are done, exit immediately to save GPU cycles
        if size == 0:
            return

        with record_function("manual state scrub"):
            qvel = wp.to_torch(self.data_d.qvel)
            ctrl = wp.to_torch(self.data_d.ctrl)

            # Scrub momentum and forces ONLY for the envs that finished
            qvel[env_ids, :16] = 0.0
            qvel[env_ids, 18] = 0.0
            ctrl[env_ids, :16] = 0.0

        with record_function("set sides done"):
            new_side = torch.randint(0, 2, size=(size,), dtype=torch.int8, device=self.device)
            if self.always_blue:
                new_side.zero_()
            self.side[env_ids] = new_side

        with record_function("ball vel setup"):
            bias = 0.0 if self.bias_to_blue else -0.5
            scale = 0.3 if self.bias_to_blue else 0.1

            qvel[env_ids, 16] = (torch.rand(size, device=self.device) + bias) * scale
            qvel[env_ids, 17] = -torch.rand(size, device=self.device) * 2.0

        with record_function("randomize ball starting pos"):
            qpos = wp.to_torch(self.data_d.qpos)
            
            # 1. Get the default starting positions from the XML
            qpos0 = wp.to_torch(self.model_d.qpos0)
            
            # 2. Hard-reset the ball's X, Y, and Z to the table center FIRST
            # Notice the `0,` here so we slice the coordinates, not the batch!
            qpos[env_ids, 16:19] = qpos0[0, 16:19]
            
            # 3. Now generate and apply the random scatter
            pos_offset_x = ((torch.rand((size, 1), device=self.device) - 0.5) * 2.0) * 0.4
            pos_offset_y = ((torch.rand((size, 1), device=self.device) - 0.5) * 2.0) * 0.3

            # Apply scatter to the freshly centered ball
            qpos[env_ids, 16:17] += pos_offset_x
            qpos[env_ids, 17:18] += pos_offset_y
        
        with record_function("joint ranges"):
            joint_ranges = wp.to_torch(self.model_d.jnt_range)[0, :16, :]
            min_range = joint_ranges[:, 0]
            max_range = joint_ranges[:, 1]

            joint_positions = torch.rand((size, 16), device=self.device) * (max_range - min_range) + min_range
            qpos[env_ids, :16] = joint_positions

    def get_sim_data(self):
        wp.synchronize()
        mjw.get_data_into(result=self.mjd, mjm=self.mjm, d=self.data_d)


class NullPolicy:
    def __init__(self, num_envs, device='cuda:0'):
        self.device =  device
        self.num_envs = num_envs

    def __call__(self, obs):
        return torch.zeros((self.num_envs, 8), device=self.device)

@torch.compile(mode="reduce-overhead")
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
    
    rewards = torch.zeros(num_envs, device=ball_pos.device)
    
    # ZERO-SYNC: Replace boolean assignments with mathematical masking
    goal_rewards_vals = torch.where(in_right_goal, 1.5 * goal_reward, -goal_reward)
    rewards = torch.where(in_goal, goal_rewards_vals, rewards)

    is_blue = (side == 0).unsqueeze(1)

    dist_vec = torch.where(is_blue, ball_pos - red_goal_center, ball_pos - blue_goal_center)
    dist = torch.linalg.vector_norm(dist_vec, dim=1)

    target_dir = dist_vec / dist.unsqueeze(1).clamp_min(1e-6)
    target_dir = target_dir[:, :2]
    vel_reward = torch.linalg.vecdot(ball_vel[:, :2], -target_dir[:, :2], dim=1) * 0.28

    rewards += vel_reward 

    # Out of Bounds Penalty (ZERO-SYNC)
    rewards -= out_of_bounds.to(torch.float) * 300.0
    
    return rewards

@torch.compile()
def compute_observations(
    obs_buf: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    is_red: torch.Tensor,
    red_goal_center: torch.Tensor,
    blue_goal_center: torch.Tensor,
) -> torch.Tensor:
    
    is_red_mask = is_red.unsqueeze(1)
    
    # ZERO-SYNC: Vectorized sign multiplier replaces boolean indexing
    sign = torch.where(is_red_mask, -1.0, 1.0)

    # 1. PLAYER & OPPONENT ROD STATES
    obs_buf[:, :8] = torch.where(is_red_mask, qpos[:, 8:16], qpos[:, :8])
    obs_buf[:, 8:16] = torch.where(is_red_mask, qvel[:, 8:16], qvel[:, :8])

    obs_buf[:, 16:24] = torch.where(is_red_mask, qpos[:, :8], qpos[:, 8:16])
    obs_buf[:, 24:32] = torch.where(is_red_mask, qvel[:, :8], qvel[:, 8:16])

    # 2. BALL POS & VELOCITY
    obs_buf[:, 32:35] = torch.where(
        is_red_mask,
        qpos[:, 16:19] - red_goal_center,
        qpos[:, 16:19] - blue_goal_center
    )
    
    obs_buf[:, 32:34] *= sign 

    obs_buf[:, 35:38] = qvel[:, 16:19]
    obs_buf[:, 35:37] *= sign 

    # 3. SIDE MARKER
    obs_buf[:, 38] = is_red.to(torch.float)

    return obs_buf