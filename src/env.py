from ctypes import alignment
from numpy.core.multiarray import dtype
from rsl_rl.env import VecEnv
from dataclasses import dataclass
from tensordict import TensorDict
import torch
import time
import warp as wp
import numpy as np

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer

TERMINATION_HEIGHT = 0.15

GOAL_CENTER_BLUE = [0.563284, 0.0, 0.289419]
GOAL_CENTER_RED = [-0.550284, 0.0, 0.289419]

@wp.kernel
def set_random_ball_position_to_player(
    envs: wp.array[wp.int32],
    ball_idx: wp.int32,
    seed: wp.int32,
    step: wp.int32,
    is_blue: wp.array[wp.bool],
    ball_start_pos: wp.array[wp.vec3], # size of number of sites
    ball_start_line_length: wp.float32,
    out_xpos: wp.array2d[wp.float32],
):
    env_idx = envs[wp.tid()]

    num_sites = wp.uint32(ball_start_pos.shape[0])

    rng = wp.rand_init(seed + step, env_idx)
    line_idx = wp.randu(rng, wp.uint32(0), num_sites)
    t = wp.randf(rng)

    # y because thats the axis of the rod
    pos = ball_start_pos[line_idx]
    pos[1] -= t*ball_start_line_length
    # go towards the opposite end (blue aims toward the -x)
    pos[0] += wp.randf(rng) * 0.01 * (-1.0 if is_blue[env_idx] else 1.0)

    out_xpos[env_idx, ball_idx + 0] = pos[0]
    out_xpos[env_idx, ball_idx + 1] = pos[1]
    out_xpos[env_idx, ball_idx + 2] = pos[2]


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
        lesson: str = "kick",
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = wp.int32(seed)

        super().__init__()

        self.step_count = 0

        self.num_envs = num_envs
        self.sync_with_viewer = sync_with_viewer
        self.always_blue = always_blue
        self.bias_to_blue = bias_to_blue
        self.lesson = lesson

        self.num_actions = 8
        self.max_episode_length = int(60 / dt) // 6 # 10 seconds
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.decimation = 1
        self.action_scale = 30.0

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
        self.ball_qpos_adr = self.mjm.body_jntadr[self.ball_id]
        self.ball_qvel_adr = self.mjm.body_dofadr[self.ball_id]
        assert self.mjm.body_jntnum[self.ball_id] == 1
        assert self.mjm.body_dofnum[self.ball_id] == 6

        blue_ball_start_sites = [
            self.mjm.site("blue_1_1_line_start").id,
            self.mjm.site("blue_1_2_line_start").id,
            self.mjm.site("blue_2_line_start").id,
            self.mjm.site("blue_3_line_start").id,
        ]

        self.ball_line_start = wp.indexedarray(
            self.model_d.site_pos,
            indices=[None, wp.array(blue_ball_start_sites, dtype=wp.int32)]
        ).contiguous()[0, :]

        self.ball_line_length = wp.float32(0.54)

        blue_goal_sensor_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SENSOR, "blue_goal_reached")
        self.blue_goal_sensor_adr = self.mjm.sensor_adr[blue_goal_sensor_id]

        red_goal_sensor_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SENSOR, "red_goal_reached")
        self.red_goal_sensor_adr = self.mjm.sensor_adr[red_goal_sensor_id]

        self.blue_goal_center = torch.tensor(GOAL_CENTER_BLUE, device=self.device).unsqueeze(0)
        self.red_goal_center = torch.tensor(GOAL_CENTER_RED, device=self.device).unsqueeze(0)

        self.is_blue = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        self.goal_reward = 300.0
        self.oob_penalty = -300.0

        # get the blue and red players for resetting purposes (only matters for some cirriculums)
        # There's probably a better way of handling this
        blue_geom_names = [
            "b1abottom0","b1bbottom0","b1cbottom0","b1abottom1","b1bbottom1","b1cbottom1","b2abottom","b2bbottom","b3abottom","b3bbottom","b3cbottom","b3dbottom","b3ebottom",
        ]
        red_geom_names = [
            "r1abottom0","r1bbottom0","r1cbottom0","r1abottom1","r1bbottom1","r1cbottom1","r2abottom","r2bbottom","r3abottom","r3bbottom","r3cbottom","r3dbottom","r3ebottom",
        ]
        self.blue_geom_ids = wp.array([self.mjm.geom(n).id  for n in blue_geom_names], device=device, dtype=wp.uint16)
        self.red_geom_ids = wp.array([self.mjm.geom(n).id  for n in red_geom_names], device=device, dtype=wp.uint16)

        assert self.blue_geom_ids.shape == self.red_geom_ids.shape
        assert self.blue_geom_ids.shape[0] == 13

        if op_policy is None:
            self.op_policy = NullPolicy(self.num_envs, device=self.device)
        else:
            self.op_policy = op_policy

        self._reset(None)

    def get_observations(self) -> TensorDict:
        is_red = ~self.is_blue
        return self._get_obs(is_red)

    def _get_obs(self, is_red: torch.Tensor) -> TensorDict:
        q_pos = wp.to_torch(self.data_d.qpos).clone()
        q_vel = wp.to_torch(self.data_d.qvel).clone()

        # Relative ball positions and velocities
        ball_pos_rel = torch.tensor(self.data_d.xpos[:, self.ball_id]).clone()


        ball_pos_rel = torch.where(
            is_red.unsqueeze(1),
            ball_pos_rel - self.red_goal_center,
            ball_pos_rel - self.blue_goal_center
        )

        # Mirror BOTH X and Y axes for Red so left/right and forward/backward match perspective
        ball_pos_rel[is_red, 0] = -ball_pos_rel[is_red, 0]
        ball_pos_rel[is_red, 1] = -ball_pos_rel[is_red, 1]

        ball_vel_rel = q_vel[:, self.ball_qvel_adr:(self.ball_qvel_adr + 3)]
        ball_vel_rel[is_red, 0] = -ball_vel_rel[is_red, 0]
        ball_vel_rel[is_red, 1] = -ball_vel_rel[is_red, 1]

        # Assuming that the rods are the first 16 entires, this should be true based on the moedl
        # Rod states
        blue_pos = q_pos[:, :8]
        blue_vel = q_vel[:, :8]
        blue_state = torch.cat([blue_pos, blue_vel], dim=-1)

        red_pos = q_pos[:, 8:16]
        red_vel = q_vel[:, 8:16]
        red_state = torch.cat([red_pos, red_vel], dim=-1)

        player_rod_state = torch.where(is_red.unsqueeze(1), red_state, blue_state)
        op_rod_state = torch.where(is_red.unsqueeze(1), blue_state, red_state)

        data = torch.cat([player_rod_state, op_rod_state, ball_pos_rel, ball_vel_rel, is_red.to(torch.float).unsqueeze(1)], dim=1)

        ret = {"policy": data}
        td = TensorDict(ret, batch_size=[self.num_envs], device=self.device) # ty: ignore
        return td


    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.step_count += 1
        # 1. Action Clamping & Scaling
        raw_actions = torch.clamp(actions, min=-1.0, max=1.0)
        scaled_actions = raw_actions * self.action_scale

        control = wp.to_torch(self.data_d.ctrl)

        is_red = ~self.is_blue
        control.zero_()

        control[is_red, 8:16] = scaled_actions[is_red]
        control[~is_red, :8] = scaled_actions[~is_red]

        raw_op_actions = torch.clamp(self.op_policy(self._get_obs(self.is_blue)), min=-1.0, max=1.0)
        op_actions = raw_op_actions * self.action_scale
        control[is_red, :8] =  op_actions[is_red]
        control[~is_red, 8:16] = op_actions[~is_red]

        for _ in range(self.decimation):
            mjw.step(self.model_d, self.data_d)

        mjw.forward(self.model_d, self.data_d)

        self.episode_length_buf += 1

        # Environment State & Resets
        ball_pos = wp.to_torch(self.data_d.xpos)[:, self.ball_id]
        out_of_bounds = ball_pos[:, 2] <= TERMINATION_HEIGHT

        sensor_data = wp.to_torch(self.data_d.sensordata)
        blue_goals = sensor_data[:, self.blue_goal_sensor_adr] > 0.5
        red_goals = sensor_data[:, self.red_goal_sensor_adr] > 0.5

        ball_vel = wp.to_torch(self.data_d.qvel)[:, self.ball_qvel_adr:(self.ball_qvel_adr + 3)]
        blue_player_pos = self.data_d.xpos[:, self.blue_geom_ids]
        red_player_pos = self.data_d.xpos[:, self.red_geom_ids]

        rewards, r_goal, r_vel, r_dist, r_alignment, r_oob = compute_rewards(
            blue_goals,
            red_goals,
            self.is_blue,
            ball_pos,
            ball_vel,
            self.blue_goal_center,
            self.red_goal_center,
            blue_player_pos,
            red_player_pos,
            out_of_bounds,
            self.goal_reward,
            self.oob_penalty,
            self.num_envs,
        )

        time_outs = (self.episode_length_buf > self.max_episode_length).bool()
        dones =  time_outs | out_of_bounds | blue_goals | red_goals
        self.episode_length_buf[dones] = 0
        if dones.any():
            self._reset(dones)

        obs = self.get_observations()

        if self.sync_with_viewer:
            self.get_sim_data()

        extras = {
            'time_outs': time_outs,
            'log': {
                'Metrics/goal': r_goal,
                'Metrics/velocity': r_vel,
                'Metrics/distance': r_dist,
                'Metrics/alignment': r_alignment,
                'Metrics/oob_rate': r_oob,
            }
        }
        return obs, rewards, dones, extras

    def _reset(self, dones: torch.Tensor | None = None):
        # update the seed so that we can a new random number each time the kernel is launched
        # Might need something more robust?
        self.seed += 1

        if dones is None:
            env_idx = slice(None)
            self.is_blue[env_idx] = torch.randint(0, 2, size=(self.num_envs,), dtype=torch.int8, device=self.device) == 0
            size = self.num_envs
        else:
            size = int(dones.sum().item())
            self.is_blue[dones] = torch.randint(0, 2, size=(size,), dtype=torch.int8, device=self.device) == 0
            mjw.reset_data(self.model_d, self.data_d, reset=wp.from_torch(dones))
            env_idx = torch.where(dones)[0].to(torch.int32)

        if self.always_blue:
            self.is_blue[env_idx] = True

        # must propagate changes in the position so the geoms are updated by relative position
        mjw.forward(self.model_d, self.data_d)

        if self.lesson == "kick":
            # spawn the ball in the line of a kicker pole
            out_ball_positions = self.data_d.qpos
            ball_id = wp.int32(self.ball_qpos_adr)
            side_wp = wp.from_torch(self.is_blue, dtype=wp.bool)

            if env_idx == slice(None):
                env_wp = wp.array([i for i in range(size)], dtype=wp.int32)
            else:
                env_wp = wp.array(env_idx, dtype=wp.int32)

            wp.launch(set_random_ball_position_to_player, dim=(size, ),
                inputs=[
                    env_wp,
                    ball_id,
                    self.seed,
                    wp.uint32(self.step_count),
                    side_wp,
                    self.ball_line_start,
                    self.ball_line_length,
                ],
                outputs=[
                    out_ball_positions
                ]
            )

        elif self.lesson == "final":
            qvel = wp.to_torch(self.data_d.qvel)
            ball_vel = qvel[:, 16:19]

            # always throw to blue
            bias = 1.0 if self.bias_to_blue else 0.0

            ball_vel[env_idx, 0] = (torch.rand(size, device=self.device) + bias)
            ball_vel[env_idx, 1] = -torch.rand(size, device=self.device) * 2.0

            # randomize ball starting position so the fooseball learns to kick right
            qpos[env_idx, 16:17] += ((torch.rand((size, 1), device=self.device) -  0.5) * 2.0) * 0.4 # from +- 0.4
            qpos[env_idx, 17:18] += ((torch.rand((size, 1), device=self.device) -  0.5) * 2.0) * 0.3 # from +- 0.3

        else:
            raise RuntimeError(f"Lesson {self.lesson} not defined.")

        # The kick lesson will place on  the "initial" positions, but afterwards we will randomize the pole
        # positions, maybe not ideal since we still need to learn how to get to the ball and not just kick
        # forward
        qpos = wp.to_torch(self.data_d.qpos)
        joint_ranges = wp.to_torch(self.model_d.jnt_range)[0, :16, :]
        min_range = joint_ranges[:, 0]
        max_range = joint_ranges[:, 1]
        joint_positions = torch.rand((size, 16), device=self.device) * (max_range - min_range) + min_range
        qpos[env_idx, :16] = joint_positions

        # Need to propagate the changes in in positions
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


@torch.compile
def compute_rewards(
    blue_goals: torch.Tensor,
    red_goals: torch.Tensor,
    is_blue: torch.Tensor,
    ball_pos,
    ball_vel,
    blue_goal_center,
    red_goal_center,
    blue_player_pos, #  n_envs x n_players x 2
    red_player_pos, # n_envs x n_players x 2
    out_of_bounds,
    goal_reward: float,
    oob_penalty: float,
    num_envs: int,
):

    ## Goal Rewards
    in_goal = blue_goals | red_goals
    in_right_goal = (is_blue & red_goals) | (~is_blue & blue_goals)
    rewards = torch.zeros(num_envs, device=ball_pos.device) # pyright: ignore

    g_rewards= torch.where(in_right_goal[in_goal], 1.5*goal_reward, -goal_reward)
    rewards[in_goal] = g_rewards

    dist_vec = torch.where(is_blue.unsqueeze(1), ball_pos - red_goal_center, ball_pos - blue_goal_center)
    dist = torch.linalg.vector_norm(dist_vec, dim=1)
    dist_reward = -dist * 2.0

    target_dir = -dist_vec[:, :2]
    target_dir = target_dir / torch.linalg.vector_norm(target_dir, dim=1).unsqueeze(1).clamp_min(1e-6)
    vel_reward = torch.linalg.vecdot(ball_vel[:, :2], target_dir, dim=1) * 2.0

    # alignment reward
    player_pos = torch.where(is_blue, blue_player_pos, red_player_pos)
    # Get the minimum distance between a player on your side and the ball, this favors alignment
    alignment_reward = 2.0 * torch.min(torch.linalg.norm(ball_pos[:, :2].unsqeeze(1) - player_pos, dim=2), dim=1).values

    # dist reward is negative distance, so closer to the goal the better
    rewards += vel_reward + dist_reward + alignment_reward

    # Out of Bounds Penalty
    oob_rewards = out_of_bounds.float() * oob_penalty
    rewards[out_of_bounds] += oob_rewards[out_of_bounds]
    return rewards, g_rewards, vel_reward, dist_reward, alignment_reward, oob_rewards
