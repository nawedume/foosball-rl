from contextlib import nullcontext
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

from kernels.kernels import (
    advance_reset_seed,
    assign_controls,
    compute_observations,
    compute_rewards,
    gather_step_state,
    invert_sides,
    prepare_resets,
    randomize_resets,
)

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
        lesson: str = "kick",
        seed: int = 42,
        use_cuda_graph: bool = True,
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
        self.reset_seed = torch.tensor([seed], dtype=torch.int32, device=device)
        self._all_reset = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.decimation = 1
        self.action_scale = 30.0

        self.device = device
        self.use_cuda_graph = use_cuda_graph and str(device).startswith("cuda")
        self.step_graph = None
        self._step_signature = None
        self._building_step_graph = False
        self.forward_graph = None
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
        self.ball_qpos_adr = self.mjm.jnt_qposadr[self.mjm.body_jntadr[self.ball_id]]
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
        self.blue_goal_wp = wp.vec3(*GOAL_CENTER_BLUE)
        self.red_goal_wp = wp.vec3(*GOAL_CENTER_RED)

        self.is_blue = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        self.goal_reward = 3000.0
        self.oob_penalty = -300.0

        # get the blue and red players for resetting purposes (only matters for some cirriculums)
        # There's probably a better way of handling this
        blue_geom_names = [
            "b1abottom0","b1bbottom0","b1cbottom0","b1abottom1","b1bbottom1","b1cbottom1","b2abottom","b2bbottom","b3abottom","b3bbottom","b3cbottom","b3dbottom","b3ebottom",
        ]
        red_geom_names = [
            "r1abottom0","r1bbottom0","r1cbottom0","r1abottom1","r1bbottom1","r1cbottom1","r2abottom","r2bbottom","r3abottom","r3bbottom","r3cbottom","r3dbottom","r3ebottom",
        ]
        self.blue_geom_ids = wp.array([self.mjm.geom(n).id  for n in blue_geom_names], device=device, dtype=wp.int32)
        self.red_geom_ids = wp.array([self.mjm.geom(n).id  for n in red_geom_names], device=device, dtype=wp.int32)
        assert self.blue_geom_ids.shape == self.red_geom_ids.shape
        assert self.blue_geom_ids.shape[0] == 13

        blue_rod_geom_names = [
            "blue_rod_10",
            "blue_rod_11",
            "blue_rod_2",
            "blue_rod_3",
        ]
        red_rod_geom_names = [
            "red_rod_10",
            "red_rod_11",
            "red_rod_2",
            "red_rod_3",
        ]
        blue_rod_gids = wp.array([self.mjm.geom(n).id for n in blue_rod_geom_names], device=device, dtype=wp.int32)
        red_rod_gids = wp.array([self.mjm.geom(n).id for n in red_rod_geom_names], device=device, dtype=wp.int32)
        assert blue_rod_gids.shape == red_rod_gids.shape
        assert blue_rod_gids.shape[0] == 4

        print(self.model_d.geom_pos.shape)
        self.blue_rod_pos = self.data_d.geom_xpos[0][blue_rod_gids].contiguous()
        self.red_rod_pos =  self.data_d.geom_xpos[0][red_rod_gids].contiguous()

        if op_policy is None:
            self.op_policy = NullPolicy(self.num_envs, device=self.device)
        else:
            self.op_policy = op_policy

        self.rewards = wp.zeros((self.num_envs, 6), device=self.device)

        self._reset(None)

        if self.use_cuda_graph:
            with wp.ScopedCapture(device=self.device) as capture:
                mjw.forward(self.model_d, self.data_d)
            self.forward_graph = capture.graph

        # Stable inputs and outputs for replay of the complete environment step.
        self._actions = torch.empty((num_envs, 8), dtype=torch.float32, device=device)
        self._opponent_actions = torch.empty_like(self._actions)
        self._step_observations = wp.empty((num_envs, 38), device=device)
        self._step_ball_pos = wp.empty(num_envs, dtype=wp.vec3, device=device)
        self._step_ball_vel = wp.empty(num_envs, dtype=wp.vec3, device=device)
        self._step_blue_players = wp.empty((num_envs, 13), dtype=wp.vec3, device=device)
        self._step_red_players = wp.empty((num_envs, 13), dtype=wp.vec3, device=device)
        self._step_blue_goals = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_red_goals = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_oob = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_time_outs = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_dones = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_is_red = wp.empty(num_envs, dtype=wp.bool, device=device)
        self._step_torch_stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None
        self._step_stream = wp.stream_from_torch(self._step_torch_stream) if self._step_torch_stream is not None else None
        if self._step_stream is not None:
            # Initialization runs on Warp's default stream before the step stream exists.
            wp.synchronize_device(device)

    def get_observations(self) -> TensorDict:
        is_red = ~self.is_blue
        return self._get_obs(is_red)

    def _get_obs(self, is_red: torch.Tensor) -> TensorDict:
        # A fresh fixed-size output preserves observations retained by callers.
        observations = torch.empty((self.num_envs, 38), dtype=torch.float32, device=self.device)
        wp.launch(
            compute_observations,
            dim=(self.num_envs, 38),
            inputs=[
                self.data_d.qpos,
                self.data_d.qvel,
                self.data_d.xpos,
                wp.from_torch(is_red),
                wp.from_torch(self.blue_goal_center),
                wp.from_torch(self.red_goal_center),
                self.ball_id,
                int(self.ball_qvel_adr),
            ],
            outputs=[wp.from_torch(observations)],
        )
        return TensorDict(
            {"policy": observations},
            batch_size=[self.num_envs],
            device=self.device,
        )


    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.step_count += 1
        if actions.shape != (self.num_envs, 8):
            raise ValueError(
                f"Expected actions shaped {(self.num_envs, 8)}, got {tuple(actions.shape)}"
            )
        null_opponent = type(self.op_policy) is NullPolicy
        actions_scaled = actions.dtype != torch.float32
        opponent_scaled = False
        caller_stream = None
        if self._step_torch_stream is not None:
            caller_stream = torch.cuda.current_stream(self.device)
            self._step_torch_stream.wait_stream(caller_stream)
        torch_scope = (
            torch.cuda.stream(self._step_torch_stream)
            if caller_stream is not None
            else nullcontext()
        )
        warp_scope = (
            wp.ScopedStream(self._step_stream)
            if caller_stream is not None
            else nullcontext()
        )
        with torch_scope, warp_scope:
            player_input = (
                torch.clamp(actions, -1.0, 1.0) * self.action_scale
                if actions_scaled
                else actions
            )
            self._actions.copy_(player_input)
            if caller_stream is not None and actions.is_cuda:
                actions.record_stream(self._step_torch_stream)
            if not null_opponent:
                opponent = self.op_policy(self._get_obs(self.is_blue))
                if opponent.shape != (self.num_envs, 8):
                    raise ValueError("Opponent actions must have shape (num_envs, 8)")
                opponent_scaled = opponent.dtype != torch.float32
                opponent_input = (
                    torch.clamp(opponent, -1.0, 1.0) * self.action_scale
                    if opponent_scaled
                    else opponent
                )
                self._opponent_actions.copy_(opponent_input)
            signature = (
                self.decimation,
                self.action_scale,
                self.max_episode_length,
                self.lesson,
                self.always_blue,
                self.bias_to_blue,
                self.goal_reward,
                self.oob_penalty,
                float(self.ball_line_length),
                null_opponent,
                actions_scaled,
                opponent_scaled,
                self.episode_length_buf.data_ptr(),
                self.is_blue.data_ptr(),
                self.reset_seed.data_ptr(),
            )
            if self.use_cuda_graph:
                if self.step_graph is None or self._step_signature != signature:
                    self._building_step_graph = True
                    try:
                        with wp.ScopedCapture(stream=self._step_stream) as capture:
                            self._simulate_step(
                                null_opponent, actions_scaled, opponent_scaled
                            )
                    finally:
                        self._building_step_graph = False
                    self.step_graph = capture.graph
                    self._step_signature = signature
                wp.capture_launch(self.step_graph, stream=self._step_stream)
            else:
                self._simulate_step(null_opponent, actions_scaled, opponent_scaled)
        if caller_stream is not None:
            caller_stream.wait_stream(self._step_torch_stream)

        # Retain independent outputs for rollout storage and external callers.
        rewards_t = wp.to_torch(self.rewards).clone()
        time_outs = wp.to_torch(self._step_time_outs).clone()
        dones = wp.to_torch(self._step_dones).clone()
        observations = wp.to_torch(self._step_observations).clone()
        obs = TensorDict(
            {"policy": observations}, batch_size=[self.num_envs], device=self.device
        )

        if self.sync_with_viewer:
            self.get_sim_data()

        extras = {
            "time_outs": time_outs,
            "log": {
                "Metrics/goal": rewards_t[:, 1],
                "Metrics/oob_rate": rewards_t[:, 2],
                "Metrics/distance": rewards_t[:, 3],
                "Metrics/velocity": rewards_t[:, 4],
                "Metrics/alignment": rewards_t[:, 5],
            },
        }
        return obs, rewards_t[:, 0], dones, extras

    def _simulate_step(self, null_opponent, actions_scaled, opponent_scaled):
        wp.launch(
            assign_controls,
            dim=(self.num_envs, 16),
            inputs=[
                wp.from_torch(self._actions),
                wp.from_torch(self._opponent_actions),
                wp.from_torch(self.is_blue),
                self.action_scale,
                self.action_scale,
                null_opponent,
                actions_scaled,
                opponent_scaled,
            ],
            outputs=[self.data_d.ctrl],
        )
        for _ in range(self.decimation):
            mjw.step(self.model_d, self.data_d)
        mjw.forward(self.model_d, self.data_d)
        wp.launch(
            gather_step_state,
            dim=self.num_envs,
            inputs=[
                self.data_d.xpos,
                self.data_d.geom_xpos,
                self.data_d.qvel,
                self.data_d.sensordata,
                self.blue_geom_ids,
                self.red_geom_ids,
                self.ball_id,
                int(self.ball_qvel_adr),
                int(self.blue_goal_sensor_adr),
                int(self.red_goal_sensor_adr),
                self.max_episode_length,
                TERMINATION_HEIGHT,
            ],
            outputs=[
                wp.from_torch(self.episode_length_buf),
                self._step_ball_pos,
                self._step_ball_vel,
                self._step_blue_players,
                self._step_red_players,
                self._step_blue_goals,
                self._step_red_goals,
                self._step_oob,
                self._step_time_outs,
                self._step_dones,
            ],
        )
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(self.is_blue),
                self._step_blue_goals,
                self._step_red_goals,
                self._step_ball_pos,
                self._step_ball_vel,
                self.blue_goal_wp,
                self.red_goal_wp,
                self._step_oob,
                self.blue_rod_pos,
                self.red_rod_pos,
                self._step_blue_players,
                self._step_red_players,
                self.goal_reward,
                self.oob_penalty,
            ],
            outputs=[self.rewards],
        )
        self._reset(self._step_dones)
        wp.launch(
            invert_sides,
            dim=self.num_envs,
            inputs=[wp.from_torch(self.is_blue)],
            outputs=[self._step_is_red],
        )
        wp.launch(
            compute_observations,
            dim=(self.num_envs, 38),
            inputs=[
                self.data_d.qpos,
                self.data_d.qvel,
                self.data_d.xpos,
                self._step_is_red,
                wp.from_torch(self.blue_goal_center),
                wp.from_torch(self.red_goal_center),
                self.ball_id,
                int(self.ball_qvel_adr),
            ],
            outputs=[self._step_observations],
        )

    def _reset(self, dones: torch.Tensor | None = None):
        if self.lesson not in ("kick", "final"):
            raise RuntimeError(f"Lesson {self.lesson} not defined.")
        mask = self._all_reset if dones is None else dones
        mask_wp = wp.from_torch(mask) if isinstance(mask, torch.Tensor) else mask
        wp.launch(
            advance_reset_seed,
            dim=1,
            inputs=[mask_wp],
            outputs=[wp.from_torch(self.reset_seed)],
        )
        if dones is not None:
            mjw.reset_data(self.model_d, self.data_d, reset=mask_wp)
        wp.launch(
            prepare_resets,
            dim=self.num_envs,
            inputs=[mask_wp, wp.from_torch(self.reset_seed), self.always_blue],
            outputs=[wp.from_torch(self.is_blue), wp.from_torch(self.episode_length_buf)],
        )

        # Randomization reads only primary state and model constants.
        # Refresh derived state once, after writing the randomized state.
        wp.launch(
            randomize_resets,
            dim=self.num_envs,
            inputs=[
                mask_wp,
                wp.from_torch(self.reset_seed),
                wp.from_torch(self.is_blue),
                self.lesson == "kick",
                self.bias_to_blue,
                int(self.ball_qpos_adr),
                int(self.ball_qvel_adr),
                self.ball_line_start,
                self.ball_line_length,
                self.model_d.jnt_range,
            ],
            outputs=[self.data_d.qpos, self.data_d.qvel],
        )
        self._forward()

    def _forward(self):
        if self._building_step_graph or self.forward_graph is None or not self.use_cuda_graph:
            mjw.forward(self.model_d, self.data_d)
        else:
            wp.capture_launch(self.forward_graph)

    def get_sim_data(self):
        wp.synchronize()
        mjw.get_data_into(result=self.mjd, mjm=self.mjm, d=self.data_d)


class NullPolicy:
    def __init__(self, num_envs, device='cuda:0'):
        self.device =  device
        self.num_envs = num_envs
    def __call__(self, obs):
        return torch.zeros((self.num_envs, 8), device=self.device)

