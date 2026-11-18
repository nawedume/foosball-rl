# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# OnPolicyRunner.learn is adapted from rsl-rl-lib 5.4.2.
# PPO itself is provided by upstream RSL-RL without an override.

"""Repository-owned RSL-RL integration with batched diagnostic readbacks."""

import copy
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger


class DeferredEpisodeLogger(Logger):
    """Preserve chronological episode statistics with one transfer per rollout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_episodes = []

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        # Preserve upstream handling for optional intrinsic-reward logging.
        if self.cfg["algorithm"]["rnd_cfg"] or intrinsic_rewards is not None:
            return super().process_env_step(rewards, dones, extras, intrinsic_rewards)
        if self.writer is None:
            return
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        finished = dones > 0
        self._pending_episodes.append(
            torch.stack(
                [
                    finished.to(self.cur_reward_sum.dtype),
                    self.cur_reward_sum,
                    self.cur_episode_length,
                ],
                dim=-1,
            )
        )
        self.cur_reward_sum.masked_fill_(finished, 0)
        self.cur_episode_length.masked_fill_(finished, 0)

    def flush_pending(self):
        if not self._pending_episodes:
            return
        # Stack copies values before clearing buffers. Filtering happens on CPU
        # in step/environment order, matching upstream's nonzero ordering.
        pending = torch.stack(self._pending_episodes).cpu().numpy()
        self._pending_episodes.clear()
        episodes = pending[pending[..., 0].astype(bool)]
        self.rewbuffer.extend(episodes[:, 1].tolist())
        self.lenbuffer.extend(episodes[:, 2].tolist())


class LocalOnPolicyRunner(OnPolicyRunner):
    """Batch environment diagnostics and reporting at rollout boundaries."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = copy.deepcopy(train_cfg)
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self.logger = DeferredEpisodeLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

    def _accumulate_nan(self, flags, obs, rewards, dones):
        for tensor in obs.values():
            flags[0].logical_or_(torch.isnan(tensor).any())
        flags[1].logical_or_(torch.isnan(rewards).any())
        flags[2].logical_or_(torch.isnan(dones).any())

    def _validate_rollout(self, flags):
        if self.is_distributed:
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        if flags.any().item():
            raise ValueError(
                "The environment returned NaN during the rollout; PPO update was skipped."
            )

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            nan_flags = torch.zeros(3, dtype=torch.bool, device=self.device)
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        self._accumulate_nan(nan_flags, obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    # Book keeping
                    self.logger.process_env_step(
                        rewards, dones, extras, intrinsic_rewards
                    )

                if self.cfg.get("check_for_nan", True):
                    self._validate_rollout(nan_flags)
                # Include deferred reporting in collection time, as upstream does.
                self.logger.flush_pending()
                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight
                if self.cfg["algorithm"]["rnd_cfg"]
                else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )  # type: ignore
            self.logger.stop_logging_writer()
