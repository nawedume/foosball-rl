"""Compare the removed reset forward with the original reset sequence."""
import dataclasses
import sys
import unittest
from pathlib import Path
import torch
import warp as wp
import mujoco_warp as mjw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from env import FoosballEnv
from kernels.kernels import advance_reset_seed, prepare_resets, randomize_resets


class OriginalResetEnv(FoosballEnv):
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

            # Preserve the existing full forward passes around reset randomization.
            self._forward()
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResetForwardTests(unittest.TestCase):
    def test_reset_matches_original_over_subsequent_steps(self):
        for use_graph in (False, True):
            old = OriginalResetEnv(num_envs=32, use_cuda_graph=use_graph)
            new = FoosballEnv(num_envs=32, use_cuda_graph=use_graph)
            for lesson in ("kick", "final"):
                for resets in (0, 16, 32):
                    with self.subTest(graph=use_graph, lesson=lesson, resets=resets):
                        old.lesson = new.lesson = lesson
                        old.episode_length_buf.zero_()
                        old.episode_length_buf[:resets] = old.max_episode_length
                        wp.synchronize()
                        for field in dataclasses.fields(old.data_d):
                            value = getattr(old.data_d, field.name)
                            if isinstance(value, wp.array):
                                wp.copy(getattr(new.data_d, field.name), value)
                        for name in ("episode_length_buf", "is_blue", "reset_seed"):
                            getattr(new, name).copy_(getattr(old, name))
                        wp.synchronize()
                        for step in range(8):
                            actions = torch.randn((32, 8), device="cuda")
                            expected = old.step(actions)
                            actual = new.step(actions)
                            wp.synchronize()
                            pairs = [(actual[0]["policy"], expected[0]["policy"]),
                                     (actual[1], expected[1]), (actual[2], expected[2]),
                                     (actual[3]["time_outs"], expected[3]["time_outs"])]
                            for field in ("qpos", "qvel", "xpos", "geom_xpos", "sensordata", "qacc", "qacc_warmstart", "ctrl"):
                                pairs.append((wp.to_torch(getattr(new.data_d, field)), wp.to_torch(getattr(old.data_d, field))))
                            for field in ("is_blue", "episode_length_buf", "reset_seed"):
                                pairs.append((getattr(new, field), getattr(old, field)))
                            for key in expected[3]["log"]:
                                pairs.append((actual[3]["log"][key], expected[3]["log"][key]))
                            for pair_index, (got, want) in enumerate(pairs):
                                # GPU solver reductions also vary in old-vs-old controls.
                                # Acceleration fields amplify that rounding; keep observable
                                # state checks substantially tighter than solver diagnostics.
                                acceleration = pair_index in (9, 10)
                                torch.testing.assert_close(
                                    got, want, rtol=1e-4 if acceleration else 1e-5,
                                    atol=1e-3 if acceleration else 1e-5,
                                    msg=lambda message: f"step={step}, pair={pair_index}: {message}")
