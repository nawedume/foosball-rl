"""Regression checks for environment graph replay and buffer ownership."""

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import warp as wp

from env import FoosballEnv
from kernels.kernels import assign_controls


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class EnvironmentGraphTests(unittest.TestCase):
    def test_controls_match_torch(self):
        wp.init()
        torch.manual_seed(19)
        count = 32
        blue = torch.rand(count, device="cuda") > 0.5
        actions = torch.randn((count, 8), device="cuda") * 2
        opponent = torch.randn_like(actions) * 2
        actions[0, 0] = float("nan")
        actions[1, 0] = float("inf")
        player_scaled = actions.clamp(-1, 1) * 30
        opponent_scaled = opponent.clamp(-1, 1) * 30
        expected = torch.empty((count, 16), device="cuda")
        expected[blue, :8] = player_scaled[blue]
        expected[~blue, 8:] = player_scaled[~blue]
        expected[blue, 8:] = opponent_scaled[blue]
        expected[~blue, :8] = opponent_scaled[~blue]
        result = torch.empty_like(expected)
        wp.launch(
            assign_controls,
            dim=(count, 16),
            inputs=[
                wp.from_torch(actions),
                wp.from_torch(opponent),
                wp.from_torch(blue),
                30.0,
                30.0,
                False,
                False,
                False,
            ],
            outputs=[wp.from_torch(result)],
        )
        wp.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)

    def test_graph_matches_eager_from_snapshot(self):
        count = 32
        graph = FoosballEnv(num_envs=count, always_blue=False)
        eager = FoosballEnv(num_envs=count, always_blue=False, use_cuda_graph=False)
        actions = torch.randn((count, 8), device="cuda")
        graph.step(actions)
        for lesson in ("kick", "final"):
            for resets in (0, count // 2, count):
                with self.subTest(lesson=lesson, resets=resets):
                    graph.lesson = eager.lesson = lesson
                    eager.episode_length_buf.zero_()
                    eager.episode_length_buf[:resets] = eager.max_episode_length
                    wp.synchronize()
                    for field in dataclasses.fields(eager.data_d):
                        value = getattr(eager.data_d, field.name)
                        if isinstance(value, wp.array):
                            wp.copy(getattr(graph.data_d, field.name), value)
                    for name in ("episode_length_buf", "is_blue", "reset_seed"):
                        getattr(graph, name).copy_(getattr(eager, name))
                    wp.synchronize()
                    expected = eager.step(actions)
                    actual = graph.step(actions)
                    wp.synchronize()
                    torch.testing.assert_close(
                        actual[0]["policy"], expected[0]["policy"]
                    )
                    torch.testing.assert_close(actual[1], expected[1])
                    torch.testing.assert_close(actual[2], expected[2])

        previous = actual[0]["policy"].clone()
        saved = actual[0]["policy"]
        graph.episode_length_buf = torch.full_like(
            graph.episode_length_buf, graph.max_episode_length
        )
        caller = torch.cuda.Stream()
        with torch.cuda.stream(caller):
            _, reward, done, _ = graph.step(torch.zeros_like(actions))
        caller.synchronize()
        self.assertTrue(done.all().item())
        self.assertTrue(torch.isfinite(reward).all().item())
        torch.testing.assert_close(saved, previous, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
