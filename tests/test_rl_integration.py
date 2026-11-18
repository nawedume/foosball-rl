import copy
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger
from rl_integration import DeferredEpisodeLogger, LocalOnPolicyRunner
from rsl_rl.algorithms import PPO
from policy_cfg import train_cfg

torch.set_num_threads(1)


class ToyEnv:
    num_envs = 4
    num_actions = 2
    max_episode_length = 10
    device = "cpu"
    cfg = {}

    def __init__(self, nan_rewards=False):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.nan_rewards = nan_rewards
        self.steps = 0

    def get_observations(self):
        return TensorDict(
            {"policy": torch.zeros(self.num_envs, 3)}, batch_size=[self.num_envs]
        )

    def step(self, actions):
        self.steps += 1
        self.episode_length_buf += 1
        rewards = actions.square().sum(-1)
        if self.nan_rewards and self.steps == 2:
            rewards[0] = float("nan")
        return self.get_observations(), rewards, self.episode_length_buf >= 10, {}


def config():
    cfg = copy.deepcopy(train_cfg)
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"]["num_learning_epochs"] = 2
    cfg["algorithm"]["num_mini_batches"] = 2
    cfg["actor"]["hidden_dims"] = [16, 8]
    cfg["critic"]["hidden_dims"] = [16, 8]
    cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
    return cfg


class IntegrationTests(unittest.TestCase):
    def test_episode_statistics_match_upstream(self):
        for rnd in (False, True):
            cfg = {"algorithm": {"rnd_cfg": {} if rnd else None}}
            args = dict(
                log_dir=None,
                cfg=cfg,
                env_cfg={},
                num_envs=4,
                is_distributed=False,
                gpu_world_size=1,
                gpu_global_rank=0,
                device="cpu",
            )
            # Logger's RND configuration is truth-tested.
            if rnd:
                args["cfg"]["algorithm"]["rnd_cfg"] = {"enabled": True}
            original = Logger(**args)
            batched = DeferredEpisodeLogger(**args)
            original.writer = batched.writer = object()
            torch.manual_seed(7)
            for step in range(80):
                rewards = torch.randn(4)
                done = torch.rand(4) > 0.4
                intrinsic = torch.rand(4) if rnd else None
                original.process_env_step(rewards, done, {}, intrinsic)
                batched.process_env_step(rewards, done, {}, intrinsic)
                if step % 7 == 0:
                    batched.flush_pending()
                    self.assertEqual(list(original.rewbuffer), list(batched.rewbuffer))
                    self.assertEqual(list(original.lenbuffer), list(batched.lenbuffer))
            batched.flush_pending()
            for name in ("rewbuffer", "lenbuffer") + (
                ("erewbuffer", "irewbuffer") if rnd else ()
            ):
                self.assertEqual(
                    list(getattr(original, name)), list(getattr(batched, name))
                )
            torch.testing.assert_close(
                original.cur_reward_sum, batched.cur_reward_sum, rtol=0, atol=0
            )

    def test_update_matches_upstream(self):
        for schedule in ("adaptive", "fixed"):
            cfg = config()
            cfg["algorithm"]["schedule"] = schedule
            stock = OnPolicyRunner(ToyEnv(), copy.deepcopy(cfg))
            local = LocalOnPolicyRunner(ToyEnv(), copy.deepcopy(cfg))
            self.assertIs(type(local.alg), PPO)
            obs = stock.env.get_observations()
            with torch.inference_mode():
                for _ in range(cfg["num_steps_per_env"]):
                    actions = stock.alg.act(obs)
                    obs, reward, done, extras = stock.env.step(actions)
                    stock.alg.process_env_step(obs, reward, done, extras)
                stock.alg.compute_returns(obs)
            local.alg.load(copy.deepcopy(stock.alg.save()), load_cfg=None, strict=True)
            local.alg.storage = copy.deepcopy(stock.alg.storage)
            torch.manual_seed(123)
            expected = stock.alg.update()
            torch.manual_seed(123)
            actual = local.alg.update()
            for key in expected:
                self.assertAlmostEqual(expected[key], actual[key], places=10)
            self.assertEqual(stock.alg.learning_rate, local.alg.learning_rate)
            for expected_model, actual_model in (
                (stock.alg.actor, local.alg.actor),
                (stock.alg.critic, local.alg.critic),
            ):
                for key, value in expected_model.state_dict().items():
                    torch.testing.assert_close(
                        value, actual_model.state_dict()[key], rtol=0, atol=0
                    )

    def test_checkpoint_round_trip(self):
        runner = LocalOnPolicyRunner(ToyEnv(), config())
        runner.learn(1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            runner.save(str(checkpoint))
            restored = LocalOnPolicyRunner(ToyEnv(), config())
            restored.load(str(checkpoint))
            for key, value in runner.alg.actor.state_dict().items():
                torch.testing.assert_close(
                    value, restored.alg.actor.state_dict()[key], rtol=0, atol=0
                )
            self.assertEqual(runner.alg.learning_rate, restored.alg.learning_rate)

    def test_nan_prevents_optimizer_update(self):
        runner = LocalOnPolicyRunner(ToyEnv(nan_rewards=True), config())
        before = {
            name: value.clone() for name, value in runner.alg.actor.named_parameters()
        }
        with self.assertRaisesRegex(ValueError, "PPO update was skipped"):
            runner.learn(1)
        for name, value in before.items():
            torch.testing.assert_close(
                value, dict(runner.alg.actor.named_parameters())[name], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
