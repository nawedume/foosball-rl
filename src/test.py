import time
import warp as wp
import torch

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer
from env import FoosballEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="headless mode")
parser.add_argument("--run_zeros", action="store_true", help="zero mode")
args_cli = parser.parse_args()

device = 'cpu'
env = FoosballEnv(device=device, sync_with_viewer=True, always_blue=True)

test_data = []

def run_env():

    if args_cli.run_zeros:
        actions = torch.zeros((env.num_envs, 16))
    else:
        actions = torch.sin(env.episode_length_buf * 0.1).unsqueeze(1).repeat(1, 16) * 20.0

    assert actions.shape == (env.num_envs, 16)
    obs, rewards, dones, _ = env.step(actions)

    ball_pos_rel = obs["policy"][0, 32:35]
    test_data.append(torch.cat([ball_pos_rel[0:1], ball_pos_rel[1:2], rewards], dim=-1))

if args_cli.headless:
    while True:
        try:
            run_env()
        except KeyboardInterrupt:
            break

else:
    with m_viewer.launch_passive(env.mjm, env.mjd) as viewer:
        while viewer.is_running():
            start = time.time()

            run_env()

            viewer.sync()
            remaining = env.mjm.opt.timestep - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)


# write test data
ball_pos_to_reward = torch.stack(test_data, dim=0)
print(f"saving ball data of shape: {ball_pos_to_reward.shape}")
torch.save(ball_pos_to_reward, "ball_reward.pt")
