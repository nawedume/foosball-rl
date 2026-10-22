import copy
import time
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer # ty: ignore
from env import FoosballEnv, NullPolicy

# IMPORTANT: Import your exact training config dictionary here!
from policy_cfg import train_cfg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chpt", type=str, required=True, help="The checkpoint to load.")
parser.add_argument("--device", type=str, default='cuda:0', help="Device to use")
parser.add_argument("--ep-len", type=int, default=None, help="Max ep length")
args = parser.parse_args()

def play():
    device = args.device

    env = FoosballEnv(device=device, sync_with_viewer=True, always_blue=True, bias_to_blue=True)

    if args.chpt != 'null':
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir="logs/foosball", device=device)
        checkpoint = args.chpt
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=device)
        print(f"Loaded model {checkpoint}")
    else:
        policy = NullPolicy(num_envs=1, device=device)
        print("Loaded Null Policy")

    if args.ep_len is not None:
        env.max_episode_length = args.ep_len

    obs = env.get_observations()

    with torch.no_grad():
        with m_viewer.launch_passive(env.mjm, env.mjd) as viewer:
            while viewer.is_running():
                start = time.time()
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                # print(f"Reward: {rewards}")
                viewer.sync()

                remaining = env.mjm.opt.timestep - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)

if __name__ == '__main__':
    play()
