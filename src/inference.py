import copy
import time
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer
from src.env import FoosballEnv

# IMPORTANT: Import your exact training config dictionary here!
from src.training import train_cfg


def play():
    device = "cuda:0"
    env = FoosballEnv(device=device, sync_with_viewer=True)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir="logs/foosball", device=device)


    checkpoint = "logs/foosball/opponent_1.pt"

    runner.load(checkpoint)

    policy = runner.get_inference_policy(device=device)
    env.opponent_policy = policy

    obs = env.get_observations()

    print(f"loaded model {checkpoint}")


    with torch.no_grad():
        with m_viewer.launch_passive(env.mjm, env.mjd) as viewer:
            while viewer.is_running():
                start = time.time()
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                viewer.sync()

                remaining = env.mjm.opt.timestep - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)

if __name__ == '__main__':
    play()