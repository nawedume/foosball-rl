import time
import warp as wp
import torch

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer
from env import FoosballEnv

device = 'cpu'
env = FoosballEnv(device=device, sync_with_viewer=True)

with m_viewer.launch_passive(env.mjm, env.mjd) as viewer:
    while viewer.is_running():
        start = time.time()

        actions = torch.sin(env.episode_length_buf * 0.1).unsqueeze(1).repeat(1, 16) * 20.0
        assert actions.shape == (env.num_envs, 16)
        env.step(actions)

        viewer.sync()

        remaining = env.mjm.opt.timestep - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
