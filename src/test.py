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

        env.step(torch.zeros(1, device=device))

        viewer.sync()

        remaining = env.mjm.opt.timestep - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
