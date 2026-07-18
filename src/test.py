import time
import warp as wp

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw
import mujoco.viewer as m_viewer

with open("./model.xml") as f:
    model_str = f.read()

mjm = mujoco.MjModel.from_xml_string(model_str)
mjd = mujoco.MjData(mjm)

m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=100)

dt = mjm.opt.timestep

with m_viewer.launch_passive(mjm, mjd) as viewer:

    while viewer.is_running():
        start = time.time()

        mjw.step(m, d)
        wp.synchronize()

        mjw.get_data_into(result=mjd, mjm=mjm, d=d)

        viewer.sync()

        remaining = dt - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
