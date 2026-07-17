# Throw a ball at 100 different velocities.

import warp as wp

wp.config.enable_mathdx_solver = False

import mujoco
import mujoco_warp as mjw

_MJCF=r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""


mjm = mujoco.MjModel.from_xml_string(_MJCF)
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=100)

# initialize velocities
wp.copy(d.qvel, wp.array([[float(i) / 100, 0, 0, 0, 0, 0] for i in range(100)], dtype=float))

# simulate physics
mjw.step(m, d)

print(f'qpos:\n{d.qpos.numpy()}')
