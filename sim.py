import os

import mujoco
import numpy as np
from PIL import Image

from control import apply_stability_control, reset_to_stand

os.makedirs("artifacts", exist_ok=True)

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

reset_to_stand(model, data)
apply_stability_control(model, data)
mujoco.mj_forward(model, data)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.lookat[:] = [0.0, 0.0, 0.28]
camera.distance = 1.35
camera.elevation = -22
camera.azimuth = 140

renderer.update_scene(data, camera=camera)
Image.fromarray(renderer.render()).save("artifacts/before.png")

# ~2.5 seconds at timestep 0.002 — enough for the red box to hit and settle
n_steps = 1250
heights = []
rolls = []
pitches = []

for _ in range(n_steps):
    apply_stability_control(model, data)
    mujoco.mj_step(model, data)
    R = data.body("torso").xmat.reshape(3, 3)
    zx, zy, zz = R[:, 2]
    rolls.append(np.arctan2(zy, zz))
    pitches.append(np.arctan2(-zx, np.sqrt(zy * zy + zz * zz)))
    heights.append(data.body("torso").xpos[2])

renderer.update_scene(data, camera=camera)
Image.fromarray(renderer.render()).save("artifacts/after.png")
renderer.close()

final_height = heights[-1]
max_abs_tilt = max(max(abs(r) for r in rolls), max(abs(p) for p in pitches))
min_height = min(heights)
print("Saved artifacts/before.png and artifacts/after.png")
print(f"Final torso height: {final_height:.3f} m (min during run: {min_height:.3f})")
print(f"Max |tilt| during run: {np.degrees(max_abs_tilt):.1f} deg")
print(f"Red box final z: {data.body('red_box').xpos[2]:.3f} m")
