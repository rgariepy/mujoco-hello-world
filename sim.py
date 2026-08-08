import mujoco
import numpy as np
from PIL import Image
import os

os.makedirs("artifacts", exist_ok=True)

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
Image.fromarray(renderer.render()).save("artifacts/before.png")

for _ in range(500):
    mujoco.mj_step(model, data)

renderer.update_scene(data)
Image.fromarray(renderer.render()).save("artifacts/after.png")
print("Saved artifacts/before.png and artifacts/after.png")
