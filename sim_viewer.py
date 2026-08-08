import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

paused = True


def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if not paused:
            mujoco.mj_step(model, data)
        viewer.sync()
