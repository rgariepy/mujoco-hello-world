import mujoco
import mujoco.viewer

from control import apply_stability_control, reset_to_stand

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)
reset_to_stand(model, data)

paused = True


def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if not paused:
            apply_stability_control(model, data)
            mujoco.mj_step(model, data)
        viewer.sync()
