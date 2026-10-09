from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "sim" / "scene.xml"


def test_scene_has_named_objects():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    for body in ["bin", "ball_red", "ball_black", "ball_sky_blue"]:
        assert m.body(body).id >= 0
    for cam in ["top", "wrist_cam"]:
        assert m.camera(cam).id >= 0
    assert m.nu == 6


def test_balls_rest_on_table():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(500):  # 2.5s
        mujoco.mj_step(m, d)
    for name in ["ball_red", "ball_black", "ball_sky_blue"]:
        z = d.body(name).xpos[2]
        assert abs(z - 0.02) < 0.003, (name, z)
