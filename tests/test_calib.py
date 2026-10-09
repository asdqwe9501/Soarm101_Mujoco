import numpy as np

from calib import HOME_NORM, MID_RAD, Calib


def test_roundtrip():
    c = Calib()
    v = np.array([[-38.2, 0.0, -55.2, 30.7, -13.3, 0.3], [52.4, 85.3, 96.5, 88.6, 3.2, 69.9], HOME_NORM])
    np.testing.assert_allclose(c.rad_to_norm(c.norm_to_rad(v)), v, atol=1e-9)


def test_home_maps_near_mid_rad():
    # 실제 캘리브레이션은 제자리 자세에서 약 2047 tick이다
    q = Calib().norm_to_rad(HOME_NORM)
    np.testing.assert_allclose(q, MID_RAD, atol=0.1)


def test_real_data_range_inside_sim_joint_limits():
    import mujoco

    m = mujoco.MjModel.from_xml_path("sim/scene.xml")
    c = Calib()
    # 실제 데이터셋 observation.state 최솟값/최댓값
    for v in [[-38.2, 0.0, -55.2, 30.7, -13.3, 0.3], [52.4, 85.3, 96.5, 88.6, 3.2, 69.9]]:
        q = c.norm_to_rad(np.array(v))
        lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
        assert np.all(q >= lo - 0.05) and np.all(q <= hi + 0.05), q
