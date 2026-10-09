# Sim ↔ real alignment notes (Task 4)

Tool: `python tools/align_overlay.py --episode 0 --frames 0 225 325`. Each PNG is [real | sim | 50% blend].
- `before_*`: the HEAD state before Task 4.
- `after_*`: the committed state.
- `*_front.png`: the overhead camera.
- `*_top.png`: the wrist camera (see the key swap below).

## Camera key swap (dataset fact)
`taeyoungss/SmolVla_grab_blackball` has the key names reversed relative to the cameras:
- `observation.images.top` = **wrist** camera
- `observation.images.front` = **overhead** camera

`sim/env.py` mirrors this: `TOP_CAM = "wrist_cam"`, `FRONT_CAM = "top"` (the MuJoCo overhead camera keeps the name `top`). Evidence: `evidence_camera_key_swap.png` (episodes 30 and 70).

## Final values
| item | value |
|---|---|
| overhead camera `top` | pos (0.494, 0.025, 0.542), targetbody `cam_target` at (0.1875, 0.0245, 0), fovy 52.5 |
| bin | center (0.165, -0.205) (scene.xml `bin` pos and env.py `BIN_CENTER`). Diameter 9 cm and height 7 cm unchanged (`BIN_R_IN` 0.045, `BIN_H` 0.07) |
| wrist_cam | pos (-0.00296, 0.05436, -0.05310) unchanged. quat (0.954498, -0.284308, -0.085070, 0.029438), i.e. old quat rotated -10° about the camera's local y. fovy 61 unchanged |
| MID_RAD | **unchanged** [0.0, -1.70, 1.54, 1.20, 0, 0]. See the blocker below |

### Overhead camera
- Fitted to the 4 table corners: FL(100,130), FR(508,128), NL(15,412), NR(578,415) px. Residual 0.8 px RMS.
- With the height forced to the user's 0.85 m the residual is ≥ 9 px RMS. Two objects of known size confirm the 0.54 m scale:
  - ball: measured 33-36 px, predicted 33 px at 0.54 m vs 21 px at 0.85 m
  - bin: measured ~75 px, predicted ~70 px at 0.54 m vs ~44 px at 0.85 m
- The camera is fixed across episodes (checked 0..88).

### Bin
- The rim and bottom ellipses back-projected through the fitted camera give a center of about (0.165, -0.205).
- Apparent rim diameter 8-9 cm and height about 7 cm, consistent with the existing geometry, so `BIN_R_IN` and `BIN_H` are unchanged. RELEASE logic in later tasks needs no change for height.
- The sim wall still reads slightly taller than the real one in the overlay.

## Blocker: joint zero offsets (MID_RAD)
Method:
- Constraints:
  - frame 225: gripper closed on the black ball. 3D ball position from back-projection = (0.254, -0.111, 0.02).
  - gripperframe pixels in frames 0, 275, 300 and 325.
- Least-squares fit of MID_RAD[0:4] gives about [-0.05, -2.10, 1.91, 1.70], with errors of about 1-2 cm and 1-17 px. Extended frames overlay well: `evidence_fitted_mid_rad.png`.
- Under the old limits the best feasible fit is [-0.08, -1.795, 1.68, 1.41]. It misses by 4-6 cm and up to 67 px: `evidence_bounded_mid_rad.png`.

What I tried after the controller allowed widening the limits:
- Widened the ranges: lift [-2.15, 1.745], elbow [-1.69, 2.01], wrist_flex [-1.658, 2.06].
- At HOME_NORM the fitted MID_RAD puts the gripper 4.5 cm **below** the table top. Contacts:
  - table vs gripper / jaws: up to 35 mm penetration
  - shoulder vs lower_arm: up to 27 mm
  - shoulder vs wrist: up to 13 mm
  - shoulder vs fixed jaw: up to 21 mm
- tests/test_env.py failed: home tracking reached lift 13.7 instead of 0.1 and wrist 44 instead of 71.7.

More fits:
- Adding a "home must be collision-free" penalty: the best collision-free fit has errors up to 100 px.
- Adding a 3D home target with the tip 3 cm above the table at the observed pixel: about 6 cm error at home, and still 35 mm penetration.

Conclusion: no constant per-joint offset (with sign +1 and 4096 ticks/rev) explains both the folded home pose and the extended poses. The changes were reverted. MID_RAD and the joint ranges are unchanged.

Likely causes to investigate:
- a joint whose tick→angle mapping is not a pure offset: wrong direction, or a homing/wrap issue (shoulder_lift's calibration range is 0..4094, i.e. the full circle)
- gripperframe ≠ the real grasp point
- the real base not being at (0, 0)

## Remaining visible mismatches
- The robot pose is off in all extended frames because MID_RAD is not calibrated.
  - At frame 225 the sim gripper overshoots the ball.
  - At frame 325 it reaches past the bin.
- The sim robot looks about 5-10% larger than the real one in the overhead view.
- Wrist view:
  - The sim jaws show a gap while the real jaws (closed) overlap visually.
  - The real camera appears to look from further to the side.
  - The PCB/lens visual geoms were not rotated with the camera (10° difference, visual only).
  - Re-check after MID_RAD is fixed, since the scene content depends on the arm pose.
- Colors and lighting differ (out of scope).
