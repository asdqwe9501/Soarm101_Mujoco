# Sim ↔ real alignment notes (Task 4)

Tool: `python tools/align_overlay.py --episode 0 --frames 0 225 325` (`--episode 20 --frames 229 310` for the other after images). Each PNG is [real | sim | 50% blend].
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
| MID_RAD | [0.02, -1.68, 1.29, 1.86, 0, 0] |
| joint ranges | wrist_flex range and ctrlrange widened to [-1.658, 2.22] (real data max is 2.164 rad). Other joints unchanged |
| SIM_HOME_NORM | [-2.8, 0.1, 93.7, **43.7**, -0.7, 1.5]: real HOME_NORM with wrist_flex -28 units. Used by the sim for reset, at_home and tests |
| overhead camera `top` | pos (0.494, 0.025, 0.542), targetbody `cam_target` at (0.1875, 0.0245, 0), fovy 52.5 |
| bin | center (0.165, -0.205) (scene.xml `bin` pos and env.py `BIN_CENTER`). Diameter 9 cm and height 7 cm unchanged (`BIN_R_IN` 0.045, `BIN_H` 0.07) |
| wrist_cam | pos (-0.00296, 0.05436, -0.05310) unchanged. quat (0.954498, -0.284308, -0.085070, 0.029438) = old quat rotated -10° about the camera's local y. fovy 78 (was 61) |
| robot base | unchanged at (0, 0, 0). A fitted offset of about 2 cm was not significant, and the base is clamped |

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

## Joint zero offsets (MID_RAD)
- Data: 10 episodes (0, 1, 2, 10, 20, 30, 40, 50, 60, 70).
- Per episode, three constraints:
  - Grasp: the frame where the gripper first closes (<15) after opening. gripperframe must equal the black ball's 3D position. The ball is detected in frame 0 of the overhead image and back-projected to z=0.02.
  - Release: the frame where the gripper reopens. gripperframe xy must be at the bin center.
  - Home: a weak 2D pixel constraint.
- Result [0.02, -1.68, 1.29, 1.86]:
  - grasp error 1.6 cm mean, 2.6 cm max
  - release xy 2.3 cm mean
  - `after_multi_episode_overhead.png` (episodes 0/20/50: home, grasp, release) shows the reach silhouettes overlapping.
- The earlier ep0-only fit [-0.05, -2.10, 1.91, 1.70] has a 5.5 cm mean grasp error on the 10 episodes. It is rejected.

### Known sim-to-real gap: the home pose
- Under this MID_RAD the real HOME_NORM (wrist_flex 71.7) folds the gripper 35 mm into the shoulder (self-collision). The table is clear.
- A fit with a collision-free-home constraint could not keep the reach errors near 2 cm. The constrained fits degraded to 3-7 cm or reached nonsensical joint configurations.
- So the sim uses `SIM_HOME_NORM` instead. It changes wrist_flex only, 71.7 → 43.7 (-28 units ≈ -0.33 rad):
  - no self-contact
  - lowest gripper geom 45 mm above the table
  - tracks within 0.02 units after 2 s
- `HOME_NORM` stays the real value.
- Consequences:
  - Sim episodes start and end with the wrist about 28 units less folded than in the real data.
  - A policy trained in sim will return to SIM_HOME_NORM.
  - On the real robot, `at_home` should use HOME_NORM.
- In the overhead image the real home gripper hangs in front of the shoulder. In the sim it folds back. The home-pose silhouette is visibly different (`after_ep0_f0_front.png`).

## Remaining visible mismatches
- Reach poses: residual about 1.5-2.5 cm. The sim upper arm reads slightly taller near the image top.
- Home pose: see the gap above.
- The sim robot looks about 5-10% larger than the real one in the overhead view.
- Wrist view:
  - The sim jaws show a gap while the real jaws (closed) overlap visually.
  - The real camera appears to look from further to the side.
  - fovy 78 is a rough fit: jaw spacing vs the 4 cm ball at the grasp frame. At release the real bin looks smaller than in the sim, i.e. the real wrist is higher above the bin by a few cm. This is consistent with the reach residual.
  - The PCB/lens visual geoms were not rotated with the camera (10° difference, visual only).
- Colors and lighting differ (out of scope).
