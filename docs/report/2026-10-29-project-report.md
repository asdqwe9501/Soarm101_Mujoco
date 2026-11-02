# SO101 Ball Sort Project Report (2026-10-10 to 10-29)

## 1. Summary

**Goal:** Given a spoken command, the SO101 arm picks the ball of that color (red / black / sky blue, where sky blue is also called "blue"), drops it in the bin, and returns home, then repeats. The work is finished in simulation first, then moved to the real robot.

**Current status (simulation):**

| Item | Result |
|---|---|
| Color accuracy (wrong ball grasped) | **0 / 600** |
| Task success (grasp → bin → home) | **86.3%** (518 / 600) |
| Command loop (consecutive commands) | 30 / 30 |
| Real robot | Code ready. Starts with camera color calibration |

**Final setup:**
- Policy: IRL v2 = SmolVLA (copy of sft_aux2) + noise-steering policy
- Color guard v2: grasps only after confirming the color with the wrist camera

**Remaining problem:** Most failures are grasp failures (82 of 600), not color errors.

**Details:** models in Section 7, training in Section 8, dataset in Section 9, evaluation in Section 10.

---

## 2. System

| Part | Description | Files |
|---|---|---|
| Simulator | Rebuilds the real workspace in MuJoCo. Camera pose, bin position and joint calibration were fit to 10 real recordings. Three balls are placed randomly (minimum spacing 7 cm). | `sim/env.py`, `calib.py` |
| Scripted expert | Generates demos from ground-truth positions and inverse kinematics. 150 / 150 success. | `expert/scripted.py` |
| Demo data | LeRobot format, same as real data. The final version (v8) has 1,500 episodes. | `data/` |
| Base model | SmolVLA trained on the demos (sft_aux2). | `train/sft_aux.py` |
| Additional training | A noise-steering policy trained with inverse RL (IRL). The base model is not changed. | `train/irl.py`, `rl/` |
| Color guard | Checks the color with the wrist camera right before the gripper closes. Rejects a wrong color and retries. | `policies/color_guard.py` |
| Command loop | Idle → command → task → return home → next command. Supports simulation and the real robot. | `run/loop.py` |

**Reward (user-defined, not to be changed):**

| Event | Reward |
|---|---|
| Grasp attempt on the correct ball | +1 |
| Lift | +2 |
| Drop in bin | +3 |
| Return home | +1 |
| Wrong ball | −1 |
| Empty grasp | −0.5 |

---

## 3. Progress

### 3.1 Simulator and demos (Plan 1)
- Calibrated cameras, bin and joints against real recordings.
- Scripted expert success: 150 / 150.

### 3.2 First SFT: failure (Plan 2)
- **sft_v1 success: 0.7%.**
  - Cause: the model ignored images and the command. It learned a shortcut of following the current joint state.
- **Grasp tolerance improvement (Plan 2b):**
  - Fixed the simulator friction setting (noslip) and added correction demos that miss and then recover.
  - Grasp success at a 1.5 cm offset went from 1 / 12 to 10 / 12.
- **sft_v3: 58.7% success.**
  - Grasping and carrying were mostly solved.
  - The remaining bottleneck was **color selection (about 23% wrong balls)**.

### 3.3 Color grounding attempts: mostly failed (Plans 3–5)

Each of the three color commands is given in the same layout; the first attempt is checked.

| Attempt | Correct ball | Wrong ball |
|---|---|---|
| sft_v3 | 71% | 23% |
| DSRL v2 (noise-steering RL) | 69% | 26% |
| sft_v5 (shortcut-removal data only) | 21% | 44% |
| LoRA2 (partial language-model fine-tuning) | 62% | 23% |

**Lesson:** Changing only the data or the training method did not push the wrong-ball rate below about 23%.

### 3.4 Breakthrough: auxiliary task (Plan 6, sft_aux2)

**Method:**
- Jointly trained an auxiliary task: predict the commanded ball's position from the first frame, before the arm moves.
- Joint state is masked in that task to block the shortcut.
- 1,500 demos with diverse command phrasings.

**Result (first attempt, same for basic and unseen commands):**
- Correct ball 92.2%, wrong ball **3.3%** (sft_v3: 23%)

**Full task:**
- Single-command success 82–85%
- 87.3% with the color guard

### 3.5 Adding RL

| Method | Result |
|---|---|
| DSRL v3 | **Performance dropped** (single success 82% → 63%). Harmful when the base model is already strong. |
| KL regularization | Keeps the policy close to the base model. Used for IRL. |
| **IRL v1** | A discriminator infers the reward from demos. At first it learned motion style, not color (AUC 0.54). Adding "swapped-command negatives" (demos with only the command changed to another color) made it separate colors (AUC 0.89). Single success 90%, 0% wrong balls on basic commands (first attempt). |
| **IRL v2** | Expanded to 30 command phrasings, with evaluation-only phrasings held out. Single success 92% on basic commands, 85% on unseen commands. |

### 3.6 Root cause of wrong balls

- **Errors happen only when balls are close together:**

  | Distance to nearest ball | Wrong-ball rate |
  |---|---|
  | ≥ 12 cm | 0 / 630 |
  | 8–9 cm | 4.5% |
  | 7–8 cm | 11.2% |

- **The wrong ball is almost always the neighbor** (56 of 62 cases). This is closer to a spatial precision problem than color confusion.
- **Mostly chance.** Replaying a wrong case with only the noise changed gives the right ball 43–78% of the time.
- **Conclusion:** Reaching 0% by training alone is hard, so errors are blocked with a **camera check (color guard)**.

### 3.7 Color guard v1 (10-27)

- **How it works:** Measures the share of color pixels in the region between the gripper jaws in the wrist image. On a wrong color it does not close; it backs off and retries.
- **Results:**
  - First-attempt color accuracy 97.9% → 99.3% (basic), 96.9% → 99.6% (unseen)
  - Full task: 6 wrong balls in 300 episodes
- **Retry strategy comparison** (home / half / lift / reset): differences were within sampling error.

### 3.8 Grasp failure analysis (10-28)

150 episodes each.

| Category | Basic | Unseen | Fix |
|---|---|---|---|
| Success | 131 | 127 | |
| Closed 1.7–2.2 cm off to the side | 4 | 4 | hardware |
| Pushed the ball away while descending | 4 | 3 | hardware + training |
| Closed in the air above the ball | 2 | 0 | training |
| Did not close near the ball / stalled | 6 | 6 | training |
| Failed after lifting (dropped, timeout) | 1 | 5 | hardware / time |
| Wrong ball | 2 | 4 | guard fix (→ 3.9) |

- **The gripper tips (8–10 mm wide) are narrow for a 4 cm ball.** The ball slips out if the close is more than 1.5 cm off to the side. Wider tips or pads would fix part of this.
- **Every failed episode used the full 30 s with 2–4 retries.** A missed ball often rolled off the table.

### 3.9 Color guard v2: 100% color accuracy (10-29)

**Four ways guard v1 let a wrong ball through, and the fixes:**

| Hole | Before | Fix |
|---|---|---|
| ① Correct color confirmed once, then the grasp missed | No more checks afterwards, so grasping another ball on the retry passed. **Main cause.** | Detects an empty gripper from the gripper joint value. Holding a ball blocks the jaws, so the joint reads about 9.5 higher; a full close means empty. If empty, the next close is checked again. |
| ② After 3 rejections | Let it through. | Rejects every time, no limit. |
| ③ Color still unknown at the moment of closing | Closed anyway. | Does not close. **Grasps only when the commanded color is confirmed.** |
| ④ Black misdetection | Saw the black background beyond the table in simulation as a black ball. | Excludes pure black (brightness 0). 99.7% of black-ball pixels have brightness ≥ 3. |

**Results (full task, based on the ball actually lifted or placed in the bin):**

| Set | Success | Wrong ball |
|---|---|---|
| Basic commands, same scenes as before | 134 / 150 (before 131) | **0** (before 2) |
| Unseen commands, same scenes as before | 127 / 150 (before 127) | **0** (before 4) |
| Basic commands, new scenes | 132 / 150 | **0** |
| Unseen commands, new scenes | 125 / 150 | **0** |
| **Total** | **518 / 600 (86.3%)** | **0 / 600** |

**Additional checks:**
- **4,251 wrist frames at the moments the policy actually closes:** 0 classified as another color.
- **Stress test, 600 episodes:** on layouts with balls ≤ 10 cm apart, closes were deliberately shifted 0–3.5 cm toward the neighbor. The guard's verdict never disagreed with the ball actually lifted (348 lifts).
- **Same layout × three commands, first attempt (300 each):**
  - Unseen commands: 0 other-color attempts.
  - Basic commands: 1. This one was counted with the simulator's "nearest ball" criterion, which sometimes differs from the ball actually lifted (13 / 600), so whether it was a real error is unconfirmed.

**Cost:**
- Unknown-color closes are also rejected, so retries increased. Timeouts in the first-attempt test rose from 2–5% to 7–8%.
- Full-task success is unchanged.

### 3.10 Command loop (10-28)

- **Flow:** home → command input → task → return home → next command
- **End-of-task detection:** The real robot has no ground-truth signal, so only the motion is used. Grasp → release → staying home for 0.5 s means done.
- **Simulation check:** 10 scenes × 3 consecutive commands = **30 / 30 success**. End-of-task detection matched the simulator ground truth all 30 times.

---

## 4. Color accuracy over time

| Stage | Wrong-ball rate |
|---|---|
| sft_v3 (first attempt) | 23% |
| sft_aux2 (auxiliary task, first attempt) | 3.3% |
| IRL v2 (first attempt) | 2.0–3.0% |
| IRL v2 + guard v1 (first attempt) | 0.3–0.7% |
| IRL v2 + guard v2 (full task) | **0 / 600** |

**What "100%" means and its limits:**
- 0 of 600 means the error rate is likely below 0.5%. It is not a proof of 0%.
- These are simulation results. They must be re-checked with the real camera.
- The policy alone is about 97–98% accurate on color; the guard blocks the rest. So **the guard's color classification being accurate is a premise**.
- A theoretical gap remains: two balls between the jaws where only the commanded color is prominent in the image. This never occurred in testing.

---

## 5. Moving to the real robot: to-do

1. **Connection info:** robot serial port (e.g. COM5), wrist camera index, front camera index
2. **Color guard calibration:**
   - `python tools/guard_calibrate.py capture --wrist-cam <index>`
   - Place a ball between the jaws and take 20+ photos per color, varying lighting and position. Also take photos with no ball.
   - `python tools/guard_calibrate.py check` → adjust the ROI and color thresholds until accuracy is 100%.
3. **Check the empty-gripper threshold (HOLD_GAP):** measure the gripper joint value when holding a real ball vs. closing empty.
4. **Run:**
   - `python -m run.loop --backend real --port <port> --wrist-cam <index> --front-cam <index>`
   - Start with a small `--max-relative-target` and keep the emergency stop within reach.
5. **Hardware (optional):** widen the gripper tips or add anti-slip pads. This covers about half of grasp failures.

**Where simulation and reality may differ:**
- Camera color rendering and lighting
- Real ball color and reflections
- Gripper joint values
- How far balls roll
- The home pose in real demo data differs from simulation only in wrist_flex, by 28 units (`calib.py`). The loop moves to the simulation home pose before starting.

---

## 6. File locations

| Content | Location |
|---|---|
| Color guard v2 results | `docs/results/guard2/analysis.md`, `docs/results/guard2/*.json` |
| Grasp failure analysis | `docs/results/grasp/analysis.md` |
| Wrong-ball root cause | `docs/results/erran/analysis.md` |
| All color grounding attempts | `docs/results/color_grounding_attempts.md` |
| Plans (Plan 1–9) | `docs/superpowers/plans/` |
| Evaluation videos (guard v2, by failure type) | `outputs/guard2/videos_{basic,heldout}_s{0,5000}/` |
| Evaluation videos (IRL v2 + guard v1, success / failure) | `outputs/irl/videos_irl2_guard_{basic,heldout}/` |
| Grasp failure videos | `outputs/grasp/videos_{basic,heldout}/` |
| Model | `outputs/train/sft_aux2_irl/pretrained_model` + `outputs/irl/irl2/last.pt` |

Each video folder's `videos.txt` lists the command for each file. (`outputs/` is not in the repository.)

---

## 7. Models

### 7.1 SmolVLA (base policy)

| Item | Value |
|---|---|
| Starting point | `lerobot/smolvla_base` (LeRobot 0.4.4) |
| Vision-language model | SmolVLM2-500M-Video-Instruct, first 16 layers only |
| Action generation | Flow-matching "action expert". Starts from noise and produces actions in 10 steps. |
| Output | Action chunk of 50 steps (1.7 s at 30 Hz) × 6 joints. All 50 steps are executed before the next inference. |
| Input | Wrist image (`observation.images.top`), front image (`observation.images.front`), 6 joint values, command text |
| Image processing | Pad-resize to 512×512. The third camera slot is empty (`empty_cameras=1`). |
| Trained parts | Vision encoder and language model frozen. Only the action expert and state projection are trained. |
| Added part | Auxiliary head (`aux_head`). Used only during training, ignored at run time. |

### 7.2 Noise-steering policy (trained with IRL)

- **Idea:** SmolVLA stays unchanged. Only the **initial noise** fed into action generation is chosen, which changes the action (DSRL approach).
  - The noise is 50 steps × 6 joints = 300 dimensions. The policy sets only 12 (6 joints × 2 temporal components: constant and slow slope); the rest is filled randomly.
- **Input features (2,438 dims):** extracted inside SmolVLA.

  | Block | Dims | How |
  |---|---|---|
  | Wrist image | 960 | token mean |
  | Front image | 512 | 4×4 grid mean → 32 dims per cell |
  | Command | 960 | token mean |
  | Joint state | 6 | |

- **Network:** MLP with 512 hidden units. Gaussian policy outputting mean and variance (KL-regularized actor-critic). Two Q functions, taking the minimum.
- **Learning rates:** actor 1e-4, critic 3e-4, discount 0.99
- **Penalty for deviating from the reference N(0, I) (β = 1):** the last layer is zero-initialized, so before training it behaves exactly like the original SmolVLA.

### 7.3 Discriminator (IRL reward model)

- **Input:** feature pair (s, s') before and after one RL step, 2 × 2,438 dims
- **Architecture:** LayerNorm → 256 → 256 → 1. Dropout 0.2, R1 penalty 10, learning rate 1e-4
- **Reward:** log D(s, s'). Always ≤ 0; closer to 0 the more demo-like.

### 7.4 Color guard (rule-based, no training)

- **Region:** the area between the jaws in the 480×640 wrist image (rows 250–480, columns 255–495)
- **Color classification:** HSV rules measure the pixel share of each color.
  - Red: hue < 15° or > 345°, saturation > 0.6, value > 0.3
  - Black: 0.01 < value < 0.2
  - Sky blue: hue 185–225°, saturation > 0.25, value > 0.45
- **Decision:** the top color wins if it is ≥ 15% and at least twice the second color; otherwise "unknown". Blobs touching the top, left or right edge are treated as background and excluded.

---

## 8. Training

### 8.1 SFT: imitation + auxiliary task (sft_aux2)

| Item | Value |
|---|---|
| Data | v8, 1,500 demos (→ 9.2) |
| Steps / batch | 30,000 / 16 |
| Optimizer | AdamW, lr 1e-4, 1,000 warmup steps, cosine decay (min 2.5e-6), gradient clipping 10 |
| Time | about 14 h (RTX 5070 Laptop 8 GB) |
| Loss | flow loss + 0.1 × auxiliary loss |

**Auxiliary task:**
- Predict the (x, y) position of the commanded ball. The mean of the action expert's hidden states goes into an MLP (256, GELU). Coordinates are normalized to [−1, 1] over the workspace box.
- Three safeguards against peeking at the answer:
  1. Computed only on frames before the arm moves (first 10 frames). Once the arm has moved toward the ball, the position is easy to guess.
  2. Joint state is fed as zeros (shortcut blocked).
  3. Actions are predicted from pure noise (t = 1), so no ground-truth action leaks in.
- A separate data loader (batch 16) samples only those frames.
- **Result:** auxiliary position error went from 9.4 cm at the start to 0.25 cm at the end.

**Lessons from earlier failures:**
- sft_v1 (no auxiliary task): learned the shortcut of following joint state.
- sft_aux1 (auxiliary loss on all frames): 0.9 cm in training but 8 cm on the first frame, because it read the position from the already-moving arm.

### 8.2 IRL v2: inverse RL (on a copy of sft_aux2)

**Procedure:**
1. **Collect demonstrations:** run the scripted expert 600 times, recording (s, s') feature pairs every RL step (50 env steps). Commands are drawn from training phrasings + IRL extension phrasings (30 kinds).
2. **Pretrain the discriminator (3,000 updates):**
   - Positives: demonstrations
   - Negatives: experience from the earlier DSRL v3 policy
   - **Swapped-command negatives (50% of each negative batch):** keep the demo motion, replace only the command features with another color's command. This makes the discriminator learn "does the action match the command" rather than motion style.
3. **Policy training (2,000 episodes):**
   - An episode ends on task success or at 900 steps (30 s).
   - The discriminator is updated 20 times per episode. Negatives are the latest 5,000 transitions plus swapped samples.
   - The critic trains alone for 2,000 updates before the actor starts. Batch 256.
   - Rewards are recomputed with the current discriminator at every sample.
   - **The user-defined staged reward (+1/+2/+3/+1/−1/−0.5) was not used for training.** It was used only for logging and evaluation.
4. **Results:**
   - About 3 h of training, 1,084,014 env steps
   - KL to the reference 0.16, success over the last 50 episodes 82%

**Reward gate before training:** first checked that the discriminator separates correct-color from wrong-color demos in the same scene.
- Plain GAIL: AUC 0.54 (fail)
- With swapped-command negatives: AUC 0.89
- Unseen phrasings: AUC 0.81, same-scene pair comparison 100%

### 8.3 Training tried but not used

| Method | Result | Reason |
|---|---|---|
| sft_v1–v3 (no auxiliary task) | 23% wrong balls | Did not read color and position from images. |
| sft_v5 / v6 (shortcut-removal data) | 17–44% wrong balls | Empty grasps increased. |
| LoRA1 / 2 (partial language-model fine-tuning) | 23–26% wrong balls | No improvement. |
| DSRL v1–v3 (SAC, staged reward) | collapse / no effect / worse | Drifted too far from a strong base model. |

---

## 9. Dataset generation

### 9.1 Demo generation (`expert/scripted.py`, `data/collect.py`)

**Scripted expert:**
- Builds joint trajectories from the simulator's ground-truth ball positions and inverse kinematics (damped least squares, multiple starts).
- Segments: home → 7 cm above the ball → descend to 1.2 cm → close → lift 8 cm → 9 cm above the bin → open → home → wait
- Segments are joined with minimum-jerk curves; segment durations are randomized ±15%.
- Ball positions get ±5 mm random error.
- For balls right in front of the robot, the hand tilts toward the body (up to 30°) to avoid hitting the shoulder.

**DART noise:**
- Small noise (σ 0.5) is added to executed actions, while the stored labels stay on the clean trajectory.
- So the model learns how to recover from slightly off states.
- No new noise during the descend and close segments.

**Correction demos:**
- Deliberately go to a point 2–4 cm off, pause 0.3–0.6 s, then correct to the right position.
- Labels in the off segment are "one step from the current position toward the correct one".

**Storage:**
- Only successful episodes are saved.
- LeRobot v3 format, same as real robot data: 6 joints (normalized units), two 480×640 images, 30 Hz.

### 9.2 Final dataset v8 (for sft_aux2)

```
python -m data.collect --same-layout --randomize --mode-mix --modes default,correction --scene-mix \
  --aux-labels --paraphrase --repo-id local/so101_ballsort_sim_v8 --root data/so101_ballsort_sim_v8 \
  --n-per-color 500 --seed 500000
```

| Item | Value |
|---|---|
| Size | 1,500 episodes (500 per color), 662,721 frames (about 6.1 h) |
| Same layout | Each layout is run with all three color commands. Only the command differs in the same scene, so the command must be read to succeed. |
| Scene types | 3 balls: 1,060 / 2 balls (one removed): 219 / one already in the bin: 221 |
| Execution mode | default 764 / correction 736 |
| Visual randomization | brightness ×0.6–1.4, color temperature ±15%, desk color ±0.15, floor gray 0.3–0.8, ball color ±0.05, bin position ±2 cm |
| Ball placement | x 12–28 cm, y −10 to +20 cm, ≥ 7 cm between balls, ≥ 12 cm from the bin |
| Auxiliary labels | (x, y) of the commanded ball per frame. Used only in the loss, not as policy input. |
| Demos discarded as failures | 5 |

### 9.3 Command diversity (`sim/commands.py`)

| Category | Per color | Examples |
|---|---|---|
| Basic | 300 | `Grab the blue ball.` |
| Similar (6 kinds) | 150 | `pick up the red one`, `take the black ball and place it in the box` |
| Rare (3 kinds) | 50 | `put the blue ball in the box`, `the red one, please` |
| IRL extension (18 kinds, IRL only) | | `grasp the red ball`, `I want the black ball in the box` |
| **Evaluation only (3 kinds, never used in training)** | | `get the {c} ball`, `lift the {c} ball and drop it in the box`, `bring the {c} one to the box` |

- Two color forms are used, such as `red` / `red-colored`. Sky blue is called `blue`.
- Capitalization and the final period are randomized.
- The verbs in the evaluation-only phrasings (get / lift / bring / drop) are never used in training phrasings, for a fair test of unseen commands.

### 9.4 Dataset history

| Version | Size | Content |
|---|---|---|
| sim (v1) | 600 | basic demos, fixed command |
| corr | 600 | correction demos |
| all (v3 training) | 1,200 | the two above combined |
| v5 | 300 | shortcut removal (same layout, randomization, wrong-ball approach demos) |
| **v8** | **1,500** | **Final.** Diverse commands and auxiliary labels, wrong-ball approach demos removed |

---

## 10. Benchmarks (evaluation)

All evaluation runs in simulation with seeds not used in training. No visual randomization during evaluation.

### 10.1 Common criteria

| Verdict | Criterion |
|---|---|
| Task success | The commanded ball is in the bin (within 4.5 cm horizontally of the center, 7 cm high), no other ball is in the bin, and the arm is home (joint error ≤ 5). Must happen within 30 s (900 steps). |
| Grasp attempt | The moment the gripper command goes above 40 and then below 20 |
| Aimed ball | The nearest ball within 3.5 cm horizontally (≤ 7 cm high) of the gripper tip at that moment |
| Lifted | Ball height ≥ 5 cm |

### 10.2 Counterfactual test: color selection (`tools/counterfactual_eval.py`)

- **Method:** In the same layout, each of the three color commands is given. The ball of the **first grasp attempt** is checked.
- **Scale and seed:** started with 30 layouts (90 runs), later 100 layouts (300 runs, seed 30000).
- **Commands:** basic (`Grab the X ball.`) / unseen phrasings
- **Metrics:**
  - Share of correct ball, other color, empty, timeout (20 s)
  - Number of layouts with all three commands correct
  - Guard rejections are logged when the guard is used.
- **Purpose:** Shortcuts like position or order cannot get all three right. Only understanding the command can.

### 10.3 Full-task evaluation (`eval/evaluate.py`)

- **Single command:** 50 per color, 150 in total (seed 10000+). Reports success and failure types (wrong ball / grasp failure / release failure / return failure).
- **Consecutive commands:** 20 runs. All three balls in one scene are commanded one after another in random order. Metrics: "all three succeeded" and per-command success.
- **Videos:** 2 successes per color plus failures are saved, with a list of commands.

### 10.4 Failure analysis

| Tool | What it does |
|---|---|
| `tools/grasp_fail_analysis.py` | Records the gripper–ball relative position at closing (closing axis / width axis / height) and ball motion. Classifies failures into 11 types and saves videos per type. Also logs guard decisions. |
| `tools/cf_error_analysis.py` | Analyzes wrong-ball cases by ball spacing, command phrasing and overlap across models. |
| `tools/cf_repeat.py` | Repeats the same scene and command 10 times with only the noise changed, to tell chance from systematic errors. |

### 10.5 Color guard classifier tests

| Tool | What it does |
|---|---|
| `tools/policy_close_frames.py` | Collects 4,251 wrist frames at the moments the policy actually closes (200 layouts × 3 commands, seed 70000). Label: the aimed ball. |
| `tools/guard_stress_frames.py` | 600 closes shifted 0–3.5 cm toward the neighbor on layouts with balls ≤ 10 cm apart (seed 90000). Label: **the ball actually lifted**. |
| `tools/guard_calibrate.py` | Checks classification on real wrist-camera photos. Saves images with the ROI and color masks overlaid. |

### 10.6 Other tests

| Test | Description |
|---|---|
| Scripted expert | 150 / 150, consecutive 20 / 20. The baseline that validates the evaluator itself. |
| Grasp tolerance (`tools/grasp_tolerance.py`) | Plans the ball position offset by radius r in 12 directions and counts successes. |
| DSRL upper bound (`tools/dsrl_upper_bound.py`) | Tries 8 noises and measures how often at least one reaches the correct ball. Result: 100%. |
| IRL reward gate (`tools/irl_reward_gate.py`) | Discriminator AUC and same-scene pair comparison |
| Command loop (`run/loop.py`) | 10 scenes × 3 consecutive commands. Checks that end-of-task detection matches simulator ground truth. |
| Unit tests | `tests/` (pytest). 15 for the color guard, 3 for the loop, etc. |

### 10.7 Seed ranges (no overlap between training and evaluation)

| Use | Seed |
|---|---|
| Full-task evaluation | 10000+ |
| Grasp failure analysis / guard v2 measurement | 0+ / 5000+ |
| Counterfactual test | 20000+ / 30000+ |
| Classifier frames | 70000+ / 90000+ |
| DSRL training | 100000+ |
| v8 demos | 500000+ |
| IRL demonstrations / gate | 700000+ / 740000+ |
| IRL training | 800000+ |
