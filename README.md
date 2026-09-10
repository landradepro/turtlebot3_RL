# TurtleBot3 RL

A simulated TurtleBot3 learns to navigate to randomly-placed goals while avoiding obstacles, trained via reinforcement learning (PPO, Stable-Baselines3) on top of ROS2 Humble and Gazebo Classic. Runs entirely on a single laptop in WSL2 or Linux — no real robot, no cloud compute.

This README assumes **zero prior experience** with ROS2, Gazebo, or reinforcement learning. Every concept is explained before it's used.

---

## 1. Overview

The task, each episode: starting from a fixed spawn point, reach a randomly sampled goal (shown as a green marker) somewhere within a 0.6–2.0m radius, while navigating around four static obstacles, using only a simulated laser scan and odometry. No map, no path planner. The driving behavior is a neural network trained purely from a reward signal (distance-to-goal, collision, a small per-step cost).

---

## 2. Architecture

```
┌─────────────────┐      /scan, /odom       ┌──────────────────────┐
│                  │◄────────────────────────│                      │
│  Gazebo Classic  │                          │  TurtleBot3GazeboEnv │
│  (gzserver)      │      /cmd_vel            │  (gym_env.py)        │
│  physics + LIDAR │◄────────────────────────│  Gymnasium Env       │
│  + obstacles     │                          │  wrapping a ROS2     │
│                  │   /reset_world,          │  node                │
│                  │   /spawn_entity,         │                      │
│                  │   /delete_entity         │                      │
└─────────────────┘◄────────────────────────└──────────┬───────────┘
                                                          │ reset()/step()
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Stable-Baselines3    │
                                              │  PPO (train.py /      │
                                              │  evaluate.py)         │
                                              └──────────────────────┘
```

**`gym_env.py`** is the one file that's genuinely specific to this project: it translates ROS2's world (topics, services, messages) into the standard Gymnasium `reset()`/`step()` contract that RL libraries expect:

- **Observation** (26 floats): the LIDAR scan downsampled into 24 direction-bins (closest object per bin) + distance-to-goal + heading-error.
- **Action** (`Box(2,)`): forward speed (0–0.2 m/s) + turn rate (-1.5–1.5 rad/s), published as `/cmd_vel`.
- **Reward**: shaped by change in distance-to-goal each step, +100 for reaching the goal, -50 for a collision (via minimum LIDAR range), a small per-step cost to discourage stalling.
- **Reset**: `/reset_world` snaps the robot to its spawn pose; a new goal is sampled and the visible marker is moved via delete+spawn of a collision-less entity.

See [`src/turtlebot3_rl/turtlebot3_rl/gym_env.py`](src/turtlebot3_rl/turtlebot3_rl/gym_env.py) in the repo for the full implementation.

---

## 3. Requirements

- WSL2 with **Ubuntu 22.04**, and **ROS2 Humble** installed (`/opt/ros/humble` should exist)
- ~8GB+ RAM available to WSL2
- An NVIDIA GPU is *not* required — training is CPU-bound by design (see [section 10](#10-technical-challenges-this-project-solved))

**Important:** if you have 32GB+ RAM and a capable GPU, consider NVIDIA Isaac Lab instead of this Gazebo-based approach — it offers GPU-parallelized training. This project targets more modest hardware (16GB RAM), where Isaac Sim's stated 32GB minimum simply isn't clearable. See [section 10](#10-technical-challenges-this-project-solved) for the full reasoning.

---

## 4. Core concepts

Skip this if you already know these terms.

**4.1 — ROS2.** A framework for writing robot software as many small programs that talk to each other, rather than one monolith. A **node** is one running program. A **topic** is a named, continuous data stream — `/scan` (laser), `/odom` (position estimate), `/cmd_vel` (drive commands). A **service** is a one-off request/response call, unlike a topic's continuous stream. A **launch file** starts several nodes together with one command.

**4.2 — Gazebo (Classic 11).** A physics simulator faking gravity, collisions, and sensor data closely enough that code written for the simulated robot works largely unchanged on real hardware. `gzserver` is the physics engine (can run headless); `gzclient` is just the 3D viewer window on top of it.

**4.3 — TurtleBot3 ("burger").** A standard two-wheeled ROS2 reference robot. **Differential drive**: no steering wheel — you command a forward speed and a turn rate, not a steering angle. Its **LIDAR** spins continuously, measuring distance to the nearest object at many angles around it.

**4.4 — Reinforcement learning.** An **agent** sees a partial view of the world (the **observation**), picks an **action**, and gets back a **reward** — a number saying "good" or "bad." One attempt start-to-finish is an **episode**. Across many episodes, the agent tunes its decision-making (the **policy**) to maximize total reward — nobody hand-codes "avoid walls"; it's discovered purely because crashing produces a bad number.

**4.5 — Gymnasium.** A standard interface any RL problem can be wrapped in: `env.reset()` → starting observation; `env.step(action)` → (next observation, reward, terminated, truncated, info). This uniformity is why an off-the-shelf algorithm can train against our robot without knowing anything about ROS2 or lasers.

**4.6 — PPO / Stable-Baselines3.** SB3 is a library of pre-built RL algorithms. **PPO** (Proximal Policy Optimization) is a comparatively stable one — "proximal" because it deliberately avoids changing the policy too drastically in any single update.

---

## 5. Installation

### Step 5.1 — Install ROS2 Gazebo + TurtleBot3 packages

```bash
sudo apt update
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3 ros-humble-turtlebot3-simulations ros-humble-turtlebot3-msgs
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
source ~/.bashrc
```

### Step 5.2 — Verify the simulator renders

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

A Gazebo window should appear with the robot in an empty world. (If nothing appears the first time, `Ctrl+C` and retry — a known one-off race condition between the model spawner and the GUI plugin, not a real problem.)

### Step 5.3 — Clone this repo and build the workspace

```bash
git clone https://github.com/landradepro/turtlebot3_RL.git ~/turtlebot3_rl_ws
cd ~/turtlebot3_rl_ws
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install gymnasium "stable-baselines3[extra]"
colcon build --symlink-install
source install/setup.bash
```

**Important:** the venv must be active *before* `colcon build` runs, or the `train`/`evaluate` console-script entry points get baked with the wrong Python interpreter — see [the FAQ](#9-troubleshooting--faq) for what that looks like and how to work around it either way. `--system-site-packages` is required so this venv can still see the `apt`-installed `rclpy` alongside the `pip`-installed RL packages.

---

## 6. Running the full pipeline

**Train** (headless, fast — see [section 7](#7-going-faster-simulated-time-and-unthrottled-physics) for why this matters):
```bash
ros2 launch turtlebot3_rl training_world.launch.py       # terminal 1
python3 -m turtlebot3_rl.train --timesteps 100000         # terminal 2
```

Watch `ep_rew_mean` in the log — it should trend upward as training progresses. The final model saves to `models/ppo_turtlebot3.zip`.

**Evaluate** (normal speed, GUI visible):
```bash
ros2 launch turtlebot3_rl evaluation_world.launch.py      # terminal 1
python3 -m turtlebot3_rl.evaluate --episodes 10            # terminal 2
```

Each episode prints its outcome:
```
Episode 1: goal reached, reward=113.21, steps=19
Episode 2: goal reached, reward=110.95, steps=17
...
```

---

## 7. Going faster: simulated time and unthrottled physics

By default this trains at only a handful of simulated steps per real second — a full run would take hours. The fix isn't Gazebo being slow; it's that the control loop originally paced itself against **wall-clock time**, capping throughput at real-time speed no matter how fast physics could actually run.

Two changes fix it, both already applied in this repo:

1. `gym_env.py` waits on **simulated time** via ROS2's `/clock` topic (`use_sim_time=True`), not `time.time()`.
2. `worlds/training_world.world` sets `<real_time_update_rate>0</real_time_update_rate>` — "run physics as fast as the CPU allows," instead of throttling to a 1.0x target.

Result on the reference machine: **4 → 37 simulated fps (~9x)**, cutting a ~7 hour training run to ~45 minutes. Full diagnosis story in [section 10](#10-technical-challenges-this-project-solved).

---

## 8. Extending the base pipeline

The repo's current state layers three additions on top of a minimal working policy — each is a self-contained change worth understanding on its own:

- **Obstacles**: `worlds/training_world.world` / `evaluation_world.world` add four static boxes in a pinwheel layout (open lanes at N/S/E/W so most randomly-sampled goals stay reachable). Requires retraining from scratch — a policy trained in an empty world has never had a reason to learn "avoid things."
- **Visible goal marker**: `_update_goal_marker()` in `gym_env.py` deletes the previous marker and spawns a fresh, **collision-less** green cylinder at each new goal, via the same `/spawn_entity` service already used to place the robot at launch. No collision shape is deliberate — a laser sensor only detects collision geometry, so a solid marker would make the robot's own goal look like a wall to itself right as it arrives.
- **Continuous control**: the action space evolved from a fixed menu of 5 discrete moves to a continuous `Box(2,)` (forward speed + turn rate), letting the policy steer more precisely. Stable-Baselines3's PPO handles this automatically — no algorithm-level changes needed, just retraining.

---

## 9. Troubleshooting / FAQ

**`ros2 run turtlebot3_rl train` fails with `ModuleNotFoundError: No module named 'stable_baselines3'`, even with the venv active.**
The console-script entry point's shebang is baked in at `colcon build` time, pointing at whatever `python3` was on `PATH` *during that build*. If the venv wasn't active then, it points at system Python. Fix: run `python3 -m turtlebot3_rl.train` instead — it uses the currently active interpreter and ignores the shebang — or rebuild with the venv active first.

**`TabError: inconsistent use of tabs and spaces in indentation`** after editing a `.py` file.
Some editors insert a tab when auto-indenting after a paste, and Python refuses to mix tabs and spaces in one block even if they render at the same width. Fix:
```bash
sed -i 's/\t/    /g' path/to/file.py
cat -A path/to/file.py   # tabs show as ^I — confirm none remain
```

**`UserWarning: You are trying to run PPO on the GPU...`**
Expected, already handled (`device="cpu"` in `train.py`/`evaluate.py`). SB3 correctly points out that a small MLP policy doesn't benefit from GPU — see [section 10](#10-technical-challenges-this-project-solved).

**Calling `/gazebo/set_entity_state` hangs forever.**
That service needs a Gazebo plugin (`libgazebo_ros_state.so`) not loaded by TurtleBot3's stock worlds. This repo avoids it — resets use `/reset_world`, and the goal marker is deleted/respawned via `/spawn_entity`/`/delete_entity` instead of moved in place.

**Gazebo's GUI window is blank or frozen on first launch.**
A known one-off race condition between the model spawner and the GUI plugin starting up. `Ctrl+C` and relaunch.

---

## 10. Technical challenges this project solved

**Isaac Sim / Isaac Lab was evaluated first, and ruled out.** It offers GPU-parallelized RL training — meaningfully faster than Gazebo. The GPU on the reference machine (RTX 3070 Laptop, 8GB VRAM) was sufficient. But total system RAM is 16GB against Isaac Sim's stated **32GB minimum** — a hardware ceiling, not something a config change fixes. Gazebo Classic + Stable-Baselines3 was the realistic choice for this hardware tier.

**WSL2 GPU passthrough and GUI rendering had to be verified before investing further.** `nvidia-smi` inside WSL2 confirmed GPU passthrough; launching Gazebo with its GUI (`empty_world.launch.py`) confirmed WSLg could actually render it. Both were unknowns worth checking early, since either failing would have changed the whole approach.

**The biggest performance bug: waiting on wall-clock time, not simulation time.** The control loop originally used `time.time()` to pace each step — a hard cap at real-time speed regardless of how fast Gazebo could actually simulate. Diagnosed via the training log's `fps` field staying near 4 even after going headless. Fixed by subscribing to ROS2's `/clock` topic (`use_sim_time`) and setting Gazebo's `real_time_update_rate` to `0` (unthrottled). Result: **4 → 37 fps (~9x)**, ~7 hours → ~45 minutes for 100k timesteps. Full explanation in [section 7](#7-going-faster-simulated-time-and-unthrottled-physics).

**`/gazebo/set_entity_state` isn't available by default.** It requires a plugin (`libgazebo_ros_state.so`) that TurtleBot3's stock worlds don't load — calling it just hangs. Resolved by using `/reset_world` for episode resets (loaded by default) and `/spawn_entity`/`/delete_entity` (the same plugin family already used to spawn the robot at launch) for the goal marker, rather than adding a new plugin dependency.

**`ros2 run`'s console-script shebang silently pointed at the wrong Python.** The entry-point wrapper script's interpreter path is baked in at `colcon build` time from whatever `python3` was on `PATH` during that specific build — not whatever's active in your current shell. Since the venv wasn't active during the first build, `ros2 run turtlebot3_rl train` kept invoking system Python (no `stable-baselines3`) no matter what was activated afterward. Worked around by invoking via `python3 -m turtlebot3_rl.train` instead, which uses the current interpreter directly.

**PPO's small MLP policy runs worse on GPU than CPU.** Stable-Baselines3 warns about this directly: GPU transfer overhead dominates for a network this small, and it wastes VRAM that Gazebo's renderer wants. Training and evaluation are pinned to `device="cpu"`.

---
