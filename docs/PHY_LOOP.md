# SuperMimic-Phy bounded training loop

This phase builds on the separate phase-1 branch/PR. Official `main`, the
`supermimic` worktree, mjlab, installed editable packages and the running
official reproduction stay unchanged. Keep this PR on the research branch;
do not merge research changes into official main.

## Loop and information boundary

`python -m supermimic.phy.train` runs:

1. Reset MuJoCo, collect a fresh replay warmup with the current actor, and
   update normalizers outside inference mode.
2. Bootstrap the force-conditioned world model (new runs only).
3. Repeatedly collect with the **updated actor**, update the world model from
   explicit force-labelled targets, and update actor/critic through forced
   differentiable model rollouts.
4. Save complete learned/optimizer state and optionally a chronological replay
   snapshot. Reload into a fresh process to resume or evaluate in MuJoCo.

Production changes are confined to `supermimic.phy`. The official reward,
termination probabilities, sensor history, actor architecture, BPTT survival
masking, gradient clipping and critic/GAE implementation are reused. This pilot
uses uniform motion sampling and a small network/budget; it is not a matched
scientific comparison to the full official reproduction.

Real forces are written before all physics substeps. Explicit next states are
cloned before manual resets; command teleports are flagged invalid. Online WM
windows never cross reset or score invalid targets. Sensor-history windows
also pad at invalid-command boundaries. Each stored transition has the actor
update-call number, showing which policy collected it. Snapshot files contain
only filled frames in chronological order, with force/time/body conventions.

A policy rollout starts with the replay snapshot's actual current force and
pulse timers. Outside CUDA Graph capture, an independent generator continues
that pulse and samples H future force/timer states. No gradients go into this
exogenous sampling. Every dynamics step receives its matching world-frame
wrench, rotated by the **predicted** current orientation inside ForceDynamics.
The actor receives only its original goals and observation history. The critic
also receives normalized **current** force and active/cooldown timers, so its
bootstrap is conditioned on the correct disturbance state. Neither network
receives the future force plan. H+1 states cover the final bootstrap.

During policy updates, world-model/representation parameters are frozen while
state/action Jacobians remain differentiable. Runtime checks require finite
metrics, changed WM/actor weights, unchanged frozen WM weights, and subsequent
collection with a newer actor. CUDA Graph state includes the timer scale.
The official CapturedUpdate warmup/capture execution semantics are preserved;
reported update-call counts are not claimed to be exact optimizer step counts.

The disturbance `enabled` flag is now enforced by both physical and imagined
forces. Setting it false clears the actual wrench; it is used for zero-force
evaluation without changing learned network geometry.

## Checkpoints and resume

Atomic checkpoints contain config, source commit, motion hash, WM/context,
actor, critic, every optimizer, progress counters and Torch/Python/NumPy/force
generator RNG state. Loading verifies exact model and optimizer state equality
before continuing. A tensor learning rate is moved to the correct device
before rebuilding CUDA Graphs. Phase-1 WM-only files are deliberately rejected
as full training checkpoints.

Simulator integrator/contact state and replay are **not** restored. Resume
creates a fresh simulator, resets sensor history, and refills replay with the
restored actor before further learning. It preserves learned/optimizer state
and progress, not bitwise continuation of a physical trajectory. A new resume
output directory keeps the original checkpoint intact. Training resume requires
the same motion contents. Evaluation performs no learning or normalization
updates and verifies that all saved learned state remains unchanged.

## Small GPU verification

Use the active runtime and paths from workspace `.env`. Run only on the
research GPU, with imports explicitly pointing at this checkout. Do not run
`uv sync` or reinstall the editable package into the baseline environment.

```bash
source "$REMOTE_ACTIVE_ENV_FILE"
cd "$REMOTE_PHY_REPO"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="$REMOTE_GPU1_UUID"

uv run --project ../supermimic --no-sync python -m supermimic.phy.train \
  --motion-file "$REMOTE_DATA_DIR/lafan1/npz/run1_subject5.npz" \
  --output-dir "$REMOTE_CHECKPOINT_DIR/phy/loop-pilot/train" \
  --replay-output "$REMOTE_DATA_DIR/phy/loop-pilot/train.pt"

uv run --project ../supermimic --no-sync python -m supermimic.phy.train \
  --resume "$REMOTE_CHECKPOINT_DIR/phy/loop-pilot/train/checkpoint.pt" \
  --cycles 4 --output-dir "$REMOTE_CHECKPOINT_DIR/phy/loop-pilot/resume" \
  --replay-output "$REMOTE_DATA_DIR/phy/loop-pilot/resume.pt"

uv run --project ../supermimic --no-sync python -m supermimic.phy.train \
  --evaluate "$REMOTE_CHECKPOINT_DIR/phy/loop-pilot/resume/checkpoint.pt" \
  --output-dir "$REMOTE_EXPERIMENT_DIR/phy/loop-pilot/eval-forced"

uv run --project ../supermimic --no-sync python -m supermimic.phy.train \
  --evaluate "$REMOTE_CHECKPOINT_DIR/phy/loop-pilot/resume/checkpoint.pt" \
  --force-mode zero --output-dir "$REMOTE_EXPERIMENT_DIR/phy/loop-pilot/eval-zero"
```

Default verification: 16 real envs, 32 warmup control steps, 4 initial WM
update calls, 6 collect/WM/policy cycles (8 real steps each), batches 16, WM
window 8 frames, policy horizon 4, proprio history 3, two 64-wide hidden layers.
Pelvis force is [-10,10] N per axis, zero torque, active 2-4 steps, cooldown 2-6,
25% zero events. These are plumbing settings, not validated force curricula.
The CLI bounds budgets to small pilots; no formal experiments are scheduled.

Before server validation, push and open a stacked PR targeting the phase-1
research branch. On the exact PR SHA run official Ruff checks and full pytest
(set `SUPERMIMIC_PHY_GPU_MOTION` for the controlled real-G1 force test), followed
by the four fresh-process commands above. Record source SHA, results and
limitations in the PR. Require WM and policy CUDA Graphs captured in initial
and resume runs, finite reloaded predictions, frozen WM weights during policy
updates, and multiple actor versions present in stored replay.

This establishes a working small training loop, checkpoint continuation and
simulator inference. It does not establish long-run stability, tracking quality,
force prediction gains, OOD/multi-body/torque generalization or robustness.
