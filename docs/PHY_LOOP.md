# Force input in the existing SuperMimic pipeline

Use the original `src/supermimic/scripts/train.py` entry point and the unchanged
`SuperMimicRunner.run()` loop. Research changes live only in the separate
`supermimic-phy` worktree/branch; official `supermimic`, main and mjlab remain
frozen. This PR is stacked on phase 1 and is not merged into official main.

`--runner.force-conditioned True` opts into four connections:

- Create ForceTrackingEnv and record actual per-step wrench, explicit pre-reset
  next state and command-transition validity. Preserve real termination flags
  before manual reset; pad histories at reset/teleport boundaries.
- Build phase-1 ForceDynamics/ForceWorldModelLearner and force-aware replay.
- Use a transition hook in WorldModelEnv: an exogenous H+1 pulse plan continues
  the sampled replay force/timers, then samples future pulses outside capture.
  Each prediction uses its matching force. Existing reward, soft termination,
  actor-critic/BPTT, gradient clipping, update ratio and reset sampling remain
  shared. Only the critic additionally sees current force/timers; the actor
  never receives the force plan. WM parameters freeze during policy updates,
  while state/action derivatives remain available.
- Keep original W&B logging and world_model.pt/actor.pt/critic.pt saving.
  ForceDynamics accepts omitted wrench as zero for existing force-free playback.

There is no separate scheduler, training CLI, or optimizer checkpoint system.
Disabled force mode takes the original path. The disturbance enabled flag now
also correctly zeroes physical and imagined forces during zero-force checks.

Small verification uses the existing entry: 16 envs, 81 control steps, buffer
128, replay ratio 1, WM batch 16/window 8, policy batch 16/horizon 4, two hidden
layers of 64, proprio history 3. This exercises roughly ten alternating update
calls and passes both CUDA Graph capture boundaries. Force is pelvis-only,
[-10,10] N per axis, zero torque; no formal experiments or robustness claims.

Tests `test_phy_pipeline.py` observe the unmodified run loop while invoking the
actual CLI, check nonzero WM/actor gradients and parameter updates, frozen WM
weights during policy updates, newer actor weights used for collection, finite
force-labelled replay, and original saved model files. A separate pytest
process reconstructs models/config from those files and runs 32 forced and 32
zero-force simulator steps with normalization/learning disabled. No optimizer
resume is added; saved model reload is the existing checkpoint contract.

Use the existing persistent runtime without uv sync/reinstall. Set PYTHONPATH
to this research checkout's src and CUDA_VISIBLE_DEVICES to the research GPU
UUID from workspace .env. Keep motion data in REMOTE_DATA_DIR, model files in
REMOTE_CHECKPOINT_DIR and logs/reports in REMOTE_EXPERIMENT_DIR. Push/open PR
before server checks, and record exact tested SHA and results there.
