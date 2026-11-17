# SuperMimic-Phy: force-conditioned dynamics, phase 1

The official worktree and `main` stay at `7e94b073940b7a19b9a82ff01e191cfb35c4ba91`.
All additions are opt-in under `supermimic.phy`; official train/play, models,
learners and the sibling mjlab dependency are unchanged. Use a separate checkout
`supermimic-phy` next to `supermimic` and `mjlab_research`. Keep this research
branch/PR separate from official main. No distributed-training changes.

GPU 0 runs the unmodified official reproduction. GPU 1 runs this phase's tests
and bounded pilots. UUID selection, SSH ports and persistent roots come from
workspace `.env`, not committed machine-specific constants. Dependencies stay
in the shared persistent runtime. Do not reinstall the research checkout into
the baseline's editable environment: prepend this checkout's `src` to
`PYTHONPATH`, and use `uv run --project ../supermimic --no-sync`.

## Data collection

`python -m supermimic.phy.collect --motion-file <npz> --checkpoint-dir <official-pilot> --output <persistent-data>/phy/pilot.pt`

Start with the frozen official pilot actor (rep-state, history 1 by default).
For other actors pass their observation/history configuration explicitly.
Actor weights are frozen; its TensorDict contains only goals and original
observation history. Force/timer data never enter it. Randomization and control
come from the fixed official G1 training configuration.

External wrench slots have exact body names. First pilot: pelvis only, per-axis
uniform force [-10,10] N, torque zero, active 2-8 control steps, cooldown 10-40,
50% zero events. These are debugging settings, not validated robustness ranges.
Wrenches are world-frame `[Fx,Fy,Fz,Tx,Ty,Tz]` about EACH body's CoM, written via
`write_external_wrench_to_sim` before the complete decimation loop and held
constant during it. Reject conflicting force/impulse/velocity-push events.
Read the actual simulator wrench buffer immediately after writing, before
reset; never log the next pulse or infer force from a modified velocity.

Store explicit `(state,action,sim_ctx,wrench,next_state,terminated,truncated)`
plus motion frame, reward and disturbance timer state. Disable auto-reset so
the label precedes environment reset; manually reset only after cloning it.
Motion-command wraparound can still teleport the state. Mark such transitions
`transition_valid=False` using frame continuity and exclude every affected
scored target, including a terminal one. Other terminal targets remain valid.
The sidecar records body slots, coordinates, units, dt, physical configuration,
actor and motion hashes. Data and outputs must live in persistent workspace
`data`, `checkpoints` and `experiments` directories, not Git.

## World-model pilot

`python -m supermimic.phy.fit --dataset <persistent-data>/phy/pilot.pt --output-dir <persistent-checkpoints>/phy/pilot`

The additional wrench channels are rotated to the current root frame before
the MLP (both force and CoM torque); body slots are retained, with no transport
to a shared point. Fixed scales are 10 N and 1 Nm per axis. Preserve the
official generalized-coordinate integration, Bryson loss, normalization,
gradient clipping, Muon/Adam and CUDA Graph updates. An H-frame window scores
H-1 transitions, matching the release. Each scored step uses its own recorded
force and sim context. Windows cannot cross a reset; terminal transitions have
explicit valid targets and can be the final scored step.

Train two models on identical sampled windows, with identical initialization,
architecture, parameter count, losses and update counts. Conditioned sees the
recorded wrench; blind always sees zero channels. This isolates access to the
force information. It is a different comparison from running force-free physics.
Split entire environment trajectories 75/25 BEFORE sampling, and estimate
normalization only from training trajectories. This tests held-out trajectories
within the same motion/force distribution; it does not establish unseen-motion
or unseen-force-configuration generalization.

Default bounded pilot: 200 updates, batch 64, 3x256 hidden layers. Report named
state loss groups over 1/8/32-step open-loop predictions, separately for zero
and forced windows, plus shuffled-force diagnostics. Save dataset hash, exact
Git commit, split IDs, learning curve and separate conditioned/blind weights.
Pilots validate plumbing, finite predictions and gradients; improvement is an
experimental question, not a pass requirement. Checkpoint files are inference
snapshots, not optimizer-resumable training checkpoints.

## Validation and next stage

Before server execution, push the branch and open a PR. On the exact PR SHA run
official Ruff checks, full pytest, then GPU 1 collection and a bounded paired
fit that passes the CUDA Graph warmup/capture boundary. Confirm the imported
`supermimic.__file__` resolves to the research checkout and baseline Git remains
clean. Record commands, SHA, reports and limitations in the PR.

Phase 2 is NOT implemented here: disturbance schedules inside differentiable
policy rollouts and alternating world-model/policy updates. Do not infer
closed-loop robustness from phase-1 prediction metrics. Formal comparisons,
force curricula, multiple slots, held-out force settings, multiple seeds and
policy retraining follow after phase-1 review.
