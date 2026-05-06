## E1 Lightning Migration Design

### Summary

Migrate the existing `trainers\e1_smoke_train.py` training flow from a hand-written PyTorch loop to PyTorch Lightning.

This migration is intentionally narrow:
- Only the runnable E1 training path is migrated.
- `trainers.evaluate` remains a separate non-Lightning evaluation entrypoint.
- `E2-E6` trainer placeholders remain unchanged.
- The goal is the minimum change set required to run E1 training under Lightning.

The user explicitly accepts Lightning-native training outputs and does not require backward compatibility for the current experiment artifact layout.

### Current State

The repository currently has:
- an E1 smoke-train entrypoint at `trainers\e1_smoke_train.py`
- manual train and validation loops
- manual AMP handling
- manual progress handling
- manual checkpoint and metrics persistence
- reusable dataset and collate code already separated under `data\`
- reusable model loading logic already separated under `models\teacher.py`

This means the main migration target is the training orchestration layer, not the data/model definitions.

### Goals

1. Replace the manual E1 training loop with a Lightning-based training flow.
2. Preserve the existing E1 core behavior:
   - teacher model loading
   - PIT SI-SNR loss computation
   - validation SI-SNRi computation
   - YAML-driven configuration
3. Keep the training entrypoint simple enough that E1 still runs from a single trainer module.
4. Add only the minimum tests required to prove the Lightning migration works.

### Non-Goals

- Do not migrate `trainers.evaluate`.
- Do not introduce a shared Lightning base class for future experiments.
- Do not migrate `E2-E6`.
- Do not preserve the current custom experiment directory structure exactly.
- Do not redesign the logging stack beyond what Lightning needs to run.

### Recommended Approach

Use a thin Lightning wrapper around the current E1 training logic.

The migration will keep:
- `MiniLibriMixDataset`
- `collate_mini_librimix_batch`
- `compute_batch_pit_sisnr_loss`
- `compute_batch_sisnri`
- pretrained/checkpoint model loading helpers

The migration will replace:
- manual epoch loops
- manual optimizer stepping
- manual AMP scaler management
- manual progress bar handling
- manual best/last checkpoint writing during training

### Target Design

#### 1. Training Module

`trainers\e1_smoke_train.py` will define a Lightning module for E1 training.

Responsibilities:
- create/load the trainable model
- run forward passes
- compute train loss
- compute validation loss
- compute validation SI-SNRi
- configure the optimizer
- emit Lightning-native logs for monitored metrics

The module should stay narrow. It should not absorb unrelated repository concerns.

#### 2. Data Loading

The first migration does not introduce a `LightningDataModule`.

Reasoning:
- existing DataLoader construction is already isolated enough
- adding a DataModule would increase abstraction surface without being necessary for the user goal
- keeping loaders in the entrypoint reduces migration risk

Train and validation DataLoaders will still be constructed directly from config values in the training script.

#### 3. Trainer Construction

The entrypoint will construct a `lightning.pytorch.Trainer` with a minimal configuration derived from the existing YAML.

Primary config mapping:
- `train.epochs` -> `max_epochs`
- `runtime.precision` -> Lightning `precision`
- `runtime.progress_bar` -> `enable_progress_bar`
- `train.max_train_steps_per_epoch` -> `limit_train_batches` when set
- `train.max_val_steps` -> `limit_val_batches` when set

Device selection will still be resolved from the current helper path, then translated into Lightning trainer arguments conservatively. For this migration, correctness matters more than full device abstraction coverage.

#### 4. Checkpointing and Logging

Lightning callbacks will replace the current custom training-loop checkpoint writes.

Expected behavior:
- use `ModelCheckpoint` with monitor and mode sourced from config
- use a simple Lightning logger, preferably `CSVLogger`
- rely on Lightning-native checkpoint filenames and directory layout

The script may still print a short completion summary at the end for CLI usability, but the source of truth for checkpointing and step/epoch orchestration becomes Lightning.

#### 5. Config Compatibility

The existing YAML file remains the entry configuration surface for E1.

Some config fields will remain fully used:
- dataset paths and split config
- batch size
- num workers
- optimizer hyperparameters
- seed
- precision
- monitor and monitor mode

Some fields become compatibility shims rather than manual-loop controls:
- `runtime.progress_bar`
- `train.max_train_steps_per_epoch`
- `train.max_val_steps`

No attempt will be made to preserve fields that only existed to support the old loop internals if Lightning already provides the behavior natively.

### File-Level Changes

Expected edits:
- `trainers\e1_smoke_train.py`
- `requirements.txt`
- `tests\test_e1_smoke_train.py`

Possible small supporting edits:
- `README.md` if the documented training command or artifact expectations become inaccurate

No broader repository reshaping is part of this work.

### Testing Strategy

Keep test coverage focused on migration risk.

#### Tests to Keep

Retain existing pure-function tests where they still describe valid behavior:
- loss helper behavior
- scalar history helpers if they remain
- any summary formatting helpers that survive the migration

#### Tests to Add or Update

1. Smoke test the Lightning E1 entry path with:
   - dummy dataset
   - dummy separation model
   - one epoch
   - one train batch
   - one val batch

2. Assert that:
   - training completes
   - monitored validation metrics are logged under the expected key
   - checkpoint callback is configured from `train.monitor` and `train.monitor_mode`

The tests should avoid requiring GPU and should run entirely on CPU.

### Risks and Mitigations

#### Risk 1: Existing tests are tightly coupled to manual loop details

Mitigation:
- update tests to assert externally meaningful behavior, not internal loop mechanics
- remove or rewrite tests that only validate the old manual progress implementation

#### Risk 2: Precision behavior differs between the old custom AMP logic and Lightning

Mitigation:
- for the first migration, accept Lightning's standard precision handling
- keep CPU test runs on stable precision settings
- avoid overfitting tests to mixed-precision internals

#### Risk 3: Artifact layout changes may invalidate current expectations

Mitigation:
- treat Lightning-native artifacts as the new contract for E1 training
- only update docs that are directly rendered inaccurate by the migration

### Success Criteria

The migration is complete when all of the following are true:

1. `python -m trainers.e1_smoke_train --config ...` still starts E1 training.
2. E1 train/validation execution is handled by Lightning instead of the manual loop.
3. A CPU smoke run can finish with a tiny dataset slice.
4. Minimal automated tests pass for the migrated E1 path.
5. `trainers.evaluate` remains runnable without requiring this migration.

### Implementation Boundary

This design intentionally stops at the first working Lightning migration for E1.

After this lands, later work can decide whether to:
- introduce a shared Lightning base module
- migrate evaluation/testing into Lightning flows
- standardize experiment logging across E1-E6

Those are separate follow-up decisions, not part of this change.
