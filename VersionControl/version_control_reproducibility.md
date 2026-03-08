# Version Control & Reproducibility for ML/LLM Projects

> **Audience:** ML engineers and LLM practitioners  
> **Goal:** Build a reproducible, version-controlled ML workflow using Git, DVC, and Hydra  
> **Stack:** Git · DVC · Hydra · Python

---

## Table of Contents

1. [Why Reproducibility Matters in ML](#1-why-reproducibility-matters-in-ml)
2. [Git Workflow for ML/LLM Projects](#2-git-workflow-for-mlllm-projects)
3. [DVC — Data & Model Versioning](#3-dvc--data--model-versioning)
4. [Pulling Pre-Versioned Artifacts from DVC Remote](#4-pulling-pre-versioned-artifacts-from-dvc-remote)
5. [Switching Between Model Versions](#5-switching-between-model-versions)
6. [Hydra — Config Management Without Code Changes](#6-hydra--config-management-without-code-changes)
7. [Overriding Configs via Hydra CLI](#7-overriding-configs-via-hydra-cli)
8. [Rolling Back to a Previous Version](#8-rolling-back-to-a-previous-version)
9. [Putting It All Together — Full Workflow](#9-putting-it-all-together--full-workflow)
10. [Hands-On Exercises](#10-hands-on-exercises)
11. [Quick Reference Cheatsheet](#11-quick-reference-cheatsheet)

---

## 1. Why Reproducibility Matters in ML

In traditional software, the code *is* the artifact. In ML projects, there are **three equally important artifacts** that must all be versioned together:

| Artifact | Examples | Problem if unversioned |
|---|---|---|
| **Code** | training scripts, pipelines | Can't reproduce the experiment |
| **Data** | datasets, tokenized corpora | Results differ on different data splits |
| **Models** | checkpoints, weights, configs | Can't roll back to a working version |

A common pain point: you train a great model on Monday, modify data preprocessing on Wednesday, and can't reproduce Monday's results by Friday. Reproducibility tooling solves this.

**The core principle:** every model artifact should be traceable back to a specific combination of code commit + data version + config snapshot.

---

## 2. Git Workflow for ML/LLM Projects

### 2.1 What Belongs in Git

Git tracks **code and configs**, but *not* large binary files. Attempting to commit model weights or large datasets directly causes slow clones, bloated history, and merge conflicts on binary files.

```
# .gitignore — things that NEVER go into Git
data/raw/
data/processed/
models/
*.pt
*.ckpt
*.bin
*.h5
mlruns/
__pycache__/
.env
```

What *should* be committed: Python scripts, Hydra config YAMLs, DVC `.dvc` pointer files, `requirements.txt`, CI configs.

### 2.2 Recommended Branch Strategy

```
main              ← stable, production-ready
├── develop       ← integration branch
│   ├── feat/new-tokenizer
│   ├── feat/rlhf-reward-model
│   └── experiment/larger-context-window
└── hotfix/fix-inference-bug
```

**Experiment branches** are a natural fit for ML. Each experiment lives on its own branch, complete with its own DVC-tracked data pointer and Hydra config. When an experiment succeeds, it gets merged into `develop`.

### 2.3 Tagging Model Versions

Use Git tags to mark the exact commit associated with a model release:

```bash
git tag -a v1.2.0 -m "Fine-tuned on customer support dataset v3"
git push origin v1.2.0
```

Later you can return to any tagged state: `git checkout v1.2.0`.

### 2.4 Commit Message Convention for ML

Use a structured format so the git log reads like a lab notebook:

```
[train] Fine-tune LLaMA-3 on QA dataset v2 — val_loss=0.312

- Changed learning rate from 2e-5 to 1e-5
- Added gradient checkpointing
- DVC: data/qa_v2.dvc, models/llama3_ft_v2.dvc
```

Prefixes like `[train]`, `[data]`, `[eval]`, `[config]`, `[fix]` make filtering history easy.

---

## 3. DVC — Data & Model Versioning

### 3.1 What DVC Is

DVC (Data Version Control) is a version control system for ML assets. It stores the *content* of large files in a remote storage backend (S3, GCS, Azure Blob, SSH, local), while storing only a small **pointer file** (`.dvc`) in Git.

Think of it this way:

- Git stores: `models/best_model.dvc` (a few hundred bytes, a YAML with a hash)
- DVC remote stores: the actual `best_model.pt` (several gigabytes)

The `.dvc` file looks like this:

```yaml
# models/best_model.dvc
outs:
- md5: a1b2c3d4e5f6...
  size: 4831838208
  path: best_model.pt
```

When you `git checkout` a commit, you get the right `.dvc` pointer. Then `dvc pull` fetches the corresponding file from the remote.

### 3.2 Initializing DVC in a Project

```bash
pip install dvc dvc-s3   # or dvc-gcs, dvc-azure, etc.

git init
dvc init
git commit -m "Initialize DVC"
```

### 3.3 Adding a Remote Storage

```bash
# Example: AWS S3 bucket
dvc remote add -d myremote s3://my-ml-bucket/dvc-cache

# Example: local filesystem (good for testing)
dvc remote add -d localremote /mnt/shared/dvc-cache

git add .dvc/config
git commit -m "Add DVC remote"
```

### 3.4 Tracking a Model or Dataset

```bash
# Track a model checkpoint
dvc add models/best_model.pt

# This creates models/best_model.dvc and updates .gitignore
git add models/best_model.dvc models/.gitignore
git commit -m "[model] Add best_model checkpoint v1"

# Push the actual file to remote storage
dvc push
```

### 3.5 DVC Pipelines (Optional but Powerful)

DVC can also track entire pipelines with `dvc.yaml`, defining stages (preprocess → train → evaluate) with their inputs, outputs, and parameters. Running `dvc repro` re-runs only the stages whose inputs have changed, similar to `make`.

```yaml
# dvc.yaml
stages:
  train:
    cmd: python train.py
    deps:
      - train.py
      - data/train.csv
    params:
      - params.yaml:
          - model.lr
          - model.batch_size
    outs:
      - models/checkpoint.pt
    metrics:
      - metrics.json
```

---

## 4. Pulling Pre-Versioned Artifacts from DVC Remote

This is the most common operation in day-to-day work: you want the model associated with a specific Git commit or tag.

### 4.1 Basic Pull

```bash
# Make sure you have the right code version
git checkout v1.2.0

# Pull all DVC-tracked files for this commit
dvc pull

# Pull a specific file only
dvc pull models/best_model.pt
```

`dvc pull` = `dvc fetch` (download to local DVC cache) + `dvc checkout` (link from cache to workspace).

### 4.2 What Happens Under the Hood

```
Git tag v1.2.0
    └── models/best_model.dvc  [md5: a1b2c3...]
                    │
                    ▼
         DVC remote (S3)
             └── files/md/a1/b2c3...  ← actual weights file
                    │
             dvc pull
                    │
                    ▼
         Local DVC cache (~/.dvc/cache)
             └── files/md/a1/b2c3...
                    │
             dvc checkout
                    │
                    ▼
         Your workspace: models/best_model.pt
```

### 4.3 Pulling in CI/CD

In a deployment pipeline, configure DVC credentials via environment variables:

```bash
# In your CI environment (GitHub Actions, GitLab CI, etc.)
export AWS_ACCESS_KEY_ID=${{ secrets.AWS_KEY }}
export AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET }}

git checkout $DEPLOY_TAG
dvc pull models/
```

### 4.4 Checking What You Have

```bash
# See the status of tracked files
dvc status

# List remote contents
dvc ls --dvc-only
```

---

## 5. Switching Between Model Versions

One of the most powerful workflows: switching between model versions without manually managing files.

### 5.1 The Pattern

Each model version corresponds to a Git commit or tag where the `.dvc` pointer was updated. Switching versions is just two commands:

```bash
# Switch to an older model version
git checkout v1.1.0          # gets the old .dvc pointer
dvc pull models/best_model.pt   # fetches the old weights

# Switch to the latest
git checkout main
dvc pull models/best_model.pt
```

### 5.2 Viewing Version History

```bash
# See all commits that touched the model pointer
git log --oneline -- models/best_model.dvc

# Example output:
# a3f91bc [model] v1.3 — added RLHF fine-tuning
# 8c2d441 [model] v1.2 — trained on larger dataset
# 1e09f32 [model] v1.1 — baseline fine-tune
```

### 5.3 Comparing Two Versions

```bash
# See what changed in the pointer between two tags
git diff v1.1.0 v1.2.0 -- models/best_model.dvc

# See metric differences (if tracked)
dvc metrics diff v1.1.0 v1.2.0

# Example output:
# Path         Metric    v1.1.0    v1.2.0    Change
# metrics.json val_loss  0.412     0.318     -0.094
```

### 5.4 Running Multiple Versions Side by Side

For A/B testing, use DVC's `get` command to download a specific version without switching your Git state:

```bash
# Download a specific version to a named path (doesn't affect your workspace)
dvc get . models/best_model.pt --rev v1.1.0 -o models/model_v1.1.pt
dvc get . models/best_model.pt --rev v1.2.0 -o models/model_v1.2.pt
```

---

## 6. Hydra — Config Management Without Code Changes

### 6.1 The Problem Hydra Solves

ML experiments require running the same code with many different configurations: different learning rates, model architectures, datasets, inference parameters. Without a config system, you end up with:

- Hardcoded values scattered in scripts
- Long argparse argument lists
- Config files copied and renamed (config_v1.yaml, config_final.yaml, config_ACTUALLY_final.yaml)

Hydra provides a **structured, composable config system** where configs are YAML files that can be overridden from the command line, composed from multiple files, and automatically logged with each run.

### 6.2 Project Structure with Hydra

```
my_ml_project/
├── conf/
│   ├── config.yaml          ← main config (entry point)
│   ├── model/
│   │   ├── llama3.yaml
│   │   └── mistral.yaml
│   ├── data/
│   │   ├── squad.yaml
│   │   └── custom_qa.yaml
│   └── training/
│       ├── default.yaml
│       └── fast_debug.yaml
├── train.py
└── infer.py
```

### 6.3 Config Files

```yaml
# conf/config.yaml — the root config
defaults:
  - model: llama3          # load conf/model/llama3.yaml
  - data: squad            # load conf/data/squad.yaml
  - training: default      # load conf/training/default.yaml
  - _self_                 # allow local overrides

experiment_name: baseline_run
output_dir: outputs/${experiment_name}
```

```yaml
# conf/model/llama3.yaml
name: meta-llama/Llama-3-8B
checkpoint_path: models/llama3_ft.pt
max_length: 2048
```

```yaml
# conf/training/default.yaml
lr: 2e-5
batch_size: 8
epochs: 3
warmup_steps: 100
fp16: true
```

### 6.4 Using Hydra in Python

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print(f"Training {cfg.model.name} on {cfg.data.dataset_name}")
    print(f"LR: {cfg.training.lr}, Batch: {cfg.training.batch_size}")
    
    # Access nested config naturally
    model = load_model(cfg.model.checkpoint_path, cfg.model.max_length)
    optimizer = Adam(model.parameters(), lr=cfg.training.lr)

if __name__ == "__main__":
    train()
```

Hydra automatically creates a timestamped output directory for each run and saves the full resolved config there — you can always reconstruct exactly what settings produced a given result.

---

## 7. Overriding Configs via Hydra CLI

This is where Hydra shines: changing any config value without touching a single file.

### 7.1 Basic Overrides

```bash
# Override a single value
python train.py training.lr=1e-4

# Override multiple values
python train.py training.lr=1e-4 training.batch_size=16

# Override nested values
python train.py model.max_length=4096
```

### 7.2 Switching Config Groups

```bash
# Use a different model config entirely
python train.py model=mistral

# Use different data + training combo
python train.py data=custom_qa training=fast_debug
```

### 7.3 Multirun — Sweeping Hyperparameters

```bash
# Run training with 3 different learning rates (sequential)
python train.py --multirun training.lr=1e-5,2e-5,5e-5

# Run a 2D grid search
python train.py --multirun training.lr=1e-5,2e-5 training.batch_size=8,16
# This runs 4 jobs: (1e-5, 8), (1e-5, 16), (2e-5, 8), (2e-5, 16)
```

### 7.4 Appending and Deleting Config Values

```bash
# Add a new key that doesn't exist in YAML
python train.py +training.gradient_clip=1.0

# Remove a key (set to null effectively removes it from the config)
python train.py ~training.warmup_steps
```

### 7.5 Using a Different Config File Entirely

```bash
# Load from a completely different config file
python train.py --config-name experimental_config
```

### 7.6 Inspecting the Resolved Config

```bash
# Print what the config would look like (dry run, no execution)
python train.py --cfg job

# Print the hydra-internal config
python train.py --cfg hydra
```

---

## 8. Rolling Back to a Previous Version

Rolling back in this stack means coordinating Git and DVC together.

### 8.1 Full Rollback (Code + Data + Model)

```bash
# Step 1: Find the version you want
git log --oneline
# e.g.: 8c2d441 [model] v1.2 — trained on larger dataset

# Step 2: Checkout the code and config pointers
git checkout 8c2d441

# Step 3: Pull the corresponding DVC artifacts
dvc pull

# You now have:
# - The code as it was at that commit
# - The model weights from that time
# - The Hydra configs from that time
```

### 8.2 Rollback Model Only (Keep Current Code)

Sometimes you want to revert just the model artifact while keeping recent code changes:

```bash
# Get the old model pointer
git checkout 8c2d441 -- models/best_model.dvc

# Pull the corresponding weights
dvc pull models/best_model.pt

# Your code is still at HEAD, but you're using the old model
```

### 8.3 Rollback Data Only

```bash
git checkout 8c2d441 -- data/train.dvc data/val.dvc
dvc pull data/
```

### 8.4 Creating a New "Good Version" Branch from an Old State

When a rollback is permanent (i.e., you want to diverge from the bad commits), create a new branch:

```bash
git checkout v1.2.0         # go to the known-good tag
git checkout -b hotfix/revert-bad-model   # branch from here
dvc pull                    # get artifacts for this state
# ... make any fixes ...
git push origin hotfix/revert-bad-model
```

### 8.5 Tagging the Rolled-Back Version

After a rollback, tag the state so it's easy to find:

```bash
git tag -a v1.2.1-stable -m "Rollback to v1.2 model after v1.3 regression"
git push origin v1.2.1-stable
```

---

## 9. Putting It All Together — Full Workflow

Here's the complete day-to-day workflow combining all tools:

### 9.1 Starting a New Experiment

```bash
# 1. Create experiment branch
git checkout -b experiment/larger-lr

# 2. Run training with modified config (no file edits!)
python train.py training.lr=5e-5 training.epochs=5

# 3. Evaluate — model checkpoint is in outputs/ (Hydra auto-creates it)
python eval.py model.checkpoint_path=outputs/2024-01-15_14-30/checkpoint.pt

# 4. If results are good, track the model with DVC
dvc add outputs/2024-01-15_14-30/checkpoint.pt
mv outputs/.../checkpoint.pt.dvc models/larger_lr_v1.dvc
dvc push

# 5. Commit everything
git add models/larger_lr_v1.dvc conf/ train.py
git commit -m "[train] LR=5e-5 experiment — val_loss=0.291 (↓ from 0.312)"
git tag -a experiment/larger-lr-v1 -m "Larger LR experiment checkpoint"
```

### 9.2 Deploying a Model

```bash
# In your deployment script or CI pipeline:

# 1. Checkout the approved version
git checkout v1.3.0-release

# 2. Pull model artifacts
dvc pull models/

# 3. Start the inference server using Hydra config
python serve.py model=production data=production
```

### 9.3 Collaborating with a Teammate

```bash
# Teammate pushes a new model version
git pull
git checkout feat/new-architecture

# Get their DVC-tracked model
dvc pull

# Run their training setup with your local data override
python train.py data=my_local_data
```

---

## 10. Hands-On Exercises

### Exercise 1: Pull a Trained Model from DVC

**Setup:** Clone a project that already has DVC configured.

```bash
git clone https://github.com/your-org/ml-project
cd ml-project

# Configure DVC remote credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Pull model artifacts
dvc pull models/

# Verify the model file is present
ls -lh models/best_model.pt
```

**Checkpoint:** You should see the model file locally. Running `dvc status` should report "Data and pipelines are up to date."

---

### Exercise 2: Switch Between Model Versions

```bash
# See available tagged versions
git tag -l

# Check what model is currently tracked
cat models/best_model.dvc

# Switch to an older version
git checkout v1.1.0
dvc pull models/best_model.pt
python eval.py  # evaluate old model

# Note the metrics, then switch back
git checkout v1.2.0
dvc pull models/best_model.pt
python eval.py  # compare metrics

# Use dvc metrics diff to compare
dvc metrics diff v1.1.0 v1.2.0
```

**Checkpoint:** You've run evaluation on two different model versions and compared their metrics without manually managing any files.

---

### Exercise 3: Override Configs via Hydra

```bash
# Run with defaults
python train.py --cfg job  # inspect the config

# Change learning rate
python train.py training.lr=1e-4

# Switch to a different model
python train.py model=mistral

# Run a quick debug configuration
python train.py training=fast_debug experiment_name=debug_run

# Sweep three learning rates
python train.py --multirun training.lr=1e-5,2e-5,5e-5

# Check what was saved in outputs/
ls outputs/
cat outputs/2024-01-15_14-30/.hydra/config.yaml
```

**Checkpoint:** Each run in `outputs/` has its own timestamped directory with the full config saved inside it.

---

### Exercise 4: Rollback to a Previous Version

**Scenario:** A new model was deployed but causes a regression in production. Roll back to the last stable version.

```bash
# Identify the last stable tag
git log --oneline --tags

# Rollback model only (keep current code)
git checkout v1.2.0-stable -- models/best_model.dvc
dvc pull models/best_model.pt

# Verify you have the right model
python eval.py  # should show the better metrics

# Create a hotfix branch + tag to document the rollback
git checkout -b hotfix/rollback-to-v1.2
git commit -m "ROLLBACK: revert to v1.2 model due to v1.3 regression"
git tag -a v1.2.1-hotfix -m "Emergency rollback"
git push origin v1.2.1-hotfix
```

**Checkpoint:** Your evaluation metrics match those from v1.2, and the rollback is documented in Git history.

---

## 11. Quick Reference Cheatsheet

### Git Commands

| Action | Command |
|---|---|
| Create experiment branch | `git checkout -b experiment/name` |
| Tag a model version | `git tag -a v1.0.0 -m "message"` |
| View model history | `git log --oneline -- models/model.dvc` |
| Checkout specific version | `git checkout v1.0.0` |
| Rollback single file | `git checkout v1.0.0 -- models/model.dvc` |

### DVC Commands

| Action | Command |
|---|---|
| Track a new file | `dvc add path/to/file` |
| Push to remote | `dvc push` |
| Pull all artifacts | `dvc pull` |
| Pull specific file | `dvc pull models/model.pt` |
| Check status | `dvc status` |
| Compare metrics | `dvc metrics diff v1.0.0 v1.1.0` |
| Get file at specific rev | `dvc get . file --rev v1.0.0 -o output_path` |

### Hydra Commands

| Action | Command |
|---|---|
| Override a value | `python script.py key=value` |
| Switch config group | `python script.py group=config_name` |
| Add new key | `python script.py +key=value` |
| Remove a key | `python script.py ~key` |
| Run hyperparameter sweep | `python script.py --multirun key=v1,v2,v3` |
| Inspect resolved config | `python script.py --cfg job` |
| Use different config file | `python script.py --config-name other_config` |

### The Golden Rule

> **Always commit `.dvc` files to Git immediately after `dvc add` and `dvc push`.** A `.dvc` pointer without a corresponding Git commit is a dangling reference — it won't be reproducible.

```bash
dvc add models/checkpoint.pt
dvc push
git add models/checkpoint.pt.dvc
git commit -m "[model] Add checkpoint v1.3"
git push
```

---

*Next topic: CI/CD for ML pipelines — automated training, evaluation, and deployment.*
