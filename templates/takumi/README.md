# Takumi template

AI Scientist v1 template that treats the **Skills format components** as the
experimental variable. Each "idea" proposes a combination of Skills sections
(S3..S12). The inner loop trains a tiny **Novice** network through K cycles of
LLM-driven mutation, guided by a Skills document the **Expert's Articulator**
wrote. The fitness is the mean reward over the configured evaluation episodes
in SlimeVolley against the built-in opponent.

The design is documented in `design_spec/04_takumi_framework_v4.md` and the
day-2 implementation guide `design_spec/05_day2_implementation_tasks.md`.

## What is Aider-controlled vs experimenter-controlled

| Layer | What | Who edits |
|---|---|---|
| `FORMAT_HYPOTHESIS["components"]` | which S3..S12 are included | **Aider** (outer loop) |
| `FORMAT_HYPOTHESIS["instructions"]` | free-text hint for Articulator | **Aider** (outer loop) |
| `config.yaml::experiment.stage` | 1 or 2 (Articulator access mode) | **Experimenter** |
| `config.yaml::env.backend` | `"jax"` (default) / `"gym"` (legacy) | **Experimenter** |
| `config.yaml::inner_loop.*` | K, eval_episodes, seed, subsample | **Experimenter** |
| `config.yaml::visualization.*` | per-cycle topology PNGs + learning curve | **Experimenter** |
| `config.yaml::models.*` | which Claude model for each role | **Experimenter** |
| `config.yaml::*_params` | temperature, max_tokens, retries | **Experimenter** |
| `config.yaml::budget.*` | $70 warn / $95 halt thresholds | **Experimenter** |
| `lib/*` | network forward, env, validators | **Experimenter** (rare) |

Stage is intentionally **not** an Aider-controlled axis: it is an experimental
condition the researcher fixes, so that Stage 1 vs Stage 2 ablations are clean
(not entangled with which components were chosen).

## Layout

```
templates/takumi/
├── config.yaml              ← experimenter's source of truth
├── prompt.json              ← researcher persona + task description
├── seed_ideas.json          ← initial Skills format hypotheses (components only)
├── experiment.py            ← Aider-editable: only the FORMAT_HYPOTHESIS dict
├── plot.py                  ← Aider-editable: visualization
├── lib/                     ← FIXED infrastructure (do NOT let Aider edit)
│   ├── config.py            ← config.yaml loader (typed dataclass)
│   ├── network.py           ← Network YAML schema, forward, mutations
│   ├── env_wrapper.py       ← SlimeVolleyGym (gym, legacy) adapter
│   ├── jax_env.py           ← EvoJAX SlimeVolley (JAX, default) adapter
│   ├── expert_loader.py     ← NEAT champion → YAML, expert trajectory
│   ├── articulator.py       ← Stage 1/2 Skills generator
│   ├── mutator.py           ← network mutation proposer
│   ├── eval.py              ← Inner Loop orchestrator
│   ├── visualize.py         ← network topology / learning curve PNGs
│   └── budget.py            ← cumulative USD tracker
├── data/
│   ├── expert_genome.yaml   ← NEAT champion in our YAML schema
│   ├── expert_trajectory.yaml ← top-5 expert episodes (subsample N=10)
│   └── sample_network.yaml
├── run_0/final_info.json    ← A0 baseline (S1+S2 only)
└── latex/                   ← writeup template
```

## Environment backends

| backend | source | when to use |
|---|---|---|
| `jax` (default) | `ref_implementations/evojax/evojax/task/slimevolley.py` (Part 1 流用) | 100-ep evaluation で `vmap` 並列が効く. `test=False/True` で 3000-step 全プレイ vs 5-life mode を正しく区別する. |
| `gym` | `slimevolleygym` (PyPI) | legacy / debug. 5-life 固定で trajectory 収集が安定. |

JAX backend は ``ref_implementations/evojax/`` をローカル import する
(``pip install evojax`` 不要). 別パスを使いたいときは
``TAKUMI_EVOJAX_PATH`` env var で上書きする.

## Running a single idea

```bash
# real run
export ANTHROPIC_API_KEY=sk-...
python experiment.py --out_dir=run_1

# offline mock (deterministic, no LLM cost)
python experiment.py --out_dir=run_test --mock

# alternate config (e.g. stage=2 ablation suite)
python experiment.py --out_dir=run_s2_a1 --config=config_stage2.yaml
```

The script writes:

- `run_<i>/final_info.json` — AI Scientist contract `{name: {means: {...}}}`
- `run_<i>/skills.md` — Articulator output for this run
- `run_<i>/novice_init.yaml`, `novice_best.yaml`, `novice_final.yaml`
- `run_<i>/inner_loop_debug.json` — per-cycle records
- `run_<i>/cost_log.json` — USD usage breakdown
- `run_<i>/learning_curve.png` — fitness trajectory + complexity plot
- `run_<i>/topology/cycle_init.png`, `cycle_NNN.png` — network after each mutation
- `run_<i>/best_play.gif` — best Network のプレイ動画 (n_trials 試合中の最高 score 試合)
  (すべて `config.yaml::visualization.*` で個別に toggle 可能)

## Stage 1 vs Stage 2 comparison workflow

The recommended workflow is to run the outer loop **twice** with different
`config.yaml::experiment.stage` values:

```bash
# 1. Stage 1 ablation suite
cp config.yaml config_stage1.yaml
# edit: experiment.stage: 1, run_label: stage1_ablation
TAKUMI_CONFIG=config_stage1.yaml python launch_scientist.py --experiment takumi --num-ideas 5

# 2. Stage 2 ablation suite (same idea space, different access mode)
cp config.yaml config_stage2.yaml
# edit: experiment.stage: 2, run_label: stage2_ablation
TAKUMI_CONFIG=config_stage2.yaml python launch_scientist.py --experiment takumi --num-ideas 5
```

Then compare `delta_score_vs_a0` per components between the two suites — that
is the empirical evidence for the sub-objective ("does cognitive measurement
make transfer more effective?").

## Outer loop (AI Scientist)

```bash
python launch_scientist.py --experiment takumi --num-ideas 1 --skip-novelty-check
```

## Budget

Configured in `config.yaml::budget`:
- Per idea per stage: ≈ $1.0 (Stage 1) — $1.4 (Stage 2)
- `warning_threshold_usd`: 70.0 (printed once)
- `hard_limit_usd`: 95.0 (raises `RuntimeError`)

## Re-generating expert artefacts

Whenever `expart_model/best.npz` changes, re-run from this directory:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from lib.expert_loader import export_expert_yaml, collect_expert_trajectories
from lib.network import load_network
NPZ = '/path/to/expart_model/best.npz'
export_expert_yaml(NPZ, 'data/expert_genome.yaml')
collect_expert_trajectories(load_network('data/expert_genome.yaml'),
    num_episodes=5, save_path='data/expert_trajectory.yaml', base_seed=0)
"
```

Do **not** regenerate inside the inner loop — these are frozen per the
`CLAUDE.md` NG list.
