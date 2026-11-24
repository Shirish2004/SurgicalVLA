# Evaluation guide

How to evaluate an SFT model, an RL model and an SFT+RL model on each benchmark, what the
protocol is, and how to compare models fairly. Training is in [`docs/training.md`](training.md);
the protocol decisions and the baseline numbers are in [`docs/baselines.md`](baselines.md).

## 1. One command for every model

```bash
python scripts/benchmark.py eval --benchmark <libero|libero_plus|robotwin|vlabench> --checkpoint <model> [--name <name>]
```

`--checkpoint` accepts:
- a Hub id (`Shirish24/surgical-vla-libero-sft-upcycled-s0`, `lerobot/smolvla_libero`), with
  `--revision` to pin a version;
- a local training output (`outputs/libero/sft`, `results/libero_train`), which resolves to its
  newest saved policy;
- an exact checkpoint directory (`outputs/libero/rl/final`).

Results go to `results/<benchmark>/<name>/`. `--dry-run` prints the commands, and arguments
after `--` go to `lerobot-eval`.

The evaluated policy loads its **own** config and normaliser from the checkpoint. That is why
every checkpoint saves `policy_preprocessor*` and `policy_postprocessor*`. Rollouts use the
deterministic ODE sampler (no RL noise) and the checkpoint's `n_action_steps` (50 for every
method, the protocol horizon).

## 2. The three cases

| Case | Model to evaluate | Command (LIBERO) | Compare against |
|---|---|---|---|
| **SFT** | final SFT checkpoint | `eval --benchmark libero --checkpoint outputs/libero/sft --name sft` | SmolVLA baseline trained with the same data (`--policy smolvla`) and `lerobot/smolvla_libero` |
| **RL** (`sequential`) | `outputs/libero/rl/final` | `eval --benchmark libero --checkpoint outputs/libero/rl/final --name rl_sequential` | the SFT model it started from |
| **SFT + RL** (`joint` / `interleaved`) | `final/` of that run, trained with `-- training.mode=joint run.out_dir=outputs/libero/rl_joint` | `eval --benchmark libero --checkpoint outputs/libero/rl_joint/final --name rl_joint` | the same SFT model, and the `sequential` run at matched compute |

Rules for all three:
- **Last checkpoint only.** Evaluate the last checkpoint of the fixed budget (`step_<steps>`
  for SFT, `final/` for RL). Validation curves during training never choose a checkpoint.
- **Evaluate the SFT model first.** Its number is the reference for the RL gain and for
  forgetting.
- **Report every suite for RL models**, including suites that RL did not train on.
  `forgetting = SFT − RL` on those suites.
- **Seeds:** 3 training seeds per arm when comparing methods (pilot), each evaluated with the
  full protocol. Report all of them.

## 3. Per-benchmark protocol

| Benchmark | Episodes | Seeds | Simulator | Output |
|---|---|---|---|---|
| **LIBERO** | 4 suites (spatial, object, goal, 10) × 10 tasks × all 50 benchmark init states | 1000, 2000, 3000 | LeRobot's canonical Docker image (default); `--local` uses the `sim-eval` env, which is not the protocol | `results/libero/<name>/<suite>/seed*/eval_info.json`, `summary.json` |
| **LIBERO-Plus** | every registered variant of the 4 perturbed suites (10,030), 1 episode each | 1000 | LIBERO-Plus Docker image | as LIBERO |
| **RoboTwin 2.0** | 50 tasks × 100 episodes, Easy (demo_clean) | 1000 | LeRobot's RoboTwin image | `results/robotwin/<name>/easy/seed1000/` |
| **VLABench** | 21 primitive + 22 composite tasks × 10 episodes | 1000, 2000, 3000 | LeRobot's VLABench image | `results/vlabench/<name>/{primitive,composite}/seed*/` |

Benchmark-specific notes:
- **LIBERO:**
  - Suite horizons are 280, 280, 300 and 520 steps, with 10 settle steps and 180°-rotated
    images. `scripts/eval_libero_baseline.py` drives `lerobot-eval` per suite and seed and
    then aggregates.
  - Our protocol baseline for `smolvla_libero` is ~66 % on Spatial with the 50-step horizon;
    the published figure is ~90 %, and the gap is explained in `docs/baselines.md` §3. The
    full 4-suite run is on hold until that is settled.
  - Subsets for quick checks, never for reported numbers:
    `--suites libero_spatial --seeds 1000 -- --eval.n_episodes=10`.
- **LIBERO-Plus:** tests robustness to perturbations (camera, lighting, layout, language,
  noise). Evaluate the same models as LIBERO and report per perturbation family.
- **RoboTwin:**
  - LeRobot's integration hard-codes `demo_clean`, so the Hard setting is not available yet.
  - The SmolVLA reference checkpoint was trained on all episodes, not on the 50-clean protocol.
    Compare against our own clean-50 baseline (`train sft --benchmark robotwin --policy smolvla`).
- **VLABench:** primitive and composite suites are reported separately and together.

Evaluation cost: the full LIBERO protocol is 6,000 episodes, RoboTwin 5,000, and VLABench
1,290. None has been timed end to end yet; time one suite × one seed before planning a full
run.

## 4. Statistics and reporting

`scripts/eval_libero_baseline.py aggregate` (run automatically) writes `summary.json` per model:

- mean ± std over seeds;
- a **95 % cluster-bootstrap CI** that resamples (task, init state) clusters with all their
  seeds. Episodes that share an init state are correlated, so a pooled Wilson interval is too
  narrow; that interval is also reported, for comparison with `lerobot-eval`.

When comparing two models (e.g. SFT vs RL):
- **Paired comparison:** use the paired difference on the same (task, init state, seed)
  episodes, with a bootstrap CI over clusters, rather than two independent CIs.
- **Per-task table:** report per-task success next to the suite mean. RL gains are usually
  concentrated in a few tasks.
- **Sample efficiency (RL):** plot `rl/val_success_rate` against environment steps
  (iterations × group size × episode length) from W&B or `metrics.csv`. Validation layouts are
  not evaluation layouts, so the plot shows progress, and the final protocol number stays the
  result.

Results table template (one per benchmark):

| Model | Hub id | Spatial | Object | Goal | Long | Avg | 95 % CI |
|---|---|---|---|---|---|---|---|
| SmolVLA (ours, same data) | `Shirish24/surgical-vla-libero-smolvla-baseline-s1000` | | | | | | |
| SurgicalVLA SFT | `Shirish24/surgical-vla-libero-sft-...` | | | | | | |
| + RL (sequential) | `...-rl-sequential-...` | | | | | | |
| + RL (joint) | `...-rl-joint-...` | | | | | | |

## 5. Per-episode records, paired comparisons, probes

- **Per-episode records.** `episodes.jsonl` (written by the aggregation) has one row per
  episode: suite, seed, task, **init state** (episode `i` of a task runs init state `i`),
  success, length (`episode_lengths.json`, recorded by `scripts/lerobot_eval.py`) and outcome.
- **Paired comparison.** `pixi run -e dev compare <A> <B>` matches episodes on (suite, seed,
  task, init state) and reports the difference B − A with a cluster-bootstrap CI and
  McNemar's exact p. Use this for every claim that a component helps.
- **Object-token usage.** Evaluate the same checkpoint with
  `-- --policy.object_token_ablation=zero` and `=shuffle`; the drop in success measures how much
  the policy relies on the object tokens.
- **Live perception.** Policies with `object_input` compute their object tokens from the
  current camera frame inside `lerobot-eval` (`attach_live_perception`). This is the same code
  as the training cache, and it adds 236 ms per action chunk.
- **Videos.** LeRobot renders the first episodes of each task to `<out>/<suite>/seed*/videos/`.
  `eval_info.json` lists each video with its episode's success.

## 6. Offline checks (no simulator)

These are useful during development, but they are not benchmark results:

- **SFT validation** (`val/*`, `val_ema/*` in W&B): action MAE on held-out demonstrations,
  split by horizon (`mae_h0_10`, `mae_h10_25`, `mae_h25_50`), gripper MAE, and jitter (predicted
  vs demonstrated chunk smoothness). Falling MAE with flat success usually points to a
  closed-loop problem: horizon, gripper or normalisation.
- **RL validation** (`rl/val_success/<suite>`): fixed seeds on fresh layouts, for monitoring
  only.
- **Pseudo-label quality** against simulator ground truth: `docs/pseudolabel_quality.md`.

## 7. Checklist before quoting a number

- [ ] The last checkpoint of a budget fixed in advance, not chosen by success.
- [ ] Protocol settings: canonical image, all init states, 3 seeds, `n_action_steps` 50
      (check the checkpoint's `config.json`).
- [ ] Same data and the same budget as the baseline it is compared with.
- [ ] `summary.json` with CIs; all seeds reported.
- [ ] RL models: forgetting reported on the suites not trained by RL.
- [ ] Component claims: paired comparison on identical starting states, with CI and p.
- [ ] Deviations (RoboTwin Easy only, multi-task policy) stated.
