# Results analysis: predicted vs observed effect of each component

Predictions were written **before any arm was trained** (2026-11-24), so they can be checked
against the observations without hindsight. "Observed" is filled in from `results/results.json`
(`pixi run -e dev figures`) and the paired comparisons (`scripts/compare_arms.py`); every
observation cites its CI and the arm it is compared with. Negative and null results stay in.

Protocol: `configs/ablations/arms.yaml` (one suite, its demonstrations, 8k steps, batch
16 × 2 accumulation = 32, 3 training seeds × 3 evaluation seeds × 10 tasks × 50 starting states).

## Mechanistic constraints that shape the predictions

1. **The VLM is frozen** (`train_expert_only`, `freeze_vision_encoder`): 442 M of 792 M
   parameters train, all in the action expert, the heads, the object projector, the state
   projection and the latent queries. Losses read from the pooled VLM prefix could only reach
   those prefix inputs (about 10 M parameters), never the visual features. The auxiliary heads
   therefore read the action expert (`aux_source: expert`).
2. **Object tokens come from single frames**, with no memory. LIBERO-Spatial scenes hold two
   identical black bowls, so the policy receives up to 3 candidates and must use the
   instruction's spatial relation to pick one.
3. **The protocol baseline is 66 %** on Spatial (50-step open-loop horizon), not 90 %. There
   is room above the baseline, and the main failure mode of open-loop execution (drift within
   a chunk) is not what object tokens or reasoning address directly.
4. **Explicit CoT traces were negative before training**: every candidate trace raised
   `smolvla_libero`'s flow-matching loss (−0.01 to −0.35 gain, 19 segments; with the v3 cache,
   −0.0004 to −0.09 on 2 episodes). The traces are bootstrapped: round 1 is unfiltered.
5. **The explicit CoT head starts from scratch in effect**: it reads layer 16 of a 32-layer
   VLM through a copy of the final LM head, and its initial loss is about 22 nats (above
   uniform over the vocabulary, about 10.8).

## Predictions (LIBERO-Spatial ladder)

| Comparison | Factor | Predicted effect | Why | Observed |
|---|---|---|---|---|
| `oracle_vision` − `smolvla` | perfect object inputs + targets | **+5 to +15 pts** | upper bound; if ≤ 0, drop the vision components (gate) | – |
| `aux_only` − `smolvla` | auxiliary targets only | **0 to +3 pts** (likely within CI) | targets shape the expert, but the frozen VLM features cannot change; weights are small (0.02–0.1) | – |
| `objtok_only` − `smolvla` | object tokens as input | **+2 to +8 pts** | explicit target location and depth; ambiguity between identical bowls limits it | – |
| `objtok_only` − `frozen_control` | object structure vs more features | **> 0** if objects matter | same encoders and token count; a null here means "more features", not "objects" | – |
| `vision` − `objtok_only` | aux on top of tokens | **0 to +2 pts** | as `aux_only` | – |
| `cot_explicit` − `vision` | explicit text | **−3 to +2 pts** | bootstrap traces are noisy; the CoT head is weak at start; decoding adds latency | – |
| `cot_latent` − `vision` | future-latent tokens | **0 to +3 pts** | cheap, no decoding; predicts motion of the scene | – |
| `cot_curriculum` − `vision` | explicit → latent | **≥ `cot_latent`** | explicit supervision early, latent-only at inference | – |
| `full` − `cot_curriculum` | up-cycled MoE (3× FFN compute) | **0 to +3 pts** | more capacity; untested routing | – |
| `full_moe_matched` − `full` | compute-matched MoE | **−2 to 0 pts** | same FLOPs as dense; 2× inference latency from the dispatch loop | – |
| `vision_lora` − `vision` | LoRA r16 on the VLM decoder | **+2 to +8 pts** | the released 90 % checkpoint adapted its VLM; LoRA lets every loss shape the prefix | – |
| `vision_lora_vis` − `vision_lora` | + LoRA on SigLIP | **0 to +4 pts** | rendering domain gap (colours) is visual; costs 2× step time | – |
| `full_lora` − `full` | LoRA on the final model | **as `vision_lora` − `vision`** | | – |

Per-suite predictions for `full` − `smolvla`: largest on **Long** (multi-step tasks, where
phases and reasoning matter) and **Spatial** (target identification); smallest on **Object**
(single salient object; baseline already high).

## Observed

Not yet run. This section is filled in per suite as arms finish, with:
- the paired difference and its 95 % cluster-bootstrap CI;
- McNemar's p;
- the per-task breakdown;
- the object-token usage probe (`object_token_ablation` zero / shuffle);
- the CoT action-likelihood gain per phase (`val/cot_gain/*`);
- the per-loss gradient cosines (`grad/cos/*`).
