# Shared Brain on churn: findings

Run `20261017-223035-churn`, Oct 17 2026. Four setups ran in parallel on the same data, models and budget: 5 generations, 4 researchers per generation, 28 sandbox runs per generation. They shared one baseline: logistic regression, val log loss 0.3947, test log loss 0.4033, noise threshold 0.0020. Full charts and per-generation detail are in `report.html`.

## Result

Test gain is the drop in test log loss of the final best `train.py` versus the starting one, averaged over seeds 1, 2 and 3. Agents never saw the test split.

| Setup | Test gain | Best val log loss | Test log loss (seeds 1, 2, 3) | New bests | Runs | LLM cost |
|---|---|---|---|---|---|---|
| One-shot | 4.1% | 0.3753 | 0.3849, 0.3875, 0.3879 | 1 | 1 | <$0.01 |
| No memory | 20.0% | 0.3118 | 0.3224, 0.3231, 0.3229 | 1 (gen 1) | 140 | $0.43 |
| Raw log | 20.0% | 0.3118 | 0.3224, 0.3231, 0.3229 | 1 (gen 1) | 131 | $0.59 |
| **Shared Brain** | **21.0%** | **0.3075** | **0.3194, 0.3191, 0.3180** | 2 (gens 1, 4) | 134 | $0.62 |

## What it shows

1. **Shared Brain was the only setup that kept improving.** Every setup switched to boosted trees in generation 1 (about 20%). After that, No memory and Raw log never beat their generation-1 model. In generation 4, a Shared Brain researcher treated surname, country and gender as categories for CatBoost. The curator re-checked it on 3 seeds (0.3075, spread 0.0003) and it became the new best. On the test split it is better on all three seeds, by 0.003 to 0.005 log loss.
2. **The notes pointed researchers at the untried direction.** By generation 4, memory listed what had already tied or failed: ensembles, CatBoost tuning, early stopping, deeper trees, an MLP and the first feature set. The one direction left was how features are encoded, and that is where the gain came from. This fits the idea that memory steers search, but one run can't prove the cause.
3. **Curation mattered, not just having notes.** Raw log gave researchers every report verbatim and ended exactly where No memory did (20.0%). Its log grew to 6,879 tokens and it used 2.3× No memory's prompt tokens. Shared Brain's memory peaked at 685 tokens and ended at 592.
4. **The curator kept memory honest.** It re-checked the top claim every generation and rejected three ties that looked like gains on one seed: generations 2 and 3 (model averaging, 0.3120 vs 0.3126) and generation 5 (seed averaging, 0.30753 vs 0.30752). It moved early stopping from "try next" to "failed" once tested. It corrected an earlier "drop surname" note when surname turned out to matter. It removed entries tested two versions back.
5. **One edit is not enough.** The single-agent, single-edit floor got 4.1%. Every multi-agent setup got about 5× that.

## How the memory evolved

| After gen | Tokens | Changes |
|---|---|---|
| 1 | 306 | Best v1: CatBoost. Works: boosted trees beat LR, RF and a small NN. Failed: class weights, LR tuning. Try next: early stopping, CatBoost tuning and ensembles. |
| 2 | 457 | Ensembles and feature set A tie (below noise). Early stopping moved to failed. Depth 10 and MLP failed. |
| 3 | 551 | Re-check: a 5-model average ties v1. Single-model CatBoost tweaks also tie. |
| 4 | 685 | **New best v2**: surname, country, gender as categories (val 0.3075). "Drop surname" note corrected. v0 entries flagged stale. |
| 5 | 592 | Seed averaging ties v2 (failed). Removed 3 stale entries. CatBoost confirmed ahead of LGBM/XGB on the v2 features. |

## Caveats

- **One research run per setup.** The 1-point gap is consistent across the three test seeds, but a repeat of the whole run is needed before calling it reliable. Which ideas researchers try varies from run to run. The One-shot result was 20% in the smoke run and 4% here.
- **Churn has little headroom** once boosted trees are in, so the room for memory to help is small. Superconductivity (about 47% headroom) is the stronger test.
- **Different best rules by design.** Shared Brain's best is a 3-seed mean checked by the curator. No memory and Raw log take the lowest single seed-0 run.
- **Rate limits.** Crusoe refused 421 researcher requests with 429s, partly because another session was using the same key. They were retried, so budgets and results are unaffected, but they made the run take 52 minutes.
- **Cost** counts Crusoe tokens only (Shared Brain includes the DeepSeek V4 Pro curator), not Modal.
