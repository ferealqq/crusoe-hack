# Shared Brain on superconductivity: partial results

Run `20261017-234325-653156-superconductivity`, Oct 17 2026. **Stopped on request after generation 1 of 5**, so this shows only the first generation and isn't a comparison of memory setups yet. In generation 1 no setup has any memory to read.

Same setup as churn: 4 setups in parallel, 4 researchers per generation, 28 sandbox runs per generation, one shared baseline. Target: critical temperature in kelvin. Loss: RMSE, lower is better. Starting model: standardized Ridge regression, val RMSE 17.77 K, test RMSE 17.82 K. Noise threshold 0.089 K.

| Setup | Best val RMSE (K) | Test gain | Generations done |
|---|---|---|---|
| Starting model | 17.77 | 0% | — |
| One-shot | 9.11 | 44.1% | 1 of 1 |
| No memory | 8.67 | 45.7% | 1 of 5 |
| Raw log | 8.73 | 46.0% | 1 of 5 |
| Shared Brain | 8.62 (3-seed mean) | 46.2% | 1 of 5 |

- As on churn, generation 1 is mostly the switch from a linear model to boosted trees, which nearly halves the error.
- The three multi-agent setups are within half a point of each other after generation 1. Generations 2–5 would show whether memory helps.
- No Crusoe rate limits this time (0 refused requests), so each generation took about 5 minutes.

To finish it, re-run `uv run python -m app.orchestrator --dataset superconductivity --setup all`. The orchestrator always starts a fresh run.
