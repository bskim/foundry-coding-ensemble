# foundry-coding-ensemble

A small proof of concept that turns several Foundry-served coding models into a
selection-based ensemble for SWE-bench, and measures whether that ensemble beats
the best single model on resolved rate and cost.

Inspired by OpenRouter's Fusion (which fuses several models into one
deep-research answer), but adapted to agentic coding: instead of synthesizing a
merged answer every turn, each model runs the full agent loop on its own and a
judge selects the single best patch at the end (Select-on-N). On the test subset
this simplified approach reaches the best-of-N ceiling and is the only
configuration that beats the single-model SOTA baseline. See
[reports/fusion-ensemble-report.md](reports/fusion-ensemble-report.md) for the
full results, including the cost trade-offs and the path to making it cheaper and
stronger.

The single-model companion to this work is the foundry-model-benchmark
repository, which reports SWE-bench Verified and custom coding-eval scores for
each model on its own. This repository adds the harness that aggregates those
single-model candidates into an ensemble.

## Results summary (5-instance Verified subset, agent loop)

| approach                         | resolved | rate | $/resolved |
|----------------------------------|:--------:|:----:|:----------:|
| best single open-weight model    | 3/5      | 60%  | 0.17       |
| gpt-5.4 (single-model SOTA)       | 2/5      | 40%  | 2.12       |
| selection ensemble (Plan A)      | 4/5      | 80%  | 1.79       |
| synthesis ensemble (Plan B)      | 4/5      | 80%  | 1.79       |
| cost-minimal 2-model panel       | 4/5      | 80%  | 0.87       |

Numbers are a small-scale signal, not a benchmark-grade result. Cost is gross
(no cache discount); update [harness/pricing.json](harness/pricing.json) to your
own rates before relying on the dollar figures.

## Layout

```
harness/
  swe_eval.py        Single-shot prediction generator (oracle context).
  agent_eval.py      Agent loop inside the SWE-bench Docker image; git diff -> patch.
  fusion_select.py   Plan A: judge selects the best candidate patch (Select-on-N).
  fusion_synth.py    Plan B: synthesize one diff over candidates, apply-gated.
  oracle_report.py   Best-of-N ceiling and selector efficiency.
  cost_report.py     Resolved-rate + $/resolved ranking from reports + pricing.
  verified_prep.py   Build a Verified subset with oracle context.
  wsl_score.sh       Score predictions with the official Docker harness (WSL).
  configs.example.json  Endpoint list (baseline + open-weight solo panel).
  pricing.json       Per-1M-token list prices used by cost_report.
reports/
  fusion-ensemble-report.md  Full method and results.
```

## How to run

The harness talks to an OpenAI-compatible gateway (for example a local LiteLLM
proxy on port 4000) that exposes each model by slug. Set the gateway key in the
environment:

```
$env:LITELLM_KEY = "<your gateway key>"
```

1. Build the evaluation subset (needs the `datasets` package, run in the swebench venv):

   ```
   python harness/verified_prep.py --n 5 --seed 0 --out verified_subset.jsonl
   ```

2. Generate one candidate per open-weight model with the agent loop:

   ```
   python harness/agent_eval.py --dataset verified_subset.jsonl \
       --configs harness/configs.example.json --config-name solo-minimax-m2.5 \
       --out preds_agent_minimax.jsonl
   ```

   Repeat for each panel model (kimi-k2, deepseek-v4-pro, glm-5.1, ...).

3. Aggregate the candidates.

   Selection (Plan A):

   ```
   python harness/fusion_select.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select.jsonl
   ```

   Synthesis (Plan B), same candidate flags plus `--synth-model`:

   ```
   python harness/fusion_synth.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --synth-model deepseek-v4-pro --out preds_fusion-synth.jsonl
   ```

4. Score with the official Docker harness (inside WSL):

   ```
   bash harness/wsl_score.sh refine /mnt/c/.../preds_fusion-select.jsonl \
       /mnt/c/.../preds_fusion-synth.jsonl
   ```

5. Analyze:

   ```
   python harness/cost_report.py --run-id refine --configs harness/configs.example.json \
       --names fusion-select fusion-synth --pricing harness/pricing.json
   python harness/oracle_report.py --reports report_solo-*.refine.json \
       --selected report_fusion-select.refine.json report_fusion-synth.refine.json
   ```

## Notes

- The harness modules are standard-library Python. Scoring uses the `swebench`
  package and subset preparation uses `datasets` (see harness/requirements.txt).
- Prediction files (`preds_*.jsonl`), harness reports (`report_*.json`), and logs
  are run artifacts and are not committed; regenerate them with the steps above.
- gpt-5.4 is the single-model baseline and is not an ensemble member.
