# foundry-coding-ensemble

Read this in Korean: [README_KO.md](README_KO.md)

A small proof of concept that turns several Foundry-served open-weight coding
models into a selection-based ensemble for SWE-bench, and measures whether that
ensemble beats the best single model on resolved rate and cost.

Inspired by OpenRouter's Fusion (which fuses several models into one
deep-research answer), but adapted to agentic coding: instead of synthesizing a
merged answer every turn, each model runs the full agent loop on its own and the
aggregator selects among the finished candidate patches. Two aggregators are
measured: execution-grounded selection (score every candidate by build,
regression, self-authored tests, and type/lint; an LLM judge breaks ties only)
and cost-aware routing (consult candidates cheapest-first and accept the first
clean one). On the disc10 set - ten SWE-bench Verified instances curated to
expose cross-model variance - execution-grounded selection resolves 10/10, above
the best single open model (glm-5.1, 9/10) and the single-model SOTA baseline
(gpt-5.4, 7/10): the first configuration where the ensemble exceeds the best
solo. See [reports/fusion-ensemble-report.md](reports/fusion-ensemble-report.md)
for the full results, the per-instance solve matrix, and the honest caveats
(curated discriminating set, small scale, judge variance).

Caveat: this is a deliberately simplified agent harness. Perceived quality - of
the solo agents and the ensemble alike - depends heavily on the maturity of a
real coding agent's client-side harness (Claude Code, Codex, Copilot, Aider), so
read the numbers as relative-within-this-harness, not as absolute capability.

Read the headline with its limits in mind. disc10 is a curated, discriminating
slice of SWE-bench Verified: the ten instances were chosen because the
panel disagrees on them, not drawn at random, so it is the set where a selector
has the most room to help and is not a representative sample of SWE-bench, let
alone of real coding work across languages and repositories. Because the panel
was selected for disagreement, its union of correct answers is unusually large,
which is exactly the condition selection exploits - on a random split the oracle
ceiling would sit much closer to the best solo and the headroom would shrink. The
open-weight standings are also easy to over-read: open models are often distilled
from frontier models and trained against public benchmarks, and SWE-bench
Verified is public, so glm-5.1's lead and minimax-m2.5 matching the SOTA baseline
may partly reflect benchmark familiarity rather than transferable skill. This is a
proof of concept; a ranking worth trusting needs a larger uncurated evaluation, a
held-out split to choose the panel without post-hoc fit, and validation on private
tasks the models have not seen.

The single-model companion to this work is the
[foundry-model-benchmark](https://github.com/jisunchoii/foundry-model-benchmark)
repository, which reports SWE-bench Verified and custom coding-eval scores for
each model on its own. This repository adds the harness that aggregates those
single-model candidates into an ensemble. The two repositories share the same
gross-cost method and the same official SWE-bench Verified Docker harness, but
they grade different instance subsets - foundry-model-benchmark scores its own
selection, while the numbers here are measured on the disc10 set - so the
per-model figures are not directly comparable across the two.

## Results summary (disc10 Verified subset, agent loop)

| approach                              | resolved | rate | $/resolved |
|---------------------------------------|:--------:|:----:|:----------:|
| execution-grounded selection (fusion) | 10/10    | 100% | 0.56       |
| best single open model (glm-5.1)      | 9/10     | 90%  | 0.23       |
| cost-aware routing (fusion)           | 9/10     | 90%  | 0.43       |
| gpt-5.4 (single-model SOTA)           | 7/10     | 70%  | 0.43       |
| cheapest strong open (minimax-m2.5)   | 7/10     | 70%  | 0.05       |
| oracle best-of-4 ceiling              | 10/10    | 100% | -          |

On disc10 the open panel does have real cross-model complementarity (glm-5.1
misses matplotlib-25311, which minimax-m2.5 solves), so the oracle best-of-4
ceiling is 10/10. Execution-grounded selection reaches that ceiling, one
resolution above the best solo and three above SOTA; cost-aware routing reaches
9/10 more cheaply. The trade-off is cost per resolved: selection pays for the
full panel plus a judge, so its value here is the extra coverage, not the lowest
price. disc10 is a curated discriminating set (chosen to expose where models
differ), so read 100% as a ceiling-on-this-set signal, not a benchmark-grade
rate. Cost is gross (no cache discount); update
[harness/pricing.json](harness/pricing.json) to your own rates before relying on
the dollar figures.

## Layout

```
harness/
  swe_eval.py        Single-shot prediction generator (oracle context).
  agent_eval.py      Agent loop inside the SWE-bench Docker image; git diff -> patch.
  fusion_select.py   Early aggregator: judge selects the best candidate patch.
  fusion_synth.py    Early aggregator: synthesize one diff over candidates, apply-gated.
  fusion_select_exec.py  Execution-grounded selection: score every candidate by
                         execution signals (applies, compiles, suite imports,
                         self-tests), judge only on ties. No gold tests required.
  fusion_route.py    Cost-aware routing on top of fusion_select_exec: consult
                     candidates cheapest-first, accept the first clean one,
                     escalate only when needed.
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

   To reproduce the exact disc10 set used in the report, pass the ten curated
   instance ids instead of a random sample:

   ```
   python harness/verified_prep.py --out disc10.jsonl --ids \
     "astropy__astropy-14508,django__django-12039,django__django-15161,matplotlib__matplotlib-25311,pydata__xarray-4695,pydata__xarray-7393,pytest-dev__pytest-7236,sphinx-doc__sphinx-7889,sympy__sympy-12419,sympy__sympy-15875"
   ```

2. Generate one candidate per open-weight model with the agent loop:

   ```
   python harness/agent_eval.py --dataset verified_subset.jsonl \
       --configs harness/configs.example.json --config-name solo-minimax-m2.5 \
       --out preds_agent_minimax.jsonl
   ```

   Repeat for each panel model (kimi-k2, deepseek-v4-pro, glm-5.1, ...).

3. Aggregate the candidates.

   Selection (LLM judge), an early aggregator:

   ```
   python harness/fusion_select.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select.jsonl
   ```

   Synthesis (end-of-run merge), an early aggregator, same candidate flags plus `--synth-model`:

   ```
   python harness/fusion_synth.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --synth-model deepseek-v4-pro --out preds_fusion-synth.jsonl
   ```

   Execution-grounded selection (no gold tests; needs WSL + Docker, same as
   agent_eval):

   ```
   python harness/fusion_select_exec.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select-exec.jsonl
   ```

   Cost-aware routing (cheapest first, escalate on demand):

   ```
   python harness/fusion_route.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --order minimax-m2.5,deepseek-v4-pro \
       --judge-model deepseek-v4-pro --out preds_fusion-route.jsonl
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

## General-purpose direction

SWE-bench grades with hidden gold tests; real coding has none. The selector here
never reads those tests (it picks among candidate diffs from the issue text
alone), so the mechanism is general. Execution-grounded selection
(`harness/fusion_select_exec.py`) and cost-aware routing
(`harness/fusion_route.py`) are measured on this subset: execution-grounded
selection reaches the best-of-4 oracle ceiling (10/10), beating the best single
open model (glm-5.1, 9/10) and the proprietary SOTA baseline (gpt-5.4, 7/10),
while cost-aware routing reaches 9/10 at lower cost (see the run steps above).
They replace the
gold-test oracle with runtime verification signals (build, regression vs existing
tests, self-authored tests, type/lint), so no gold tests are required. disc10 is
a curated discriminating set built for cross-model variance, so the result shows
selection can reach the oracle ceiling when the panel disagrees - not that it does
so on a random or full split. What
remains is scaling the evaluation on an uncurated sample, then testing a mixed
open/proprietary panel. In real
use grading is not run every task; qualitative human review stands in, and
SWE-bench is kept only as a method-validation harness. See section 6 of
[reports/fusion-ensemble-report.md](reports/fusion-ensemble-report.md) for
details.
