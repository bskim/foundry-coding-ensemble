# Selection-based ensemble vs solo on SWE-bench Verified

Read this in Korean: [fusion-ensemble-report_ko.md](fusion-ensemble-report_ko.md)

This is a proof of concept. It was inspired by OpenRouter's Fusion (which fuses
several models for deep-research answers), but adapted to agentic coding by
replacing answer synthesis with the simpler idea of selection over N candidates
(Select-on-N). Even with that simplification, the ensemble reaches the
best-of-N ceiling on the test subset and is the only configuration here that
beats the single-model SOTA baseline. The numbers below are a signal at small
scale, not a benchmark-grade result, and there is clear room to advance on both
cost and quality (see "Room to improve").

Subset (5 instances): matplotlib__matplotlib-24970, mwaskom__seaborn-3187,
pallets__flask-5014, psf__requests-1921, pydata__xarray-7393.

Cost basis: gross (every prompt token at the input rate, every completion token
at the output rate, no cache discount). The same method is used for every
config, so $/resolved is directly comparable. Prices are in
[harness/pricing.json](../harness/pricing.json).

gpt-5.4 is the single-model SOTA baseline and is not an ensemble member. The
open-weight panel is kimi-k2, deepseek-v4-pro, minimax-m2.5, grok-4.3, glm-5.1.

## 1. Single-shot, oracle context

Model handed the gold files, one rewrite, no repo exploration and no test
feedback (swe_eval.py, run_id s2).

| config               | resolved | rate | $/resolved |
|----------------------|:--------:|:----:|:----------:|
| solo-deepseek-v4-pro | 3/5      | 60%  | 0.11       |
| solo-kimi-k2         | 3/5      | 60%  | 0.21       |
| ensemble-cost        | 3/5      | 60%  | 0.47       |
| ensemble-swe-elite   | 3/5      | 60%  | 0.54       |
| gpt-5.4 (SOTA)       | 2/5      | 40%  | 0.53       |
| ensemble-crossjudge  | 2/5      | 40%  | 0.62       |
| ensemble-diverse     | 2/5      | 40%  | 0.65       |
| ensemble-wide        | 2/5      | 40%  | 1.10       |

In single-shot mode the per-turn fusion variants give no quality gain over the
best solo at 2 to 7 times the cost.

## 2. Agent loop, 30 turns, Docker

Real repo at base_commit, tools (ls, read, grep, write, run pytest), git diff is
the patch, scored by the official SWE-bench Docker harness (agent_eval.py,
run_id agent_subset). Runs sequentially, one Foundry job at a time.

| config               | resolved | rate | tokens    | $ total | $/resolved |
|----------------------|:--------:|:----:|:---------:|:-------:|:----------:|
| solo-kimi-k2         | 3/5      | 60%  | 1,270,620 | 1.38    | 0.46       |
| solo-deepseek-v4-pro | 3/5      | 60%  | 1,642,470 | 2.96    | 0.99       |
| gpt-5.4 (SOTA)       | 2/5      | 40%  | 1,385,052 | 4.25    | 2.12       |
| per-turn-fusion-cost | 2/5      | 40%  | 7,080,297 | 8.22    | 4.11       |
| per-turn-fusion-elite| 1/5      | 20%  | 6,163,894 | 8.18    | 8.18       |

Per-turn synthesis (fusing every agent turn) is worse in agent mode: the
synthesizer gets stuck in read loops and gateway errors, burns 4 to 5 times the
tokens, and costs 9 to 18 times the best solo per resolved issue.

### Per-instance resolution (agent loop)

| instance          | kimi | deepseek | gpt-5.4 | fusion-cost | fusion-elite |
|-------------------|:----:|:--------:|:-------:|:-----------:|:------------:|
| matplotlib-24970  | yes  | yes      | yes     | yes         | no           |
| seaborn-3187      | no   | no       | no      | no          | no           |
| flask-5014        | yes  | yes      | yes     | yes         | no           |
| requests-1921     | yes  | no       | no      | no          | yes          |
| xarray-7393       | no   | yes      | no      | no          | no           |
| resolved          | 3/5  | 3/5      | 2/5     | 2/5         | 1/5          |

### Oracle best-of-N ceiling

If a perfect selector always picked a resolving candidate among the open-weight
solos, the resolved set is the union of the candidates:

- best-of-2 of kimi and deepseek covers matplotlib, flask, requests, xarray = 4/5 (80%).
- The ceiling on this subset is 4/5. seaborn-3187 is unsolved by every model and config.

That 4/5 ceiling (against gpt-5.4 at 2/5, best solo at 3/5, per-turn fusion at 1
to 2/5) is the motivation for the refined approach: keep the diversity, drop the
per-turn synthesis.

## 3. Why per-turn fusion failed, and what changed

- OpenRouter Fusion ([blog](https://openrouter.ai/blog/announcements/fusion-beats-frontier/))
  is designed for deep research (text answers, no long-horizon tasks), not
  agentic coding. Its own FAQ notes Fusion is not a drop-in replacement for
  coding models. It fits a selectively invoked tool for hard sub-questions, not
  the base coding loop.
- About three quarters of Fusion's reported lift comes from the synthesis step
  and about one quarter from model diversity.
- Mixture-of-Agents ([arXiv 2406.04692](https://arxiv.org/abs/2406.04692)) uses
  layered proposers and an aggregator and is strong on chat and instruction
  benchmarks, but not evaluated on agentic coding.
- Code is verifiable, so the coding analog of synthesis is selection-based
  best-of-N: generate diverse candidates and pick one with a judge (or a
  test-grounded selector). There is no full-file LLM merge, which was the exact
  failure mode behind the read loops and gateway errors above.

### Refined design

1. Panel: K diverse open-weight solo agents (reusing agent_eval.py), each
   produces one candidate patch through the normal agent loop. No per-turn fan-out.
2. Selector: one open-weight judge picks the single best candidate from the issue
   and all candidate diffs. This is selection, not synthesis.
3. Score the selected patch with the same Docker harness, giving a comparable
   resolved rate and $/resolved.

Token cost is roughly K solo runs plus one small judge call, far below the per-turn
fan-out of 6 to 7 million tokens.

## 4. Refined results: Plan A (select) and Plan B (synth)

Agent loop, 30 turns, Docker harness, run_id refine. Open-weight panel of 4 solo
agents (kimi-k2, deepseek-v4-pro, minimax-m2.5, glm-5.1), each producing one
candidate patch. Two aggregators run over the same candidate set:

- Plan A, fusion-select: an open-weight judge (deepseek-v4-pro) picks the single
  best candidate patch per instance. Selection, no merge.
- Plan B, fusion-synth: deepseek-v4-pro synthesizes one unified diff over all
  candidate patches, once, at the end. An execution apply-gate requires the
  synthesized diff to pass `git apply --check` in the instance container,
  otherwise it falls back to the fusion-select patch.

### 4.1 Candidate solos (open-weight panel)

| candidate            | resolved | rate | tokens    | $ total | $/resolved |
|----------------------|:--------:|:----:|:---------:|:-------:|:----------:|
| solo-minimax-m2.5    | 3/5      | 60%  | 1,295,531 | 0.51    | 0.17       |
| solo-glm-5.1         | 3/5      | 60%  | 1,311,252 | 2.31    | 0.77       |
| solo-kimi-k2         | 3/5      | 60%  | 1,270,620 | 1.38    | 0.46       |
| solo-deepseek-v4-pro | 3/5      | 60%  | 1,642,470 | 2.96    | 0.99       |

### 4.2 Aggregators vs the oracle ceiling

| config            | resolved | rate | tokens    | $ total | $/resolved | oracle eff. |
|-------------------|:--------:|:----:|:---------:|:-------:|:----------:|:-----------:|
| fusion-select     | 4/5      | 80%  | 5,531,445 | 7.18    | 1.79       | 4/4 (100%)  |
| fusion-synth      | 4/5      | 80%  | 5,532,167 | 7.18    | 1.79       | 4/4 (100%)  |
| oracle best-of-4  | 4/5      | 80%  | -         | -       | -          | -           |
| gpt-5.4 (SOTA)    | 2/5      | 40%  | 1,385,052 | 4.25    | 2.12       | -           |
| per-turn fusion   | 2/5      | 40%  | 7,080,297 | 8.22    | 4.11       | 2/4 (50%)   |

### 4.3 Per-instance resolution (refine run)

| instance         | minimax | glm | kimi | deepseek | fusion-select | fusion-synth |
|------------------|:-------:|:---:|:----:|:--------:|:-------------:|:------------:|
| matplotlib-24970 | yes     | yes | yes  | yes      | yes           | yes          |
| seaborn-3187     | no      | no  | no   | no       | no            | no           |
| flask-5014       | yes     | yes | yes  | yes      | yes           | yes          |
| requests-1921    | yes     | yes | yes  | no       | yes           | yes          |
| xarray-7393      | no      | no  | no   | yes      | yes           | yes          |
| resolved         | 3/5     | 3/5 | 3/5  | 3/5      | 4/5           | 4/5          |

### 4.4 Findings

1. Quality reaches the ceiling. Both aggregators reach 4/5 (80%), the oracle
   best-of-N ceiling, at 100% selector efficiency. This beats every other config
   here: gpt-5.4 at 2/5, best solo at 3/5, per-turn fusion at 1 to 2/5. No single
   open-weight model exceeds 3/5, but the union of a diverse panel reaches 4/5,
   and both a judge (select) and a synthesizer (synth) recover that full union.
2. Tokens are lower but not free. 5.53M tokens against the per-turn 6 to 7M,
   because the cost is now K real agent candidates (4 times about 1.3M) rather
   than a per-turn fan-out. The aggregation call itself is negligible: synth adds
   only about 722 tokens (about 0.003 dollars) over select.
3. $/resolved is still above the best solo. 1.79 per resolved is about 10 times
   solo-minimax's 0.17, because you pay to generate 4 candidates to buy the extra
   resolution (xarray, which only deepseek solves). The refined ensemble trades
   cost for a higher quality ceiling, the opposite of per-turn fusion, which cost
   more and lost quality.
4. Selection is about equal to synthesis on atomic fixes. Both reach 4/5 on the
   same instances. On SWE-bench-style fixes each resolvable issue is solved whole
   by at least one candidate, so there is nothing to combine: synthesis cannot
   exceed selection, and the apply-gate only adds risk (a malformed merged diff)
   for no gain. The reported synthesis lift was measured on deep-research answer
   fusion, where partial answers genuinely combine, and it does not transfer to
   atomic code patches. Prefer fusion-select for its simplicity; keep
   fusion-synth only for tasks where the correct fix spans multiple candidates.

### 4.5 Cost-minimal panel

xarray-7393 is resolved only by deepseek-v4-pro, and the other three resolvable
instances are covered by minimax-m2.5. So the 2-model panel of
{minimax-m2.5, deepseek-v4-pro} already spans the full oracle 4/5:

| panel                                | resolved | $ total (approx) | $/resolved (approx) |
|--------------------------------------|:--------:|:----------------:|:-------------------:|
| 4-model (kimi, deepseek, minimax, glm) | 4/5    | 7.18             | 1.79                |
| 2-model (minimax, deepseek)            | 4/5    | 3.47             | 0.87                |
| solo-minimax-m2.5 (single model cap)   | 3/5    | 0.51             | 0.17                |

Pruning the panel to the two complementary models that cover the ceiling halves
the cost while keeping 4/5. The remaining gap to the single-model price (0.17) is
the intrinsic price of the extra resolution, not overhead.

## 5. Room to improve

This is a small proof of concept, and the headline result (Select-on-N reaches
the best-of-N ceiling) holds on 5 instances. The path to making it both cheaper
and stronger is concrete:

- Scale the evaluation. 5 instances is a signal, not a benchmark. Re-run on a
  larger Verified subset (50 to 500) to confirm the ceiling effect and to get
  stable cost numbers.
- Prune the panel. The cost-minimal panel already halves the cost at the same
  quality. A learned or measured complementarity score can pick the smallest
  panel that spans the ceiling per repo or per issue type.
- Route instead of always running N. Start with the cheapest solo and escalate to
  the panel only when candidates disagree or tests still fail, so the full panel
  cost is paid only on the hard instances.
- Make selection test-grounded. The judge is an LLM today. Running the repo's own
  tests against each candidate before selecting would turn selection into a
  verifiable signal and reduce reliance on the judge's reasoning.
- Raise the ceiling itself. The ceiling here is bounded by the panel: seaborn-3187
  is unsolved by every member. Adding a model that covers currently unsolved
  instances lifts the oracle ceiling that selection can then capture.
- Re-evaluate against the newest SOTA baseline. The advantage over gpt-5.4 should
  be re-measured whenever a stronger baseline becomes available.

The takeaway: even a simplified, selection-only ensemble already beats the
single-model SOTA on this subset, and the cost can be brought down substantially
by pruning and routing. With more scale and a test-grounded selector, there is
meaningful headroom on both axes.

## 6. Toward a general-purpose coding selector

SWE-bench has two things real coding work does not: a fixed problem set and hidden
gold tests that grade each patch. That raises a fair question: is this a general
approach, or is it only tuned to a benchmark whose answers are known?

What was and was not fit to the benchmark:

- The selector never saw the gold tests. fusion-select picks among candidate diffs
  from the issue text alone, the same information a real coding agent has. Grading
  by gold tests happens offline, after selection. So the selection mechanism
  itself is general.
- One result is fit to the benchmark and should be read with care: the
  cost-minimal 2-model panel in 4.5 was chosen after seeing per-instance results.
  That is a post-hoc, oracle-fit configuration and is not evidence of general
  performance.

How evaluation works outside SWE-bench. In real use you do not grade every task
with gold tests; qualitative human review stands in for grading. SWE-bench is used
here only as a method-validation harness: because it grades objectively, it can
answer the one question we need, which is whether an implementation built this way
scores higher than a single model. If execution-grounded selection raises the
SWE-bench score, the mechanism is doing real work, not memorizing answers.

### Plan 1: execution-grounded selection (runtime, no gold tests)

Replace the LLM-only judge with verifiable signals the agent actually has at
runtime, scored and combined:

1. Build / compile passes (the cheapest hard filter; a candidate that does not
   build is dropped).
2. Regression check against the repo's existing test suite (a candidate that
   breaks currently-passing tests is penalized).
3. Self-authored tests: each panel agent writes the fix plus a test for it, and
   those tests are cross-applied to the other candidates as a runtime oracle.
4. Type check and lint (mypy / tsc / ruff and equivalents).
5. Candidate agreement (independent agents converging on a similar diff add
   confidence).
6. LLM judge as the tiebreaker only when the execution signals are tied.

selection score = f(build, regression, self-test pass rate, type/lint, agreement,
judge). This turns selection from a guess into a verifiable decision, which is
what lets it work outside SWE-bench.

### Plan 2: a general evaluation methodology

- Held-out test split: split a repo's tests into a set visible to the agent and
  selector and a hidden set used only for grading. This mirrors real development
  and keeps the selector away from the answer.
- Scale and diversity: 50 to 500 instances, more than one language and task type
  (not only Python bug-fixes), to check the panel and selector are robust to the
  task distribution.
- Train/test panel split: decide which models go in the panel on a train split,
  and report numbers only on a test split. This removes the post-hoc fit behind
  the 2-model panel.
- Test-less tasks (docs, UI, refactors): calibrate an LLM rubric judge against
  human ratings, or collect proxy signals (CI pass rate, review acceptance,
  post-merge revert rate) as longer-term evidence.

### Plan 3: cost-aware routing (built on Plan 1)

Running N agents on every task is impractical in real use.

- Difficulty routing: try the cheapest solo first; finish if its build and tests
  pass.
- Escalate on disagreement: call the panel only when the first attempt fails or
  candidates disagree.
- Early stop: if one candidate passes all available tests (Plan 1 signals), skip
  evaluating the rest.

### Measurement protocol

Because Plan 3 is built on Plan 1, the two are measured in order:

1. Measure Plan 1 on SWE-bench: does execution-grounded selection hold the
   coverage the LLM-judge version reached?
2. Measure Plan 1 + 3: does routing keep the same coverage while cutting
   $/resolved?

| step | change | success criterion |
|------|--------|-------------------|
| 1 | selector: LLM-only to execution-grounded | selection without gold tests holds the oracle selection efficiency |
| 2 | held-out test split, scale to 50-500 | selection efficiency does not collapse from the small-subset 100% |
| 3 | panel chosen on a train/test split | coverage advantage over the best single model holds without post-hoc fit |
| 4 | multi-language, multi-task types | coverage gain reproduces off-distribution |
| 5 | add difficulty routing (Plan 1 + 3) | same coverage at substantially lower $/resolved |

Measured run on this subset (4-model panel: minimax-m2.5, deepseek-v4-pro,
kimi-k2, glm-5.1; gold-test scoring, judge only on ties):

| config | resolved | $/resolved | tokens |
|--------|----------|------------|--------|
| Plan 1 (execution-grounded selection) | 4/5 (80%) | 1.79 | 5.53M |
| Plan 1 + 3 (cost-aware routing) | 4/5 (80%) | 0.29 | 1.66M |

Routing held the same 80% coverage (the same four instances resolved, seaborn-3187
missed by both) while cutting tokens by about 70% and $/resolved by about 84%: it
consulted the cheapest candidate first and stopped as soon as that candidate's
verifiable signals were clean, escalating to a second candidate on only one
instance. Execution-grounded Plan 1 reproduced the LLM-judge selector's 4/5 from
section 4, so swapping the gold-test-free signals in did not cost coverage here.
On five instances this is a small signal, not a benchmark-grade result.

This is still a proof of concept. The claim is narrow: the selection mechanism is
general because it never reads the answer, and the way to prove it on real coding
work is to replace the gold-test oracle with runtime verification signals and
re-validate with a held-out, train/test-split methodology.

## Companion benchmark

For single-model SWE-bench results on these same Foundry-served models, see the
foundry-model-benchmark repository (reports for SWE-bench Verified 500 and a
custom coding eval). This repository is the ensemble companion: it shares the
harness that turns those single-model candidates into a selection-based ensemble.
