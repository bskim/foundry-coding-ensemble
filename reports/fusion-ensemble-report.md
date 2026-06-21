# Selection-based ensemble vs solo on SWE-bench Verified

Read this in Korean: [fusion-ensemble-report_ko.md](fusion-ensemble-report_ko.md)

This is a proof of concept. A diverse panel of open-weight models each produces one
candidate patch through an agent loop, and an aggregator picks among the finished
candidates. Two aggregators are implemented and measured here:

- Execution-grounded selection: score every candidate by build, regression,
  self-authored tests, and type/lint, and pick the best; an LLM judge breaks ties only.
  Implemented in fusion_select_exec.py.
- Cost-aware routing: the same execution signals plus difficulty routing that consults
  candidates cheapest-first and stops as soon as one passes, so the full panel is paid for
  only on hard instances. Implemented in fusion_route.py.

Headline on this subset (honest, not inflated). On disc10 the open panel has real
cross-model complementarity: glm-5.1 resolves 9/10 but misses matplotlib-25311, which
minimax-m2.5 solves, so the oracle best-of-4 ceiling is 10/10. Execution-grounded
selection reaches that ceiling - 10/10 - one resolution above the best single open model
(glm-5.1, 9/10) and three above the single-model SOTA baseline (gpt-5.4, 7/10). On this set,
then, the approach shows it can widen coverage beyond the best solo - a possibility
confirmed here, not a general performance gain. Cost-aware
routing reaches 9/10 more cheaply. The trade-off is price per resolved: selection runs the
full panel plus a judge, so its value is the extra coverage, not the lowest cost - the
cheapest strong open model (minimax-m2.5) reaches 7/10 at $0.05 per resolved. The numbers
below are a signal at small scale on a curated discriminating set, not a benchmark-grade
result.

Caveat on the subset and harness. disc10 is curated to expose cross-model variance - the
instances were chosen because models disagree on them - so it is the set where selection
has the most room to help, and the 10/10 should be read as a ceiling-on-this-set signal,
not a general resolved rate. The harness is also deliberately simplified (a fixed tool set,
a single context-trimming rule, no repo map or retrieval, no test-driven iteration loop
beyond what the model asks for). Perceived quality - both of the solo agents and of the
ensemble on top of them - depends heavily on the maturity of a real coding agent's
client-side harness (Claude Code, Codex, Copilot, Aider). A stronger harness would move
every number here, so read the comparison as relative-within-this-harness, not as an
absolute capability measurement.

How the subset is built. disc10 is a ten-instance slice of SWE-bench Verified, assembled
from the public Verified split as a curated id list. The instances were not drawn at random: they were
picked for discrimination, so the panel spans the full range from all-solve to
only-one-solves instead of clustering at easy or hard. That makes the set useful for
separating models, but it also means it is not a representative sample of SWE-bench, let
alone of real coding work across languages, repositories, and task types. Read every number
here as a directional signal on one hand-picked set, not a capability score.

Why selection looks good here. Because the instances were chosen for cross-model
disagreement, the panel's union of correct answers is unusually high relative to any single
model - which is exactly the condition a selection ensemble exploits. On a random or
representative split the panel would agree far more often, the oracle ceiling would sit much
closer to the best solo, and the headroom for selection would shrink accordingly. The 10/10
here is therefore a best case for fusion, not a typical one; the honest claim is that
selection can reach the oracle ceiling when the panel genuinely disagrees, not that
selection adds a resolution on average.

On the open-weight scores. glm-5.1 leading the panel and minimax-m2.5 matching the
proprietary SOTA baseline is striking, but it should be read with care. Open-weight models
are frequently distilled from frontier models and trained against public benchmarks, and
SWE-bench Verified is public, so part of an open model's apparent lead can be benchmark
familiarity rather than transferable problem-solving. Nothing here rules contamination in or
out; it only means the ranking has to be confirmed on private, real-world tasks the models
have not seen before it can be trusted.

Scope. This is a proof of concept, not a product. The point was to test whether selection
over open-weight candidates can beat the best solo at all, and on this set it can. Turning
that into a general claim needs a larger uncurated evaluation, a held-out split used to
choose the panel without post-hoc fit, and validation on tasks outside any public benchmark
- the conditions under which a real coding system would actually run.

Subset (disc10: ten Verified instances curated to discriminate models):
astropy__astropy-14508, django__django-12039, django__django-15161,
matplotlib__matplotlib-25311, pydata__xarray-4695, pydata__xarray-7393,
pytest-dev__pytest-7236, sphinx-doc__sphinx-7889, sympy__sympy-12419, sympy__sympy-15875.
The set was built from the public SWE-bench Verified split by selecting instances with high
cross-model variance (not all-easy, not all-hard), so it deliberately surfaces the cases
where a selector can matter.

Cost basis: gross (every prompt token at the input rate, every completion token at the
output rate, no cache discount). The same method is used for every config, so $/resolved
is directly comparable. Prices are in [harness/pricing.json](../harness/pricing.json). This
matches the cost method of the single-model companion
[foundry-model-benchmark](https://github.com/jisunchoii/foundry-model-benchmark), and both
use the same official SWE-bench Verified Docker harness - but the two grade different
instance subsets (foundry-model-benchmark scores its own selection; the figures here are
disc10), so per-model numbers are not directly comparable across the two repositories.

gpt-5.4 is the single-model SOTA baseline and is not an ensemble member. The open-weight
panel is kimi-k2, deepseek-v4-pro, minimax-m2.5, glm-5.1.

## 1. The panel and the oracle ceiling

Agent loop, 30 turns, real repo at base_commit, tools (ls, read, grep, edit, write, run
pytest), git diff is the patch, scored by the official SWE-bench Docker harness
(agent_eval.py, disc10). Each panel model produces one candidate patch.

| candidate            | resolved | rate | tokens    | $ total | $/resolved |
|----------------------|:--------:|:----:|:---------:|:-------:|:----------:|
| solo-glm-5.1         | 9/10     | 90%  | 1,101,394 | 2.07    | 0.23       |
| solo-minimax-m2.5    | 7/10     | 70%  | 1,105,525 | 0.38    | 0.05       |
| solo-deepseek-v4-pro | 5/10     | 50%  | 985,402   | 1.77    | 0.35       |
| solo-kimi-k2         | 3/10     | 30%  | 1,081,260 | 1.20    | 0.40       |

Per-instance, the panel splits as follows (gpt-5.4 shown for reference; it is the SOTA
baseline, not a panel member):

| instance         | glm | minimax | deepseek | kimi | gpt-5.4 (SOTA) |
|------------------|:---:|:-------:|:--------:|:----:|:--------------:|
| xarray-7393      | yes | yes     | yes      | yes  | yes            |
| sphinx-7889      | yes | yes     | yes      | yes  | yes            |
| xarray-4695      | yes | yes     | yes      | no   | yes            |
| astropy-14508    | yes | yes     | yes      | no   | no             |
| django-12039     | yes | no      | yes      | no   | yes            |
| pytest-7236      | yes | yes     | no       | no   | yes            |
| sympy-15875      | yes | yes     | no       | no   | yes            |
| matplotlib-25311 | no  | yes     | no       | no   | yes            |
| django-15161     | yes | no      | no       | no   | no             |
| sympy-12419      | yes | no      | no       | yes  | no             |
| resolved         | 9/10| 7/10    | 5/10     | 3/10 | 7/10           |

Two facts drive everything below:

1. The open panel has real internal complementarity on this set. glm-5.1 resolves 9/10 but
   misses matplotlib-25311; minimax-m2.5 solves matplotlib-25311. Their union, and the
   union of the full four-model panel, is 10/10. So the oracle best-of-4 ceiling is 10/10,
   one above the best single open model - unlike a random subset, disc10 was curated so
   this complementarity exists, which is the point of the set.
2. SOTA is not the ceiling here. gpt-5.4 resolves 7/10 and misses three instances the open
   panel covers (astropy-14508, django-15161, sympy-12419); two of those (django-15161,
   sympy-12419) are covered only because glm and kimi disagree. A selector over the open
   panel can therefore exceed gpt-5.4, which the result below confirms.

Every instance is solved by at least one model; the spread runs from instances all five
solve (xarray-7393, sphinx-7889) to instances only one panel model solves (matplotlib-25311
by minimax, django-15161 by glm), so the set is genuinely discriminating rather than
all-easy or all-hard.

## 2. How the design arrived at selection

The current selector did not start here. The first attempts copied OpenRouter's Fusion
([blog](https://openrouter.ai/blog/announcements/fusion-beats-frontier/)) literally -
fuse several models into one answer - and that does not transfer to agentic coding:

- Fusing every agent turn (per-turn synthesis) was worse, not better. The synthesizer got
  stuck in read loops and gateway errors, burned 6 to 7M tokens, and resolved fewer
  instances than the best single model. Fusion's own FAQ notes it is not a drop-in
  replacement for coding models; it fits deep-research text answers, not a long-horizon
  coding loop.
- Code is verifiable, so the coding analog of answer-fusion is selection over candidates,
  not a full-file LLM merge. The merge step was the exact source of the read loops and
  gateway errors.

That moved the design to a panel of diverse solo agents plus an aggregator over their
candidate patches. An early aggregator pair was compared on the same panel:

- An LLM-judge selector picked the single best candidate per instance.
- An end-of-run synthesizer merged all candidate diffs once, gated by `git apply --check`.

Both reached the same ceiling, and synthesis never beat selection: on SWE-bench-style fixes
each resolvable issue is solved whole by at least one candidate, so there is nothing to
combine, and a merged diff only adds risk. The reported synthesis lift was measured on
deep-research answers, where partial answers genuinely combine, and it does not carry over
to atomic code patches. So the implemented harness selects rather than merges - and
grounds that selection in execution signals rather than an LLM judge alone, which is the
execution-grounded selection described below.

## 3. Result: selection, routing, and SOTA

Both aggregators run over the same 4-model panel and the same disc10 subset, scored by the
same Docker harness with gold tests. The judge (deepseek-v4-pro) is used only to break
ties.

| config                              | resolved | rate | tokens    | $ total | $/resolved | oracle eff. |
|-------------------------------------|:--------:|:----:|:---------:|:-------:|:----------:|:-----------:|
| execution-grounded selection        | 10/10    | 100% | 4,368,246 | 5.61    | 0.56       | 10/10 (100%)|
| cost-aware routing                  | 9/10     | 90%  | 3,312,882 | 3.90    | 0.43       | 9/10 (90%)  |
| oracle best-of-4                    | 10/10    | 100% | -         | -       | -          | -           |
| best solo (glm-5.1)                 | 9/10     | 90%  | 1,101,394 | 2.07    | 0.23       | -           |
| gpt-5.4 (SOTA)                      | 7/10     | 70%  | 1,127,715 | 3.04    | 0.43       | -           |

Execution-grounded selection picks a resolving candidate on all ten instances, including
the three where the panel only narrowly covers the bug: matplotlib-25311 (only minimax),
django-15161 (only glm), and sympy-12419 (glm or kimi). It reaches the 10/10 oracle ceiling
- above the best solo (glm 9/10) and SOTA (gpt-5.4 7/10). Cost-aware routing reaches 9/10:
it diverges from selection on exactly one instance, sympy-15875, where its cheapest-first
tiebreak settles on a candidate that does not resolve the bug while selection, scoring all
candidates, lands on glm-5.1's correct patch.

Findings:

1. On this set the ensemble covers more than the best solo. On disc10, execution-grounded
   selection resolves 10/10 against glm-5.1's 9/10 and gpt-5.4's 7/10. This is the first
   subset measured here where selection gets ahead of the best single model, and it happens
   because disc10 has the cross-model complementarity (glm misses matplotlib-25311, minimax
   covers it) that a selector can exploit - so read it as a possibility shown on a favorable
   set, not a general gain.
2. The value is coverage, not cost. Selection reaches 10/10 at $0.56 per resolved, more
   than the cheapest strong solo (minimax at 7/10 / $0.05) per resolved. The ensemble buys
   the extra resolutions by paying for the full panel and a judge; if your objective is
   lowest cost per resolved rather than maximum coverage, a single cheap-strong model is
   still the better pick.
3. Routing trades one resolution for cost. Cost-aware routing holds 9/10 at $0.43 per
   resolved by consulting candidates cheapest-first and stopping early, so it pays the full
   panel only on hard instances. It loses sympy-15875 because cheapest-first does not score
   every candidate, so it cannot always recover the one correct patch that full selection
   finds.
4. SOTA is not the ceiling on a discriminating set. gpt-5.4 (7/10) is matched by minimax
   alone (7/10, far cheaper) and beaten by glm (9/10) and by both fusion configs. When the
   open panel genuinely disagrees, aggregating it is worth more than reaching for a single
   proprietary model.

## 4. Execution-grounded selection

Selection uses verifiable signals the agent actually has at runtime, scored and combined,
so it works on real repos where no hidden gold tests exist:

1. Build / compile passes (the cheapest hard filter; a candidate that does not build is
   dropped).
2. Regression check against the repo's existing test suite (a candidate that breaks
   currently-passing tests is penalized).
3. Self-authored tests: each panel agent writes the fix plus a test for it, and those
   tests are cross-applied to the other candidates as a runtime oracle.
4. Type check and lint (mypy / tsc / ruff and equivalents).
5. Candidate agreement (independent agents converging on a similar diff add confidence).
6. LLM judge as the tiebreaker only when the execution signals are tied.

selection score = f(build, regression, self-test pass rate, type/lint, agreement, judge).
This turns selection from a guess into a verifiable decision, which is what lets it work
outside SWE-bench: the selector never reads gold tests, only signals the agent already
has. Implemented in [harness/fusion_select_exec.py](../harness/fusion_select_exec.py).

## 5. Cost-aware routing

Running N agents on every task is impractical in real use. Routing consults the same
candidates cheapest-first and stops early:

- Difficulty routing: try the cheapest solo first; finish if its build and tests pass.
- Escalate on disagreement: call the next candidate only when the first fails its execution
  signals.
- Early stop: accept the first candidate whose signals are clean and skip the rest.

On disc10 routing accepts the cheapest candidate whose signals are clean and escalates only
when they are not, resolving 9/10; it diverges from full selection only on sympy-15875,
where cheapest-first does not surface the one correct patch. Implemented in
[harness/fusion_route.py](../harness/fusion_route.py).

## 6. Generalization and room to improve

SWE-bench has two things real coding work does not: a fixed problem set and hidden gold
tests that grade each patch. The selector here never reads those gold tests - it picks
among candidate diffs from the issue text alone, the same information a real coding agent
has, and grading happens offline after selection. So the selection mechanism itself is
general: it would run unchanged on a real repository where no gold tests exist, with
qualitative human review standing in for the offline grade. SWE-bench is used here only as
a method-validation harness, because it can confirm objectively whether the selected patch
was the right one.

What disc10 shows, and what it does not. It shows that when an open panel genuinely
disagrees, execution-grounded selection can reach the oracle ceiling and beat both the best
solo and a proprietary SOTA baseline. It does not show that this holds on a random or full
Verified split: disc10 was curated to contain cross-model variance, so it is the favorable
case for selection, and the honest next step is to measure the same aggregators on an
uncurated sample where complementarity is rarer. A mixed open/proprietary panel is the
natural configuration to test next.

To turn this small result into evidence for a general-purpose coding selector:

- Scale the evaluation. 10 instances is a discriminating signal, not a benchmark. Re-run on
  an uncurated 50 to 500 Verified instances, across more than one language and task type,
  for stable numbers.
- Use a held-out test split. Split a repo's tests into a set visible to the agent and
  selector and a hidden set used only for grading, so the selector is never near the
  answer.
- Choose the panel on a train/test split. Decide panel membership on a train split and
  report only on a test split, which removes any post-hoc fit.
- Cover test-less tasks. For docs, UI, and refactors with no tests, calibrate an LLM
  rubric judge against human ratings, or collect proxy signals (CI pass rate, review
  acceptance, post-merge revert rate).
- Add a mixed panel. Test a mixed open/proprietary panel to see whether selection lifts the
  ceiling further.

This is still a proof of concept. The claim is narrow: on a discriminating set where the
open panel disagrees, execution-grounded selection reaches the oracle ceiling and beats the
best solo and the SOTA baseline; the selection mechanism is general because it never reads
the answer, and the way to prove it on real coding work is to keep the runtime verification
signals it already uses and re-validate with a held-out, train/test-split methodology at
scale.

## Practical deployment

On this limited set, execution-grounded fusion showed broader problem coverage than the
SOTA model, but it also cost more (10/10 at $0.56 per resolved versus gpt-5.4's 7/10 at
$0.43). In practice, rather than wiring it in as a single coding-agent backend that pays
the full panel on every task, a more cost-and-coverage-balanced option is to expose it as
an MCP tool that is only invoked when a problem is hard - a cheap single agent handles the
easy majority, and the panel is escalated to only when the difficulty warrants it. That
keeps the extra coverage available for the cases that actually need it without paying panel
cost on every task.

## Companion benchmark

For single-model SWE-bench results on these same Foundry-served models, see the
[foundry-model-benchmark](https://github.com/jisunchoii/foundry-model-benchmark)
repository (reports for SWE-bench Verified 500 and a custom coding eval). This repository
is the ensemble companion: it shares the harness that turns those single-model candidates
into a selection-based ensemble.
