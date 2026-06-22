# foundry-coding-ensemble

영어 원문은 여기에서 볼 수 있습니다: [README.md](README.md)

Foundry에서 서빙하는 여러 코딩 모델을 하나만 쓰는 대신 후보로 두고 그중에서 고르는 선택
기반(selection) 방식이 SWE-bench에서 풀리는 문제의 범위를 넓힐 수 있는지를 묻는 작은 개념
증명(proof of concept)입니다. 성능 향상을 입증한 것이 아니라 그 가능성을 탐색한 것으로, 목적은
이 접근이 커버리지를 넓힐 수 있는지 자체를 점검하는 것이었고 제한된 큐레이션 세트에서 그
가능성의 일부를 확인했습니다.

방식은 OpenRouter의 Fusion(여러 모델을 하나의 deep-research 답변으로 합성)에서 영감을 받되,
에이전트 코딩에 맞게 바꿨습니다. 매 턴 답을 합성(synthesis)하는 대신, 각 모델이 독립적으로
전체 에이전트 루프를 돌려 후보 패치를 하나씩 만들고, 집계기가 완성된 후보들 중에서 고릅니다.
두 가지 집계기를 실측합니다: 실행 기반 선택(모든 후보를 빌드, 회귀, 자작 테스트, 타입/린트로
점수화하고 동점일 때만 LLM 심판이 최종 선택을 판단)과 비용 인지 라우팅(후보를 저렴한
순으로 조회해 처음 깨끗한 후보를 채택).

교차 모델 분산을 드러내도록 큐레이션한 10개 SWE-bench Verified 인스턴스(disc10 세트)에서, 실행
기반 선택은 오라클 best-of-4 상한(10/10)에 도달해 최고 단일 오픈 모델(glm-5.1, 9/10)과 단일 모델
SOTA 베이스라인(gpt-5.4, 7/10)보다 앞섭니다. 이는 일반적인 결과가 아니라 유리한 세트에서 확인된
가능성으로 읽어야 합니다: disc10은 패널이 의견을 달리하는 문제를 골라 구성했으므로 선택이 도움이
될 여지가 가장 큰 경우입니다. 인스턴스별 해결 매트릭스와 정직한 단서(큐레이션된 변별 세트,
소규모)까지 담은 전체 결과는
[reports/fusion-ensemble-report_ko.md](reports/fusion-ensemble-report_ko.md)를 참고하세요.

단서: 이것은 의도적으로 간략화한 에이전트 하네스입니다. 체감 품질은 단일 에이전트든
앙상블이든 실제 코딩 에이전트의 클라이언트 사이드 하네스(Claude Code, Codex, Copilot, Aider)
성숙도에 따라 크게 달라지므로, 수치는 절대 역량이 아니라 '이 하네스 안에서의 상대 비교'로
읽어야 합니다.

헤드라인은 그 한계와 함께 읽어야 합니다. disc10은 SWE-bench Verified에서 큐레이션한 변별
조각입니다. 열 개 인스턴스를 무작위로 뽑은 게 아니라 패널이 의견을 달리하는 문제를 골랐기
때문에, 선택기가 도움이 될 여지가 가장 큰 세트이며 SWE-bench를 대표하는 표본도, 하물며 여러
언어와 리포지토리에 걸친 실제 코딩 작업을 대표하는 표본도 아닙니다. 패널을 불일치 기준으로
골랐으니 정답의 합집합이 유난히 크고, 이것이 바로 선택이 파고드는 조건입니다. 무작위 분할이라면
오라클 상한이 최고 단일 모델에 훨씬 가까워지고 여지도 줄어듭니다. 오픈웨이트 순위도 과대
해석하기 쉽습니다: 오픈 모델은 프런티어 모델에서 증류되고 공개 벤치마크를 상대로 학습되는
경우가 많고 SWE-bench Verified는 공개 데이터이므로, glm-5.1의 우위와 minimax-m2.5가 SOTA
베이스라인과 맞먹는 것은 전이 가능한 실력이 아니라 벤치마크에 대한 익숙함을 일부 반영한 것일 수
있습니다. 이것은 개념 증명입니다. 신뢰할 만한 순위를 얻으려면 더 큰 비큐레이션 평가, 사후 적합
없이 패널을 고르도록 채점용 데이터를 따로 떼어 두는 분리(held-out), 그리고 모델이 본 적 없는 비공개 과제에서의 검증이 필요합니다.

이 프로젝트의 단일 모델 짝(companion)은
[foundry-model-benchmark](https://github.com/jisunchoii/foundry-model-benchmark)
리포지토리이며, 각 모델을
단독으로 돌린 SWE-bench Verified 및 커스텀 코딩 평가 점수를 정리합니다. 이 리포지토리는
그 단일 모델 후보들을 앙상블로 묶는 하네스(harness)를 더합니다. 두 리포지토리는 같은 총량(gross)
비용 방식과 같은 공식 SWE-bench Verified Docker 하네스를 공유하지만, 채점하는 인스턴스
서브셋이 다릅니다(foundry-model-benchmark는 자체 선정 세트를, 여기 수치는 disc10 세트를
평가). 따라서 두 리포지토리의 모델별 수치를 직접 비교할 수는 없습니다.

## 결과 요약 (disc10 Verified 서브셋, 에이전트 루프)

| 접근 방식                           | 해결    | 비율  | $/해결 |
|------------------------------------|:------:|:----:|:------:|
| 실행 기반 선택 (fusion)             | 10/10  | 100% | 0.56   |
| 최고 단일 오픈 모델 (glm-5.1)        | 9/10   | 90%  | 0.23   |
| 비용 인지 라우팅 (fusion)           | 9/10   | 90%  | 0.43   |
| gpt-5.4 (단일 모델 SOTA)            | 7/10   | 70%  | 0.43   |
| 최저가 강한 오픈 모델 (minimax-m2.5) | 7/10   | 70%  | 0.05   |
| 오라클 best-of-4 상한               | 10/10  | 100% | -      |

disc10에서는 오픈 패널에 실제 교차 모델 상보성이 있습니다(glm-5.1이 놓치는 matplotlib-25311을
minimax-m2.5가 풉니다). 그래서 오라클 best-of-4 상한이 10/10입니다. 실행 기반 선택은 그 상한에
도달해 최고 단일 모델보다 하나, SOTA보다 셋을 더 풀고, 비용 인지 라우팅은 9/10을 더 싸게
달성합니다. 대가는 해결당 비용입니다: 선택은 전체 패널과 심판 비용을 치르므로, 여기서의 가치는
최저가가 아니라 추가 커버리지입니다. disc10은 모델 차이를 드러내려 큐레이션한 변별 세트이므로
100%는 '이 세트에서의 상한 신호'이지 벤치마크급 해결률이 아닙니다. 비용은 총량(gross) 기준(캐시
할인 없음)이며, 달러 수치를 신뢰하기 전 [harness/pricing.json](harness/pricing.json)을 본인
요율로 맞춰 두세요.

## 구성

```
harness/
  swe_eval.py        단일 샷 예측 생성기 (oracle context).
  agent_eval.py      SWE-bench Docker 이미지 안의 에이전트 루프; git diff -> 패치.
  fusion_select.py   초기 집계기: 심판이 최고 후보 패치를 선택.
  fusion_synth.py    초기 집계기: 후보들을 하나의 diff로 합성, apply-gate 적용.
  fusion_select_exec.py  실행 기반 선택: 실행 신호(적용, 컴파일, 스위트
                         임포트, 자작 테스트)로 점수화, 동점일 때만 심판이 판단.
                         gold 테스트 불필요.
  fusion_route.py    fusion_select_exec 위의 비용 인지 라우팅: 후보를 저렴한
                     순으로 소비하고 처음 깨끗한 후보를 채택, 필요할 때만
                     상위로 에스컬레이션.
  oracle_report.py   best-of-N 상한과 선택기 효율.
  cost_report.py     리포트와 가격으로 해결률과 $/해결 순위 산출.
  verified_prep.py   oracle context를 포함한 Verified 서브셋 생성.
  wsl_score.sh       공식 Docker 하네스로 예측 채점 (WSL).
  configs.example.json  엔드포인트 목록 (베이스라인 + 오픈웨이트 solo 패널).
  pricing.json       cost_report가 사용하는 1M 토큰당 정가.
reports/
  fusion-ensemble-report.md  전체 방법과 결과.
```

## 실행 방법

하네스는 각 모델을 슬러그(slug)로 노출하는 OpenAI 호환 게이트웨이(예: 포트 4000에서 도는 로컬
LiteLLM 프록시)와 통신합니다. 게이트웨이 키는 환경 변수에 설정하세요:

```
$env:LITELLM_KEY = "<your gateway key>"
```

여기서 쓰는 모델 슬러그(gpt-5.4, glm-5.1, minimax-m2.5, deepseek-v4-pro, kimi-k2)는
게이트웨이 고유 이름이지 공개 모델 id가 아닙니다. 오픈웨이트 패널 구성은 임의로 고른
것이므로, 재현할 때는 사용자 선호에 따라 원하는 모델로 얼마든지 바꿔도 됩니다. 다만
그러려면 패널에 넣을 각 모델의 엔드포인트를 배포하거나 확보해 LiteLLM(또는 다른
OpenAI 호환 게이트웨이) 설정에 연결해, 하네스가 슬러그로 각각을 호출할 수 있게 해야 합니다.

1. 평가 서브셋 생성 (`datasets` 패키지 필요, swebench venv에서 실행):

   ```
   python harness/verified_prep.py --n 5 --seed 0 --out verified_subset.jsonl
   ```

   리포트에서 사용한 disc10 세트를 그대로 재현하려면 무작위 샘플 대신 큐레이션한
   10개 인스턴스 id를 직접 넘기면 됩니다:

   ```
   python harness/verified_prep.py --out disc10.jsonl --ids \
     "astropy__astropy-14508,django__django-12039,django__django-15161,matplotlib__matplotlib-25311,pydata__xarray-4695,pydata__xarray-7393,pytest-dev__pytest-7236,sphinx-doc__sphinx-7889,sympy__sympy-12419,sympy__sympy-15875"
   ```

2. 에이전트 루프로 오픈웨이트 모델마다 후보 하나씩 생성:

   ```
   python harness/agent_eval.py --dataset verified_subset.jsonl \
       --configs harness/configs.example.json --config-name solo-minimax-m2.5 \
       --out preds_agent_minimax.jsonl
   ```

   각 패널 모델(kimi-k2, deepseek-v4-pro, glm-5.1, ...)에 대해 반복합니다.

3. 후보들을 집계.

   선택 (LLM 심판), 초기 집계기:

   ```
   python harness/fusion_select.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select.jsonl
   ```

   합성 (마지막 병합), 초기 집계기, 동일한 candidate 플래그에 `--synth-model` 추가:

   ```
   python harness/fusion_synth.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --synth-model deepseek-v4-pro --out preds_fusion-synth.jsonl
   ```

   실행 기반 선택 (gold 테스트 불필요, agent_eval과 동일하게 WSL + Docker 필요):

   ```
   python harness/fusion_select_exec.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select-exec.jsonl
   ```

   비용 인지 라우팅 (저렴한 순으로 소비, 필요 시 에스컬레이션):

   ```
   python harness/fusion_route.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --order minimax-m2.5,deepseek-v4-pro \
       --judge-model deepseek-v4-pro --out preds_fusion-route.jsonl
   ```

4. 공식 Docker 하네스로 채점 (WSL 내부):

   ```
   bash harness/wsl_score.sh refine /mnt/c/.../preds_fusion-select.jsonl \
       /mnt/c/.../preds_fusion-synth.jsonl
   ```

5. 분석:

   ```
   python harness/cost_report.py --run-id refine --configs harness/configs.example.json \
       --names fusion-select fusion-synth --pricing harness/pricing.json
   python harness/oracle_report.py --reports report_solo-*.refine.json \
       --selected report_fusion-select.refine.json report_fusion-synth.refine.json
   ```

## 참고

- 하네스 모듈은 표준 라이브러리 Python입니다. 채점은 `swebench` 패키지를, 서브셋 준비는
  `datasets`를 사용합니다 (harness/requirements.txt 참고).
- 예측 파일(`preds_*.jsonl`), 하네스 리포트(`report_*.json`), 로그는 실행 산출물이며
  커밋하지 않습니다. 위 단계로 재생성하세요.
- gpt-5.4는 단일 모델 베이스라인이며 앙상블 멤버가 아닙니다.
## 범용 방향

SWE-bench는 숨은 gold 테스트로 채점하지만 실무 코딩에는 그게 없습니다. 여기 선택기는 그
테스트를 읽지 않고 이슈 텍스트만으로 후보 diff 중 하나를 고르므로 메커니즘은 범용입니다.
실행 기반 선택(`harness/fusion_select_exec.py`)과 비용 인지 라우팅(`harness/fusion_route.py`)은
이 서브셋에서 실측됐습니다: 실행 기반 선택은 best-of-4 오라클 상한(10/10)에 도달해 최고 단일
오픈 모델(glm-5.1, 9/10)과 독점 SOTA 베이스라인(gpt-5.4, 7/10)보다 모두 앞서고, 비용 인지
라우팅은 더 낮은 비용으로 9/10에 도달합니다(위 실행 단계 참고). 이들은 gold 테스트 oracle을
런타임 검증 신호(빌드, 기존 테스트 회귀, 자작 테스트, 타입/린트)로 대체하므로 gold 테스트가
필요 없습니다. disc10은 교차 모델 분산을 담도록 큐레이션된 변별 세트이므로, 이 결과는 패널이
의견을 달리할 때 선택이 오라클 상한에 도달할 수 있음을 보여줄 뿐 무작위나 전체 분할에서
그렇다는 것은 아닙니다. 남은 일은 비큐레이션 표본에서 평가 규모를 키우고, 오픈/독점 혼합
패널을 시험하는 것입니다. 실무에서는 매 과제를 채점하지 않고
사람의 정성 리뷰가 대신하며, SWE-bench는 방법 검증용 하네스로만 둡니다. 자세한 내용은
[reports/fusion-ensemble-report_ko.md](reports/fusion-ensemble-report_ko.md)의 6절을
참고하세요.
