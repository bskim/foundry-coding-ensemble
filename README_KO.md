# foundry-coding-ensemble

영어 원문은 여기에서 볼 수 있습니다: [README.md](README.md)

Foundry에서 서빙하는 여러 코딩 모델을 선택 기반(selection) 앵상블로 묶어 SWE-bench에서
평가하고, 이 앵상블이 가장 좋은 단일 모델보다 해결률과 비용 면에서 더 나은지 확인하는
작은 개념 증명(proof of concept)입니다.

OpenRouter의 Fusion(여러 모델을 하나의 deep-research 답변으로 합성)에서 영감을 받되,
에이전트 코딩에 맞게 바꿈습니다. 매 턴 답을 합성(synthesis)하는 대신, 각 모델이 독립적으로
전체 에이전트 루프를 돌려 후보 패치를 하나씩 만들고, 마지막에 심판(judge) 모델이 그중
가장 좋은 패치 하나를 고릅니다(Select-on-N). 테스트 서븋셋에서 이 단순한 방식이
best-of-N 상한(ceiling)에 도달했고, 단일 모델 SOTA 베이스라인보다 높은 문제해결 커버리지를 난
유일한 구성이었습니다.
비용 트레이드오프와 더 저렴하고 강하게 만드는 방법까지 담은 전체 결과는
[reports/fusion-ensemble-report_ko.md](reports/fusion-ensemble-report_ko.md)를 참고하세요.

이 프로젝트의 단일 모델 짝(companion)은 foundry-model-benchmark 리포지토리이며, 각 모델을
단독으로 돌린 SWE-bench Verified 및 커스텀 코딩 평가 점수를 정리합니다. 이 리포지토리는
그 단일 모델 후보들을 앵상블로 묶는 하니스(harness)를 더합니다.

## 결과 요약 (5개 인스턴스 Verified 서브셋, 에이전트 루프)

| 접근 방식                         | 해결    | 비율 | $/해결 |
|----------------------------------|:------:|:----:|:------:|
| 최고 단일 오픈웨이트 모델          | 3/5    | 60%  | 0.17   |
| gpt-5.4 (단일 모델 SOTA)          | 2/5    | 40%  | 2.12   |
| 선택 앙상블 (Plan A)              | 4/5    | 80%  | 1.79   |
| 합성 앙상블 (Plan B)              | 4/5    | 80%  | 1.79   |
| 비용 최소 2-모델 패널             | 4/5    | 80%  | 0.87   |

이 수치는 소규모 신호일 뿐 벤치마크급 결과가 아닙니다. 비용은 gross 기준(캐시 할인 없음)입니다.
달러 수치를 그대로 신뢰하기 전에 [harness/pricing.json](harness/pricing.json)을 본인 요율로
맞춰 두세요.

## 구성

```
harness/
  swe_eval.py        단일 샷 예측 생성기 (oracle context).
  agent_eval.py      SWE-bench Docker 이미지 안의 에이전트 루프; git diff -> 패치.
  fusion_select.py   Plan A: 심판이 최고 후보 패치를 선택 (Select-on-N).
  fusion_synth.py    Plan B: 후보들에 대해 하나의 diff로 합성, apply-gate 적용.
  fusion_select_exec.py  범용 Plan A: 실행 신호(적용, 컴파일, 스위트 임포트,
                         자작 테스트)로 선택, 동점일 때만 심판 사용. gold
                         테스트 불필요.
  fusion_route.py    fusion_select_exec 위의 비용 인지 라우팅: 후보를 저렴한
                     순으로 소비하고 처음 깨끗한 후보를 채택, 필요할 때만
                     상위로 에스컬레이션.
  oracle_report.py   best-of-N 상한과 선택기 효율.
  cost_report.py     리포트와 가격으로 해결률과 $/해결 순위 산출.
  verified_prep.py   oracle context를 포함한 Verified 서브셋 생성.
  wsl_score.sh       공식 Docker 하니스로 예측 채점 (WSL).
  configs.example.json  엔드포인트 목록 (베이스라인 + 오픈웨이트 solo 패널).
  pricing.json       cost_report가 사용하는 1M 토큰당 정가.
reports/
  fusion-ensemble-report.md  전체 방법과 결과.
```

## 실행 방법

하니스는 각 모델을 슬러그(slug)로 노출하는 OpenAI 호환 게이트웨이(예: 포트 4000에서 돌는 로컬
LiteLLM 프록시)와 통신합니다. 게이트웨이 키는 환경 변수에 설정하세요:

```
$env:LITELLM_KEY = "<your gateway key>"
```

1. 평가 서브셋 생성 (`datasets` 패키지 필요, swebench venv에서 실행):

   ```
   python harness/verified_prep.py --n 5 --seed 0 --out verified_subset.jsonl
   ```

2. 에이전트 루프로 오픈웨이트 모델마다 후보 하나씩 생성:

   ```
   python harness/agent_eval.py --dataset verified_subset.jsonl \
       --configs harness/configs.example.json --config-name solo-minimax-m2.5 \
       --out preds_agent_minimax.jsonl
   ```

   각 패널 모델(kimi-k2, deepseek-v4-pro, glm-5.1, ...)에 대해 반복합니다.

3. 후보들을 집계.

   선택 (Plan A):

   ```
   python harness/fusion_select.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --judge-model deepseek-v4-pro --out preds_fusion-select.jsonl
   ```

   합성 (Plan B), 동일한 candidate 플래그에 `--synth-model` 추가:

   ```
   python harness/fusion_synth.py --dataset verified_subset.jsonl \
       --candidate minimax-m2.5=preds_agent_minimax.jsonl \
       --candidate deepseek-v4-pro=preds_agent_deepseek.jsonl \
       --synth-model deepseek-v4-pro --out preds_fusion-synth.jsonl
   ```

   실행 신호 기반 범용 선택 (gold 테스트 불필요, agent_eval과 동일하게 WSL +
   Docker 필요):

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

4. 공식 Docker 하니스로 채점 (WSL 내부):

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

- 하니스 모듈은 표준 라이브러리 Python입니다. 채점은 `swebench` 패키지를, 서브셋 준비는
  `datasets`를 사용합니다 (harness/requirements.txt 참고).
- 예측 파일(`preds_*.jsonl`), 하니스 리포트(`report_*.json`), 로그는 실행 산출물이며
  커밋하지 않습니다. 위 단계로 재생성하세요.
- gpt-5.4는 단일 모델 베이스라인이며 앙상블 멤버가 아닙니다.
## 범용 방향 (다음 단계)

SWE-bench는 숨은 gold 테스트로 채점하지만 실무 코딩에는 그게 없습니다. 여기 선택기는 그
테스트를 읽지 않고 이슈 테스트만으로 후보 diff 중 하나를 고르므로 메커니즘은 범용이지만,
보고된 수치(특히 사후로 고른 2-모델 패널)는 벤치마크에 일부 맞춰져 있습니다. 범용 코딩
선택기로 쓰려면 gold 테스트 oracle을 런타임 검증 신호(빌드, 기존 테스트 회귀, 자작
테스트, 타입/린트)로 대체하고 held-out + train/test 분리 방법론으로 검증하는 것이
계획입니다. 실무에서는 매 과제를 채점하지 않고 정성적 사람 리뷰가 갈음하며, SWE-bench는
방법 검증용 하니스로만 둡니다. Plan C(비용 라우팅)는 Plan A(실행 기반 선택) 위에
얹히므로, Plan A를 먼저 측정하고 이어 Plan A + C를 측정해 커버리지는 유지하면서 비용이
줄어드는지 확인합니다. Plan A와 Plan C는 이제 `harness/fusion_select_exec.py`와
`harness/fusion_route.py`로 구현되어 있습니다(위 실행 단계 참고). 전체 계획은
[reports/fusion-ensemble-report_ko.md](reports/fusion-ensemble-report_ko.md)의 6절을 참고하세요.