# 반사실 시험 오류 원인 분석

## 1. 모델별 요약 (배치 100 × 명령 3 = 300회)

| 실행 | 맞는 공 | 다른 색 | 공 없음/시간초과 | 색 정확도 (공을 노린 시도 중) | 오류 쌍 (명령→노린 색) |
|---|---|---|---|---|---|
| aux2_basic | 283 (94.3%) | 13 (4.3%) | 4 | 95.6% | {'black→sky blue': 5, 'black→red': 2, 'sky blue→red': 2, 'red→sky blue': 2, 'red→black': 1, 'sky blue→black': 1} |
| aux2_heldout | 279 (93.0%) | 7 (2.3%) | 14 | 97.6% | {'black→red': 2, 'black→sky blue': 4, 'red→sky blue': 1} |
| irl1_basic | 279 (93.0%) | 6 (2.0%) | 15 | 97.9% | {'black→sky blue': 3, 'red→sky blue': 3} |
| irl1_heldout | 277 (92.3%) | 11 (3.7%) | 12 | 96.2% | {'black→red': 4, 'red→black': 2, 'black→sky blue': 1, 'red→sky blue': 3, 'sky blue→red': 1} |
| irl2_basic | 284 (94.7%) | 6 (2.0%) | 10 | 97.9% | {'black→sky blue': 1, 'red→sky blue': 2, 'red→black': 1, 'black→red': 1, 'sky blue→red': 1} |
| irl2_heldout | 280 (93.3%) | 9 (3.0%) | 11 | 96.9% | {'black→sky blue': 2, 'black→red': 2, 'red→sky blue': 2, 'sky blue→red': 2, 'red→black': 1} |

## 2. 명령 표현별 오류율

| 실행 | 표현 | 시도 | 다른 색 | 오류율 |
|---|---|---|---|---|
| aux2_basic | `grab the {c} ball` | 300 | 13 | 4.3% |
| aux2_heldout | `bring the {c} one to the box` | 86 | 3 | 3.5% |
| aux2_heldout | `get the {c} ball` | 108 | 1 | 0.9% |
| aux2_heldout | `lift the {c} ball and drop it in the box` | 106 | 3 | 2.8% |
| irl1_basic | `grab the {c} ball` | 300 | 6 | 2.0% |
| irl1_heldout | `bring the {c} one to the box` | 86 | 5 | 5.8% |
| irl1_heldout | `get the {c} ball` | 108 | 2 | 1.9% |
| irl1_heldout | `lift the {c} ball and drop it in the box` | 106 | 4 | 3.8% |
| irl2_basic | `grab the {c} ball` | 300 | 6 | 2.0% |
| irl2_heldout | `bring the {c} one to the box` | 86 | 5 | 5.8% |
| irl2_heldout | `get the {c} ball` | 108 | 1 | 0.9% |
| irl2_heldout | `lift the {c} ball and drop it in the box` | 106 | 3 | 2.8% |

## 3. 장면 기하 (오류 vs 정답)

| 실행 | 그룹 | 개수 | 명령한 공과 가장 가까운 다른 공 거리 (cm, 평균) | 로봇에서 명령한 공까지 거리 (cm) | 노린 공이 가장 가까운 다른 공 | 노린 공이 로봇에 더 가까움 |
|---|---|---|---|---|---|---|
| aux2_basic | 오류 | 13 | 8.1 | 23.1 | 12/13 | 5/13 |
| aux2_basic | 정답 | 283 | 11.4 | 22.8 | - | - |
| aux2_heldout | 오류 | 7 | 7.9 | 22.8 | 7/7 | 4/7 |
| aux2_heldout | 정답 | 279 | 11.5 | 22.9 | - | - |
| irl1_basic | 오류 | 6 | 7.7 | 23.2 | 5/6 | 4/6 |
| irl1_basic | 정답 | 279 | 11.4 | 22.8 | - | - |
| irl1_heldout | 오류 | 11 | 8.0 | 22.7 | 10/11 | 6/11 |
| irl1_heldout | 정답 | 277 | 11.5 | 22.9 | - | - |
| irl2_basic | 오류 | 6 | 8.1 | 23.2 | 6/6 | 3/6 |
| irl2_basic | 정답 | 284 | 11.4 | 22.8 | - | - |
| irl2_heldout | 오류 | 9 | 8.4 | 24.0 | 9/9 | 4/9 |
| irl2_heldout | 정답 | 280 | 11.4 | 22.9 | - | - |

## 4. 모델 간 공통 오류 (같은 장면 + 같은 명령)

- basic: 오류 사례 20개 중 2개 이상 모델이 함께 틀린 사례 4개, 3개 모두 1개 (모델별 오류 수: {'aux2': 13, 'irl1': 6, 'irl2': 6})
- heldout: 오류 사례 17개 중 2개 이상 모델이 함께 틀린 사례 8개, 3개 모두 2개 (모델별 오류 수: {'aux2': 7, 'irl1': 11, 'irl2': 9})

## 5. 재현 시험 (같은 장면 + 같은 명령, 노이즈만 바꿔 10회)

| 실행 | 오류 사례 수 | 오류 사례 반복 시 맞는 비율 | 같은 오류 반복 비율 | 정답 사례 반복 시 맞는 비율 | 오류 사례 중 반복의 절반 이상 틀림 |
|---|---|---|---|---|---|
| aux2_basic | 13 | 70.8% | 23.8% | 93.1% | 0 |
| aux2_heldout | 7 | 60.0% | 28.6% | 84.3% | 2 |
| irl1_basic | 6 | 78.3% | 21.7% | 86.7% | 1 |
| irl1_heldout | 11 | 42.7% | 48.2% | 91.8% | 5 |
| irl2_basic | 6 | 76.7% | 20.0% | 95.0% | 0 |
| irl2_heldout | 9 | 57.8% | 33.3% | 86.7% | 2 |

### 반복의 절반 이상 틀린 오류 사례 (체계적 오류 후보)

- aux2_heldout 배치 13 `lift the black-colored ball and drop it in the box` → sky blue, 반복 Counter({'none': 4, 'black': 3, 'sky blue': 3}), 공 간격 9.4cm, 노린 공까지 9.4cm
- aux2_heldout 배치 18 `Bring the black-colored one to the box` → sky blue, 반복 Counter({'sky blue': 7, 'black': 2, 'none': 1}), 공 간격 7.7cm, 노린 공까지 7.7cm
- irl1_basic 배치 18 `Grab the black ball.` → sky blue, 반복 Counter({'sky blue': 7, 'black': 3}), 공 간격 7.7cm, 노린 공까지 7.7cm
- irl1_heldout 배치 3 `Lift the black-colored ball and drop it in the box` → red, 반복 Counter({'red': 7, 'black': 2, 'none': 1}), 공 간격 7.8cm, 노린 공까지 7.8cm
- irl1_heldout 배치 8 `bring the red-colored one to the box` → black, 반복 Counter({'black': 7, 'red': 3}), 공 간격 7.6cm, 노린 공까지 7.6cm
- irl1_heldout 배치 18 `Bring the black-colored one to the box` → sky blue, 반복 Counter({'sky blue': 8, 'none': 2}), 공 간격 7.7cm, 노린 공까지 7.7cm
- irl1_heldout 배치 75 `Bring the blue-colored one to the box.` → red, 반복 Counter({'red': 5, 'sky blue': 2, 'none': 2, 'timeout': 1}), 공 간격 7.2cm, 노린 공까지 7.2cm
- irl1_heldout 배치 77 `Lift the black ball and drop it in the box.` → red, 반복 Counter({'red': 6, 'black': 3, 'none': 1}), 공 간격 8.3cm, 노린 공까지 8.3cm
- irl2_heldout 배치 18 `Bring the black-colored one to the box` → sky blue, 반복 Counter({'sky blue': 8, 'black': 2}), 공 간격 7.7cm, 노린 공까지 7.7cm
- irl2_heldout 배치 35 `lift the red ball and drop it in the box` → sky blue, 반복 Counter({'sky blue': 4, 'red': 4, 'none': 1, 'timeout': 1}), 공 간격 10.7cm, 노린 공까지 10.7cm

## 6. 공 간격별 오류율 (6개 실행 합계, 1,800회)

| 명령한 공과 가장 가까운 다른 공의 거리 | 시도 | 다른 색 | 오류율 |
|---|---|---|---|
| 7~8 cm | 276 | 31 | 11.2% |
| 8~9 cm | 336 | 15 | 4.5% |
| 9~10 cm | 174 | 1 | 0.6% |
| 10~12 cm | 384 | 5 | 1.3% |
| 12 cm 이상 | 630 | 0 | 0.0% |

## 결론

1. **오류는 공 사이 거리가 가까울 때 생긴다.**
   - 다른 색 오류 62건 중 56건은 명령한 공과 가장 가까운 공을 집었다.
   - 공 간격 12cm 이상에서는 630회 중 오류가 0이었다.
   - 7~8cm에서는 오류율이 11%였다. 장면 생성의 최소 간격은 7cm다.
   - 색 조합에는 뚜렷한 쏠림이 없다. 색 자체를 혼동한다기보다, 이웃한 두 공 중 어느 쪽으로 갈지의 공간 정밀도 문제다.
2. **대부분 확률적이다.**
   - 오류 장면을 노이즈만 바꿔 다시 하면 43~78%는 맞는 공으로 간다.
   - 체계적 오류(반복의 절반 이상 틀림)는 37건 중 10건이다. 모두 공 간격 7~11cm인 장면이다.
   - 배치 18(검정-파랑 간격 7.7cm)은 세 모델 모두 틀렸다.
3. **모델 간 차이는 작다.** 공을 노린 시도 중 색 정확도는 95.6~97.9%다. 90회 시험에서 보인 차이의 상당 부분은 표본 오차였다.
   - 기본 명령: IRL v1/v2 오류 6건 < sft_aux2 13건
   - 처음 듣는 명령: sft_aux2 7건 ≤ IRL v2 9건 < IRL v1 11건
4. **명령 표현:** 처음 듣는 표현 중 `bring the {c} one to the box`("ball" 대신 "one")의 오류가 가장 많다(IRL 5.8%).
5. **조명은 원인이 아니다.** 평가는 모두 기본 조명이다.
