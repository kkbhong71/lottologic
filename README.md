# lottologic — LOTTO ULTIMATE 우주 최강 로또 예측 시스템

> 32종 알고리즘 + 6대 방법론을 통합한 로또 6/45 분석 엔진.
> GitHub Actions가 매주 자동 실행하고, GitHub Pages가 결과를 보여줍니다.
> **배포 페이지:** https://kkbhong71.github.io/lottologic/

## 정직한 전제
- 로또 1등 확률은 1/8,145,060이며 어떤 알고리즘도 이를 바꾸지 못합니다.
- 이 엔진의 유일한 수학적 레버는 **인기 조합 회피** — 당첨 시 배당을 나누는 사람을 줄이는 것입니다.
- 엔진은 매 실행마다 15종 무작위성 검정과 walk-forward 백테스트로 스스로를 검증합니다.

## 구조
```
index.html                  ← Pages 대시보드 (results/latest.json 표시)
engine/lotto_ultimate.py    ← 코어: 6중 방어 로더 · 8종 통계 · 인기도 회피 · 8종 필터 · 백테스트 · 예측 추적
engine/lotto_ultimate_ml.py ← RF+GB · 네트워크 커뮤니티 · 유전 알고리즘 · 15종 검정
engine/lotto_ultimate_swarm.py ← 진정난수 시드 · VRF · PSO/ACO/점균류 군집지능
data/new_XXXX.csv           ← 당첨번호 (round, draw date, num1~num6) — 매주 추가
results/latest.json         ← Actions 자동 생성 결과
results/prediction_log.csv  ← 사전 등록 → 사후 채점 누적
.github/workflows/weekly.yml
```

## 매주 운영 방법 (이것만 하면 됩니다)
1. 토요일 추첨 후 `new_1242.csv`처럼 최신 회차까지 담긴 파일을 `data/`에 올립니다 (GitHub 웹에서 *Add file → Upload files*).
2. 커밋하면 Actions가 자동으로 엔진을 실행합니다 (약 5~10분).
3. 페이지가 갱신됩니다. 직전 회차 예측은 자동 채점되어 "예측 추적 기록"에 쌓입니다.

수동 실행: **Actions → LOTTO ULTIMATE weekly run → Run workflow**.

## 처음 한 번 설정
1. 저장소 **Settings → Pages → Source: Deploy from a branch → `main` / `/ (root)`** 저장.
2. **Settings → Actions → General → Workflow permissions → Read and write** 선택 (결과 커밋에 필요).
3. `data/`에 첫 CSV를 올리면 첫 실행이 시작됩니다.

## 로컬·Colab 실행
```bash
pip install -r engine/requirements.txt
cd engine
python lotto_ultimate.py ../data/new_1241.csv --sets 10 --ml --ga --net --swarm --seed auto --vrf 2 --tests --backtest 40 --out ../results --json
```
| 옵션 | 내용 |
|---|---|
| `--ml` | RandomForest + GradientBoosting 볼별 출현확률 |
| `--ga` | 유전 알고리즘 조합 최적화 (적응형 돌연변이) |
| `--net` | 동반출현 네트워크 커뮤니티 8개 |
| `--swarm` | PSO · ACO · 점균류 군집지능 + 앙상블 합의 |
| `--seed auto` | 양자난수(ANU) → 대기잡음(random.org) → 지진파(USGS) → OS CSPRNG 폴백 |
| `--vrf N` | SHA-256 검증가능 난수 조합 N세트 (커밋 해시 공개) |
| `--tests` | 15종 무작위성 검정 (Holm-Bonferroni) |
| `--backtest N` | 최근 N회 walk-forward 백테스트 vs 랜덤 대조군 |
| `--pop-penalty λ` | 인기도 감점 강도 (기본 1.0) |

## 면책
본 프로젝트는 통계·알고리즘 연구 및 교육 목적의 오픈소스입니다. 당첨을 보장하지 않으며, 구매 결정과 결과에 대한 책임은 사용자에게 있습니다.
**도박 문제 상담: 1336** (한국도박문제예방치유원, 24시간)

MIT License · Oseong Vibe Coding EduHub · K.G.B.
