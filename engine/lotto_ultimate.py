#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOTTO ULTIMATE ENGINE — Step 1: 코어 (데이터·스코어링·인기도회피·필터·검증)
Creative Director: K.G.B. | Oseong Vibe Coding EduHub

원칙
 - 로또 6/45는 완전 무작위(1/8,145,060). 당첨 확률 자체는 어떤 알고리즘도 못 올린다.
 - 유일한 수학적 레버 = 인기 조합 회피(공동당첨 최소화 → 기대 배당금 최대화).
 - 모든 판단은 자기 검증(walk-forward 백테스트 + 랜덤 대조군 + 부트스트랩 CI)을 동반한다.

사용법 (Colab/로컬)
    python lotto_ultimate.py new_1241.csv --sets 10 --backtest 100
"""
from __future__ import annotations
import sys, csv, io, math, argparse, secrets, itertools
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

N_BALL, N_PICK = 45, 6

# =============================================================================
# [LAYER 1] DATA — 6중 방어 안전 로더
#   방어1 인코딩 자동감지(UTF-8-sig→CP949→EUC-KR→latin1)
#   방어2 3형식 자동인식(표준컬럼 / 동행복권 원본 / 헤더 없는 숫자열)
#   방어3 컬럼명 기반 추출(iloc[:,-6:] 금지 — 줄 끝 쉼표 유령열 대비)
#   방어4 값 범위 검사(1~45, 6개 서로 다름)
#   방어5 회차 연속성·중복 검사
#   방어6 정렬 보장(회차 오름차순, 번호 오름차순)
# =============================================================================
def load_lotto_csv(path: str) -> pd.DataFrame:
    raw = open(path, "rb").read()
    text = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):          # 방어1
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("인코딩 해독 실패")

    df = pd.read_csv(io.StringIO(text))
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]  # 방어3: 유령열 제거
    cols = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = cols

    # 방어2: 형식 인식
    num_cols = [c for c in cols if c.startswith("num")]
    if len(num_cols) == 6:                                   # 형식A 표준(round, draw_date, num1~6)
        rnd = "round" if "round" in cols else cols[0]
        out = pd.DataFrame({"round": df[rnd]})
        out["draw_date"] = df["draw_date"] if "draw_date" in cols else pd.NaT
        for i, c in enumerate(sorted(num_cols), 1):
            out[f"num{i}"] = df[c]
    elif any("회차" in c for c in cols):                    # 형식B 동행복권 원본(CP949)
        rnd = [c for c in cols if "회차" in c][0]
        picked = [c for c in cols if c.isdigit() or c.startswith("당첨번호") or c in
                  ("1","2","3","4","5","6")][:6]
        if len(picked) < 6:  # 컬럼명이 애매하면 회차 다음 6개의 정수형 열을 사용
            int_cols = [c for c in cols if pd.api.types.is_integer_dtype(df[c]) and c != rnd]
            picked = int_cols[:6]
        out = pd.DataFrame({"round": df[rnd], "draw_date": pd.NaT})
        for i, c in enumerate(picked, 1):
            out[f"num{i}"] = df[c]
    else:                                                    # 형식C 헤더 없는 숫자열
        df = pd.read_csv(io.StringIO(text), header=None)
        out = pd.DataFrame({"round": df.iloc[:, 0], "draw_date": pd.NaT})
        for i in range(6):
            out[f"num{i+1}"] = df.iloc[:, i + 1]

    for c in ["round"] + [f"num{i}" for i in range(1, 7)]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[f"num{i}" for i in range(1, 7)]).astype(
        {c: int for c in ["round"] + [f"num{i}" for i in range(1, 7)]})

    # 방어4: 값 범위·중복
    nums = out[[f"num{i}" for i in range(1, 7)]].to_numpy()
    bad = ((nums < 1) | (nums > 45)).any(axis=1) | (np.sort(nums, 1)[:, 1:] == np.sort(nums, 1)[:, :-1]).any(axis=1)
    if bad.any():
        print(f"⚠ 이상 행 {bad.sum()}건 제거: 회차 {out.loc[bad,'round'].tolist()[:10]}")
        out = out[~bad]
    # 방어5·6: 정렬, 회차 연속성
    out = out.sort_values("round").drop_duplicates("round").reset_index(drop=True)
    nums = np.sort(out[[f"num{i}" for i in range(1, 7)]].to_numpy(), axis=1)
    for i in range(6):
        out[f"num{i+1}"] = nums[:, i]
    gaps = np.diff(out["round"].to_numpy())
    if (gaps != 1).any():
        print(f"⚠ 회차 누락 지점 {int((gaps!=1).sum())}곳")
    return out


def data_audit(df: pd.DataFrame) -> dict:
    """무결성 감사 리포트 — 균등성 χ² 포함(자유도 44)."""
    nums = df[[f"num{i}" for i in range(1, 7)]].to_numpy().ravel()
    freq = np.bincount(nums, minlength=46)[1:]
    expected = len(nums) / 45
    chi2 = float(((freq - expected) ** 2 / expected).sum())
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, 44))
    return dict(rounds=len(df), first=int(df["round"].iloc[0]), last=int(df["round"].iloc[-1]),
                first_date=str(df["draw_date"].iloc[0]), last_date=str(df["draw_date"].iloc[-1]),
                freq_min=int(freq.min()), freq_max=int(freq.max()), chi2=chi2, p_uniform=p)


# =============================================================================
# [LAYER 2] SCORE — 8종 통계 스코어 (모두 0~1로 정규화 → 가중 합산)
#   중요: 히스토리 'hist'는 반드시 예측 시점 이전 데이터만 (walk-forward 누수 방지)
# =============================================================================
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.full_like(x, 0.5)

def appear_matrix(hist: np.ndarray) -> np.ndarray:
    """(회차 × 45) 0/1 출현 행렬 — 벡터화 핵심 자료구조."""
    m = np.zeros((len(hist), N_BALL), dtype=np.int8)
    m[np.arange(len(hist))[:, None], hist - 1] = 1
    return m

def score_frequency(A):        # 1 전체 빈도
    return _norm(A.sum(0))

def score_recency(A, n=50):    # 2 최근 N회 빈도(지수 가중: 최근일수록 크게)
    w = np.exp(np.linspace(-2, 0, min(n, len(A))))
    return _norm((A[-len(w):] * w[:, None]).sum(0))

def score_gap(A):              # 3 갭 — 마지막 출현 후 경과 회차(길수록 ↑, 평균회귀 가정)
    last = np.array([np.max(np.where(A[:, b])[0]) if A[:, b].any() else -1 for b in range(N_BALL)])
    return _norm(len(A) - 1 - last)

def score_zscore(A):           # 4 Z-Score — 기대 출현수 대비 편차(음수=콜드). 콜드 우선이면 부호 반전
    f = A.sum(0); mu = f.mean(); sd = f.std() or 1
    return _norm(-(f - mu) / sd)   # 콜드번호 재평가 방향(시그모이드 아이디어 계승)

def score_momentum(A, short=20, long=100):   # 5 모멘텀 — 단기 출현율 / 장기 출현율
    s = A[-short:].mean(0); l = A[-long:].mean(0) + 1e-9
    return _norm(s / l)

def score_pair(A, hist):       # 6 쌍빈도 — 다른 번호와의 동시출현 합
    P = A.T @ A; np.fill_diagonal(P, 0)
    return _norm(P.sum(1))

def score_markov(hist):        # 7 마르코프 — 직전 회차 번호 → 다음 회차 번호 전이확률(라플라스 평활)
    T = np.ones((N_BALL, N_BALL))          # α=1 평활
    for prev, nxt in zip(hist[:-1], hist[1:]):
        for a in prev:
            T[a - 1, nxt - 1] += 1
    T /= T.sum(1, keepdims=True)
    return _norm(T[hist[-1] - 1].sum(0))   # 마지막 회차 조건부 확률 합

def score_lift(A):             # 8 Lift 연관규칙 — P(A∩B)/(P(A)P(B)) 연속 점수 합
    n = len(A); p = A.mean(0)
    joint = (A.T @ A) / n
    lift = joint / (np.outer(p, p) + 1e-12); np.fill_diagonal(lift, 0)
    return _norm(lift.sum(1))

SCORE_WEIGHTS = dict(frequency=1.0, recency=1.2, gap=1.0, zscore=0.8,
                     momentum=1.0, pair=0.8, markov=1.0, lift=0.8)

EXTRA_WEIGHTS = dict(ml=1.5, network=0.8)   # Step 2: ML은 다른 점수의 정보를 종합하므로 가중 ↑

def composite_scores(hist: np.ndarray, weights=SCORE_WEIGHTS, extra: dict | None = None) -> tuple[np.ndarray, dict]:
    """8종 통계 + (선택) extra={'ml':..., 'network':...} 를 가중 평균 → 45볼 종합점수."""
    A = appear_matrix(hist)
    parts = dict(frequency=score_frequency(A), recency=score_recency(A), gap=score_gap(A),
                 zscore=score_zscore(A), momentum=score_momentum(A), pair=score_pair(A, hist),
                 markov=score_markov(hist), lift=score_lift(A))
    w = dict(weights)
    for k, v in (extra or {}).items():
        parts[k] = v; w[k] = EXTRA_WEIGHTS.get(k, 1.0)
    total = sum(w[k] * v for k, v in parts.items()) / sum(w.values())
    return total, parts


# =============================================================================
# [LAYER 3] POPULARITY — 인기도 역산 감점 (Phase 3 quasi-Poisson 계수 계승)
#   유의 피처: sum(-4.9%/SD), same-row-max(-7.4%/SD), tail-digit-repeat(+5.9%/SD),
#              prev-overlap(-1.4%/SD). 값이 클수록 "대중이 많이 고른 조합" → 회피 대상.
#   + 규칙 감점: 생일 범위(1~31) 집중, 행운수(3,7,13,21,27), 33 이상 부재
# =============================================================================
POP_COEF = dict(sum=-0.049, same_row_max=-0.074, tail_repeat=+0.059, prev_overlap=-0.014)
LUCKY = {3, 7, 13, 21, 27}

def combo_features(c: tuple, prev: np.ndarray) -> dict:
    c = sorted(c)
    rows = Counter((n - 1) // 7 for n in c)                 # OMR 용지 1행 = 7개 번호
    tails = Counter(n % 10 for n in c)
    return dict(sum=sum(c), same_row_max=max(rows.values()),
                tail_repeat=sum(v - 1 for v in tails.values() if v > 1),
                prev_overlap=len(set(c) & set(prev.tolist())))

# 피처 표준화용 기준(무작위 조합 이론값 근사): mean, sd
POP_STD = dict(sum=(138, 32), same_row_max=(2.1, 0.75), tail_repeat=(1.2, 0.9), prev_overlap=(0.8, 0.75))

def popularity_index(c: tuple, prev: np.ndarray) -> float:
    """상대 인기도(로그 스케일). 0 = 평균, 양수 = 인기(회피), 음수 = 비인기(선호)."""
    f = combo_features(c, prev)
    z = sum(POP_COEF[k] * (f[k] - POP_STD[k][0]) / POP_STD[k][1] for k in POP_COEF)
    z += 0.03 * sum(1 for n in c if n <= 31) - 0.03 * 3.9   # 생일범위 편향(기준 3.9개)
    z += 0.04 * sum(1 for n in c if n in LUCKY)
    return float(z)


# =============================================================================
# [LAYER 4] FILTER — 8종 조합 필터 (실제 당첨 조합의 대다수를 통과시키는 완화 기준)
# =============================================================================
def ac_value(c) -> int:
    diffs = {abs(a - b) for a, b in itertools.combinations(c, 2)}
    return len(diffs) - (N_PICK - 1)

def passes_filters(c: tuple) -> tuple[bool, str]:
    c = sorted(c)
    s = sum(c)
    if not 100 <= s <= 175:                          return False, "sum"      # 1 합계
    if ac_value(c) < 7:                              return False, "ac"       # 2 AC값
    odd = sum(n % 2 for n in c)
    if odd in (0, 1, 5, 6):                          return False, "oddeven"  # 3 홀짝 2:4~4:2
    low = sum(n <= 22 for n in c)
    if low in (0, 1, 5, 6):                          return False, "lowhigh"  # 4 저고
    runs = sum(1 for a, b in zip(c, c[1:]) if b == a + 1)
    if runs >= 3 or any(c[i+2] == c[i] + 2 for i in range(4)):
                                                     return False, "consec"   # 5 3연속 금지
    if max(Counter(n % 10 for n in c).values()) >= 3: return False, "tail"    # 6 끝수 3개 이상
    if len({(n - 1) // 10 for n in c}) < 3:          return False, "decade"   # 7 번호대 3구간+
    # 8 인기 패턴 하드 제외: 등차수열, 단일행 4개+, 모두 ≤31
    d = [b - a for a, b in zip(c, c[1:])]
    if len(set(d)) == 1:                             return False, "arith"
    if max(Counter((n - 1) // 7 for n in c).values()) >= 4: return False, "row"
    if all(n <= 31 for n in c):                      return False, "birthday"
    return True, ""


# =============================================================================
# [LAYER 5] GENERATION — 진정 난수(secrets, OS CSPRNG) + 점수 가중 샘플링 + KMeans 다양성
#   secrets 모듈은 OS 엔트로피 풀(/dev/urandom, CryptGenRandom)을 사용 → 유사난수 패턴 제거.
#   모듈로 바이어스 제거: 거부 샘플링(rejection sampling).
# =============================================================================
_sysrand = secrets.SystemRandom()

def weighted_pick(scores: np.ndarray, k: int = N_PICK, temperature: float = 0.6) -> tuple:
    """점수를 softmax 확률로 바꿔 비복원 가중 샘플(진정 난수 기반)."""
    p = np.exp((scores - scores.mean()) / max(temperature, 1e-6))
    p = p / p.sum()
    chosen = []
    avail = list(range(N_BALL)); pw = p.copy()
    for _ in range(k):
        pw_a = pw[avail] / pw[avail].sum()
        r = _sysrand.random(); cum = 0.0
        for idx, prob in zip(avail, pw_a):
            cum += prob
            if r <= cum:
                chosen.append(idx + 1); avail.remove(idx); break
        else:
            chosen.append(avail[-1] + 1); avail.pop()
    return tuple(sorted(chosen))

def generate_sets(scores: np.ndarray, prev: np.ndarray, n_sets: int = 10, pool: int = 3000,
                  pop_penalty: float = 1.0, extra_cands: list | None = None, labels=None) -> list[dict]:
    """
    1) 후보 pool개 생성 → 2) 필터 통과 → 3) 조합점수 = 평균 볼점수 − pop_penalty×인기도
    4) KMeans(k=n_sets)로 클러스터별 최고점 1개씩 → 다양성 보장
    """
    from sklearn.cluster import KMeans
    cands, seen = [], set()
    src = [weighted_pick(scores) for _ in range(pool)] + list(extra_cands or [])   # 샘플링 + GA 후보 합류
    for c in src:
        if c in seen: continue
        ok, why = passes_filters(c)
        if not ok: continue
        seen.add(c)
        pop = popularity_index(c, prev)
        ball = float(scores[np.array(c) - 1].mean())
        comm = len({labels[n - 1] for n in c}) if labels is not None else 0     # 커뮤니티 다양성
        cands.append(dict(combo=c, ball=ball, pop=pop, comm=comm,
                          total=ball - pop_penalty * pop + 0.02 * comm))
    if len(cands) < n_sets:
        return sorted(cands, key=lambda d: -d["total"])
    X = np.array([[1 if n in d["combo"] else 0 for n in range(1, 46)] for d in cands], float)
    labels = KMeans(n_clusters=n_sets, n_init=5, random_state=_sysrand.randrange(10**6)).fit_predict(X)
    picked = []
    for lab in range(n_sets):
        grp = [d for d, l in zip(cands, labels) if l == lab]
        if grp: picked.append(max(grp, key=lambda d: d["total"]))
    return sorted(picked, key=lambda d: -d["total"])


# =============================================================================
# [LAYER 6] VALIDATION — walk-forward 백테스트 + 랜덤 대조군 + 부트스트랩 CI
#   측정치: 추천 세트당 평균 일치 개수. 무작위 기대치 = 6×6/45 = 0.8개.
#   모델 CI가 대조군 CI와 겹치면 "무작위와 통계적 동등" → 정직한 결과.
# =============================================================================
def random_combo() -> tuple:
    return tuple(sorted(_sysrand.sample(range(1, 46), 6)))

def backtest(df: pd.DataFrame, n_rounds: int = 100, sets_per_round: int = 5, control: int = 100,
             use_ml: bool = False, use_ga: bool = False, pop_penalty: float = 1.0, retrain_every: int = 10):
    hist_all = df[[f"num{i}" for i in range(1, 7)]].to_numpy()
    model_hits, ctrl_hits = [], []
    start = len(hist_all) - n_rounds
    models = None
    for t in range(start, len(hist_all)):
        hist = hist_all[:t]                      # ← t회차 이전 데이터만 사용(누수 없음)
        actual = set(hist_all[t].tolist())
        extra, ga_c = {}, None
        if use_ml or use_ga:
            import lotto_ultimate_ml as X
            A = X.appear_matrix(hist)
            if use_ml:
                if models is None or (t - start) % retrain_every == 0:
                    models = X.train_ml(A)           # 재학습 시점에도 hist(과거)만 사용
                extra["ml"], _ = X.score_ml(A, models)
        scores, _ = composite_scores(hist, extra=extra)
        if use_ga:
            ga_c, _ = X.genetic_optimize(scores, hist[-1], popularity_index, passes_filters,
                                         pop_size=120, generations=25, lam=pop_penalty, n_out=20)
        sets = generate_sets(scores, hist[-1], n_sets=sets_per_round, pool=600,
                             pop_penalty=pop_penalty, extra_cands=ga_c)
        model_hits.append(np.mean([len(actual & set(d["combo"])) for d in sets]))
        ctrl_hits.append(np.mean([len(actual & set(random_combo())) for _ in range(control)]))
    m, c = np.array(model_hits), np.array(ctrl_hits)
    def boot_ci(x, B=2000):
        idx = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
        means = x[idx].mean(1); return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    from scipy.stats import ttest_rel
    t, p = ttest_rel(m, c)
    return dict(rounds=n_rounds, model_mean=float(m.mean()), model_ci=boot_ci(m),
                control_mean=float(c.mean()), control_ci=boot_ci(c), theory=6 * 6 / 45,
                paired_t=float(t), p_value=float(p),
                match3plus_rate=float(np.mean(m >= 3)))


# =============================================================================
# [LAYER 7] PREDICTION TRACKER — 사전 등록 → 사후 채점 (prediction_log.csv)
# =============================================================================
def register_predictions(sets: list[dict], target_round: int, path="prediction_log.csv") -> bool:
    """회차당 1회만 등록(중복 실행 방어) — 이미 등록된 회차면 False 반환."""
    import os
    new = not os.path.exists(path)
    if not new:
        try:
            if (pd.read_csv(path)["target_round"] == target_round).any():
                print(f"📋 제{target_round}회 예측은 이미 등록됨 — 기존 사전등록 유지(사후채점 무결성)")
                return False
        except Exception:
            pass
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["target_round", "set_id", "n1", "n2", "n3", "n4", "n5", "n6", "pop_index", "hits", "scored"])
        for i, d in enumerate(sets, 1):
            w.writerow([target_round, i, *d["combo"], round(d["pop"], 4), "", 0])
    return True

def score_predictions(df: pd.DataFrame, path="prediction_log.csv"):
    import os
    if not os.path.exists(path): return None
    log = pd.read_csv(path)
    actual = {int(r["round"]): set(int(r[f"num{i}"]) for i in range(1, 7)) for _, r in df.iterrows()}
    for i, r in log.iterrows():
        if r.scored == 0 and int(r.target_round) in actual:
            log.at[i, "hits"] = len(actual[int(r.target_round)] & {int(r[f"n{j}"]) for j in range(1, 7)})
            log.at[i, "scored"] = 1
    log.to_csv(path, index=False)
    done = log[log.scored == 1]
    recent = done.tail(30)
    return dict(n=int(len(done)), mean_hits=float(done.hits.mean()) if len(done) else None,
                match3plus=int((done.hits >= 3).sum()) if len(done) else 0,
                recent=[dict(round=int(r.target_round), nums=[int(r[f"n{j}"]) for j in range(1, 7)], hits=int(r.hits))
                        for _, r in recent.iterrows()])


# =============================================================================
# MAIN
# =============================================================================
def nxt_round(df): return int(df["round"].iloc[-1]) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--sets", type=int, default=10)
    ap.add_argument("--backtest", type=int, default=0, help="백테스트 라운드 수(0=생략)")
    ap.add_argument("--ml", action="store_true", help="RF+GB 볼 확률 점수 편입")
    ap.add_argument("--ga", action="store_true", help="유전 알고리즘 조합 최적화")
    ap.add_argument("--net", action="store_true", help="네트워크 커뮤니티 분석")
    ap.add_argument("--tests", action="store_true", help="15종 무작위성 검정 배터리")
    ap.add_argument("--pop-penalty", type=float, default=1.0, help="인기도 감점 강도 λ")
    ap.add_argument("--swarm", action="store_true", help="군집지능 PSO+ACO+점균류 후보 합류")
    ap.add_argument("--seed", default="none", help="진정난수 시드: auto|qrng|random_org|usgs|os|none")
    ap.add_argument("--vrf", type=int, default=0, help="VRF 검증가능 조합 n세트 추가")
    ap.add_argument("--out", default=".", help="출력 폴더 (prediction_log.csv, latest.json)")
    ap.add_argument("--json", action="store_true", help="results JSON 생성 (GitHub Pages용)")
    a = ap.parse_args()
    import os, json, datetime
    os.makedirs(a.out, exist_ok=True)
    LOG = os.path.join(a.out, "prediction_log.csv")
    R = dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             options=vars(a))  # ← JSON 리포트 누적 객체

    df = load_lotto_csv(a.csv)
    au = data_audit(df); R["audit"] = au
    print("=" * 64); print("📊 DATA AUDIT")
    print(f"  회차 {au['first']}~{au['last']} ({au['rounds']}회) | {au['first_date']} ~ {au['last_date']}")
    print(f"  번호별 출현 {au['freq_min']}~{au['freq_max']}회 | 균등성 χ²={au['chi2']:.1f}, p={au['p_uniform']:.3f}"
          + ("  → 무작위와 부합" if au['p_uniform'] > 0.05 else "  → 편향 의심"))

    hist = df[[f"num{i}" for i in range(1, 7)]].to_numpy()
    extra, labels, ga_c = {}, None, None
    if a.ml or a.net or a.ga or a.tests:
        import lotto_ultimate_ml as X
        A = X.appear_matrix(hist)
    if a.tests:
        print("\n🧪 무작위성 검정 배터리 15종 (Holm-Bonferroni 보정)")
        bat = X.randomness_battery(hist); R["tests"] = bat
        for r in bat:
            flag = "★보정후 유의" if r["sig_adj"] else ("·원p<.05" if r["sig_raw"] else "")
            print(f"  {r['name']:<10s} p={r['p']:.3f}  p_adj={r['p_adj']:.3f}  {flag:<12s} {r['desc']}")
        n_raw = sum(r["sig_raw"] for r in bat); n_adj = sum(r["sig_adj"] for r in bat)
        print(f"  → 원 p<0.05: {n_raw}/15, 보정 후 유의: {n_adj}/15  "
              + ("(무작위와 통계적 동등)" if n_adj == 0 else "(추가 조사 필요)"))
    if a.ml:
        extra["ml"], raw_p = X.score_ml(A); R["ml_prob"] = [round(float(p), 4) for p in raw_p]
        print(f"\n🧠 ML(RF+GB) 다음회차 출현확률: 평균 {raw_p.mean():.3f} (이론 0.133) | "
              f"최고 {raw_p.max():.3f}(#{raw_p.argmax()+1}) 최저 {raw_p.min():.3f}(#{raw_p.argmin()+1})")
    if a.net:
        extra["network"], labels, _ = X.network_analysis(A); R["communities"] = [int(l) for l in labels]
        print("\n🕸️ 네트워크 커뮤니티 8개:")
        for k in range(8):
            print(f"  C{k}: {[int(v) for v in np.where(labels == k)[0] + 1]}")
    scores, parts = composite_scores(hist, extra=extra)
    top = np.argsort(-scores)[:12] + 1
    R["ball_scores"] = [round(float(s), 4) for s in scores]
    R["score_parts"] = {k: [round(float(x), 4) for x in v] for k, v in parts.items()}
    print("\n🔢 45볼 종합점수 상위 12:", ", ".join(f"{n}({scores[n-1]:.2f})" for n in top))
    if a.ga:
        ga_c, hist_ga = X.genetic_optimize(scores, hist[-1], popularity_index, passes_filters,
                                           labels=labels, lam=a.pop_penalty)
        g0, gN = hist_ga[0], hist_ga[-1]
        print(f"\n🧬 유전 알고리즘: {len(hist_ga)}세대, 최고적합도 {g0[1]:.3f}→{gN[1]:.3f}, "
              f"돌연변이율 {g0[3]:.2f}→{gN[3]:.2f}, 다양성 {gN[2]:.2f}, 후보 {len(ga_c)}개")

    seed_bytes, seed_src, vrf_c = None, None, []
    if a.seed != "none" or a.vrf:
        import lotto_ultimate_swarm as S
        seed_bytes, seed_src = S.get_true_seed(a.seed if a.seed != "none" else "os")
        R["seed"] = dict(source=seed_src, fingerprint=seed_bytes.hex()[:16], commit=S.vrf_commit(seed_bytes, nxt_round(df)))
        print(f"\n🔮 진정난수 시드: 출처={seed_src}  지문={seed_bytes.hex()[:16]}…  "
              f"커밋={S.vrf_commit(seed_bytes, nxt_round(df))}")
        if seed_src == "os_csprng" and a.seed not in ("os", "none"):
            print("   (외부 소스 접근 실패 → OS CSPRNG 폴백. 네트워크/API 상태를 확인하세요)")
        # 시드로 가중 샘플러도 재시딩 → 이후 모든 무작위 선택이 진정난수에서 파생
        global _sysrand; _sysrand = __import__("random").Random(int.from_bytes(seed_bytes, "big"))
    if a.swarm:
        import lotto_ultimate_swarm as S
        cooc = (appear_matrix(hist).T @ appear_matrix(hist)).astype(float); np.fill_diagonal(cooc, 0)
        fit = S.make_fitness(scores, hist[-1], popularity_index, passes_filters, lam=a.pop_penalty, labels=labels)
        pso_c, pso_f = S.pso_optimize(scores, fit)
        aco_c, aco_f = S.aco_optimize(scores, fit, cooc)
        sl_c, sl_f, strength = S.slime_optimize(scores, fit, cooc)
        print(f"\n🐝 군집지능: PSO 최고적합도 {pso_f:.3f}({len(pso_c)}개) | ACO {aco_f:.3f}({len(aco_c)}개) | 점균류 {sl_f:.3f}({len(sl_c)}개)")
        print(f"   점균 굵은 노드 상위 10: {(np.argsort(-strength)[:10]+1).tolist()}")
        lists = dict(PSO=pso_c, ACO=aco_c, SLIME=sl_c)
        if ga_c: lists["GA"] = ga_c
        vote, agreed = S.consensus(lists)
        R["swarm"] = dict(pso_best=pso_f, aco_best=aco_f, slime_best=sl_f, consensus_top=(np.argsort(-vote)[:10]+1).tolist(),
                          slime_hubs=(np.argsort(-strength)[:10]+1).tolist(), agreed=[[list(c), n] for _, c, n in agreed[:5]])
        print(f"   탐색기 합의 상위 10볼: {(np.argsort(-vote)[:10]+1).tolist()}")
        if agreed:
            print(f"   ≥2개 탐색기 동일 조합 {len(agreed)}개, 예: {agreed[0][1]} ← {agreed[0][2]}")
        ga_c = (ga_c or []) + pso_c + aco_c + sl_c
    if a.vrf:
        vrf_c = S.vrf_sets(seed_bytes, nxt_round(df), a.vrf, passes_filters)

    sets = generate_sets(scores, hist[-1], n_sets=a.sets, pop_penalty=a.pop_penalty,
                         extra_cands=ga_c, labels=labels)
    for i, c in enumerate(vrf_c):     # VRF 세트는 점수 무관 — 편향 제거 목적의 별도 트랙
        sets.append(dict(combo=c, ball=float(scores[np.array(c)-1].mean()), pop=popularity_index(c, hist[-1]),
                         comm=0, total=0.0, vrf=True))
    nxt = au["last"] + 1
    print(f"\n🎯 제{nxt}회 추천 {len(sets)}세트 (필터 통과 · 인기도 회피 · KMeans 다양성)")
    for i, d in enumerate(sets, 1):
        tag = "비인기✓" if d["pop"] < 0 else "보통"
        cm = f"  커뮤니티={d['comm']}" if d.get("comm") else ("  [VRF 검증가능]" if d.get("vrf") else "")
        print(f"  #{i:2d}  {' '.join(f'{n:2d}' for n in d['combo'])}   합={sum(d['combo']):3d}  "
              f"볼점수={d['ball']:.3f}  인기도={d['pop']:+.3f} {tag}{cm}")
    R["target_round"] = nxt
    R["sets"] = [dict(nums=list(d["combo"]), sum=int(sum(d["combo"])), ball=round(d["ball"], 4), pop=round(d["pop"], 4),
                      comm=int(d.get("comm", 0)), vrf=bool(d.get("vrf", False))) for d in sets]
    register_predictions(sets, nxt, path=LOG)
    sc = score_predictions(df, path=LOG); R["tracker"] = sc
    if sc and sc["n"]: print(f"\n📋 누적 사후채점: {sc['n']}세트, 평균 일치 {sc['mean_hits']:.3f}개 (무작위 기대 0.800)")

    if a.backtest:
        print(f"\n🛡️ WALK-FORWARD 백테스트 (최근 {a.backtest}회차, 라운드당 대조군 100세트)")
        r = backtest(df, n_rounds=a.backtest, use_ml=a.ml, use_ga=a.ga, pop_penalty=a.pop_penalty); R["backtest"] = r
        print(f"  모델   평균일치 {r['model_mean']:.3f}  95%CI [{r['model_ci'][0]:.3f}, {r['model_ci'][1]:.3f}]")
        print(f"  대조군 평균일치 {r['control_mean']:.3f}  95%CI [{r['control_ci'][0]:.3f}, {r['control_ci'][1]:.3f}]")
        print(f"  이론값 {r['theory']:.3f} | 대응 t={r['paired_t']:.2f}, p={r['p_value']:.3f}")
        verdict = "무작위와 통계적 동등(정직 작동)" if r["p_value"] > 0.05 else "차이 감지 — 다중비교 보정 필요"
        print(f"  판정: {verdict}")
    if a.json:
        jp = os.path.join(a.out, "latest.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(R, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        # 회차별 이력도 보존(웹에서 과거 추천 열람용)
        with open(os.path.join(a.out, f"round_{nxt}.json"), "w", encoding="utf-8") as f:
            json.dump(R, f, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"\n💾 JSON 저장: {jp}")
    print("\n⚠ 면책: 로또는 완전 무작위입니다. 본 도구는 기대 배당금 최적화·교육 목적이며 당첨을 보장하지 않습니다.")
    print("   도박 문제 상담: 1336 (한국도박문제예방치유원)")

if __name__ == "__main__":
    main()
