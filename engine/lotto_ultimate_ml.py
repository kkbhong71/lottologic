#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOTTO ULTIMATE — Step 2 확장 모듈
  [ML]   RandomForest + GradientBoosting 볼별 다음회차 출현확률 (9번째 점수)
  [NET]  동반출현 네트워크 → 스펙트럴 커뮤니티 8개 + 허브 점수 (10번째 점수)
  [GA]   유전 알고리즘 조합 최적화 (적응형 돌연변이율)
  [TEST] 15종 무작위성 검정 배터리 + Holm-Bonferroni 보정

모든 함수는 lotto_ultimate.py(코어)에서 import하여 사용한다.
"""
from __future__ import annotations
import math, secrets, itertools
from collections import Counter
import numpy as np
from scipy import stats

N_BALL, N_PICK = 45, 6
_rng = secrets.SystemRandom()

def _norm(x):
    x = np.asarray(x, float); r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.full_like(x, 0.5)

def appear_matrix(hist):
    m = np.zeros((len(hist), N_BALL), dtype=np.int8)
    m[np.arange(len(hist))[:, None], hist - 1] = 1
    return m


# =============================================================================
# [ML] 구조 ML — 각 (회차 t, 번호 b)를 하나의 표본으로 만들어 "t+1에 b가 나오는가"를 학습
#   피처(7): 최근10/30/100회 출현율, 갭, 모멘텀(10/100), 직전회차 출현, 번호 자체(위치정보)
#   주의: 표본은 t 시점 이전 정보만 사용 → walk-forward에서도 누수 없음
#   해석: 진짜 무작위라면 모든 확률이 ≈6/45=0.133 근처로 수렴한다. 편차가 크면 그것이 "패턴 후보".
# =============================================================================
def _ball_features(A: np.ndarray, t: int) -> np.ndarray:
    """A[:t]까지의 정보로 45개 번호의 피처 행렬(45×7) 생성."""
    H = A[:t]
    f10, f30, f100 = H[-10:].mean(0), H[-30:].mean(0), H[-100:].mean(0)
    last = np.array([np.max(np.where(H[:, b])[0]) if H[:, b].any() else -1 for b in range(N_BALL)])
    gap = (t - 1 - last) / 45.0
    mom = f10 / (f100 + 1e-9)
    prev = H[-1].astype(float)
    ball = np.arange(1, N_BALL + 1) / 45.0
    return np.column_stack([f10, f30, f100, gap, mom, prev, ball])

def train_ml(A: np.ndarray, train_rounds: int = 400, seed: int | None = None):
    """직전 train_rounds회를 학습표본으로 RF+GB 학습 → (rf, gb) 반환."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    T = len(A)
    Xs, ys = [], []
    for t in range(max(101, T - train_rounds), T):        # 라벨은 A[t] (t회차 출현), 피처는 A[:t]
        Xs.append(_ball_features(A, t)); ys.append(A[t])
    X, y = np.vstack(Xs), np.concatenate(ys)
    seed = seed if seed is not None else _rng.randrange(10**6)
    rf = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=20,
                                n_jobs=-1, random_state=seed).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=seed).fit(X, y)
    return rf, gb

def score_ml(A: np.ndarray, models=None) -> np.ndarray:
    """다음 회차 45볼 출현확률(RF·GB 평균) → 0~1 정규화 점수."""
    rf, gb = models if models else train_ml(A)
    X = _ball_features(A, len(A))
    p = 0.5 * rf.predict_proba(X)[:, 1] + 0.5 * gb.predict_proba(X)[:, 1]
    return _norm(p), p


# =============================================================================
# [ML-ADV] XGBoost + 간이 시퀀스 모델 (LSTM 대용) — 교육용 시연
#   목적: "더 강력한 모델을 써도 독립 추첨에서는 이론값(0.133)으로 수렴한다"를 데이터로 보여줌
#   XGBoost: 같은 피처 7개에 gradient boosted trees → RF+GB와 비교
#   시퀀스 모델: 직전 10회차의 출현 패턴(10×45)을 입력 → 시그모이드 1층 (LSTM 대용, 순수 numpy)
#     실제 LSTM은 tensorflow/pytorch가 필요하지만, 교육 목적에서는 "시퀀스를 입력으로 받는 학습 모델"이
#     무작위 데이터에서 어떤 성능을 보이는지를 보여주는 것이 핵심.
# =============================================================================
def train_xgb(A: np.ndarray, train_rounds: int = 400, seed: int | None = None):
    """XGBoost 분류기 학습 — RF+GB와 동일 피처, 동일 표본."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        try:
            from lightgbm import LGBMClassifier as XGBClassifier
        except ImportError:
            print("  ⚠ xgboost/lightgbm 미설치 — RF+GB 결과를 대용합니다")
            return None
    T = len(A); Xs, ys = [], []
    for t in range(max(101, T - train_rounds), T):
        Xs.append(_ball_features(A, t)); ys.append(A[t])
    X, y = np.vstack(Xs), np.concatenate(ys)
    seed = seed if seed is not None else _rng.randrange(10**6)
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.8, use_label_encoder=False, eval_metric='logloss',
                          verbosity=0, random_state=seed)
    model.fit(X, y)
    return model

def score_xgb(A: np.ndarray, model=None):
    """XGBoost 45볼 출현확률."""
    if model is None: model = train_xgb(A)
    if model is None: return None, None
    X = _ball_features(A, len(A))
    p = model.predict_proba(X)[:, 1]
    return _norm(p), p


def _seq_features(A: np.ndarray, t: int, window: int = 10) -> np.ndarray:
    """직전 window회차의 출현 패턴을 1차원으로 펼침 (window×45 → 450차원)."""
    start = max(0, t - window)
    seg = A[start:t].astype(float)
    if len(seg) < window:
        seg = np.vstack([np.zeros((window - len(seg), N_BALL)), seg])
    return seg.ravel()

def train_sequence_model(A: np.ndarray, train_rounds: int = 400, window: int = 10, seed: int | None = None):
    """간이 시퀀스 모델 (1층 시그모이드, LSTM 대용) — 순수 numpy.
    W(450×1), b(1) 를 SGD로 학습. 이진 교차엔트로피 손실.
    LSTM과 다른 점: 게이트 구조 없음, 순환 없음(단순 윈도우). 교육 목적의 핵심은 동일:
    "시퀀스 입력 → 학습 → 예측"이 무작위에서 작동하는지를 보여줌."""
    rng = np.random.default_rng(seed or _rng.randrange(10**6))
    T = len(A); dim = window * N_BALL
    W = rng.normal(0, 0.01, (dim, 1)).astype(float)
    b = np.float64(0.0)
    lr = 0.001
    for epoch in range(3):
        for t in range(max(window + 1, T - train_rounds), T):
            x = _seq_features(A, t, window).reshape(1, -1)  # (1, 450)
            z_val = float((x @ W)[0, 0] + b)
            z_val = np.clip(z_val, -20, 20)
            p = 1 / (1 + np.exp(-z_val))
            y_avg = A[t].mean()  # 평균 출현(6/45)
            grad = (p - y_avg)
            W -= lr * grad * x.T
            b -= lr * grad
    return W, b, window

def score_sequence(A: np.ndarray, model=None):
    """간이 시퀀스 모델 45볼 출현확률."""
    if model is None: model = train_sequence_model(A)
    W, b_val, window = model
    b_val = float(np.asarray(b_val).ravel()[0])
    probs = np.zeros(N_BALL)
    base_x = _seq_features(A, len(A), window).reshape(1, -1)
    for ball in range(N_BALL):
        x_b = base_x.copy()
        for w in range(window):
            x_b[0, w * N_BALL + ball] *= 2.0
        z = np.clip(float((x_b @ W)[0, 0]) + b_val, -20, 20)
        probs[ball] = 1 / (1 + np.exp(-z))
    return _norm(probs), probs


def ml_comparison(A: np.ndarray, n_test: int = 50):
    """RF+GB vs XGBoost vs 시퀀스 모델 walk-forward 비교 — 교육용 시연.
    각 모델이 n_test 회차에 대해 "다음 회차 출현 번호를 얼마나 높은 확률로 잡았는가"를 측정.
    측정치: 출현 번호 6개의 평균 예측 확률 (높을수록 좋음, 이론값 0.133).
    """
    T = len(A)
    results = {name: [] for name in ['RF+GB', 'XGBoost', 'Sequence', 'Random']}
    rf_gb = train_ml(A[:T - n_test])
    xgb = train_xgb(A[:T - n_test])
    seq = train_sequence_model(A[:T - n_test])

    for t in range(T - n_test, T):
        actual = np.where(A[t])[0]  # 출현 번호 인덱스
        X = _ball_features(A[:t], t)

        # RF+GB
        p_rf = 0.5 * rf_gb[0].predict_proba(X)[:, 1] + 0.5 * rf_gb[1].predict_proba(X)[:, 1]
        results['RF+GB'].append(float(p_rf[actual].mean()))

        # XGBoost
        if xgb is not None:
            p_xgb = xgb.predict_proba(X)[:, 1]
            results['XGBoost'].append(float(p_xgb[actual].mean()))
        else:
            results['XGBoost'].append(float(p_rf[actual].mean()))

        # Sequence
        _, p_seq = score_sequence(A[:t], seq)
        results['Sequence'].append(float(p_seq[actual].mean()))

        # Random baseline
        results['Random'].append(6 / 45)

    summary = {}
    for name, vals in results.items():
        arr = np.array(vals)
        summary[name] = {
            'mean': round(float(arr.mean()), 4),
            'std': round(float(arr.std()), 4),
            'vs_theory': round(float(arr.mean() - 6/45), 4),
        }
    return summary, results


# =============================================================================
# [NET] 네트워크 분석 — 동반출현 가중 그래프
#   허브 점수: 가중 차수(다른 번호들과 함께 나온 총 횟수)를 기대치로 나눈 초과분
#   커뮤니티: 스펙트럴 클러스터링 k=8 (v2.1 계승). 조합 생성 시 커뮤니티 다양성 제약에 사용.
# =============================================================================
def network_analysis(A: np.ndarray, k: int = 8, seed: int = 0):
    from sklearn.cluster import SpectralClustering
    W = (A.T @ A).astype(float); np.fill_diagonal(W, 0)
    expected = W.sum() / (N_BALL * (N_BALL - 1))
    hub = W.sum(1) / (expected * (N_BALL - 1))             # 1.0 = 평균
    labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                                assign_labels="discretize").fit_predict(W + 1e-9)
    return _norm(hub), labels, W

def community_spread(combo, labels) -> int:
    """조합이 걸치는 커뮤니티 수 (클수록 다양)."""
    return len({labels[n - 1] for n in combo})


# =============================================================================
# [GA] 유전 알고리즘 — 조합 단위 최적화
#   적합도 = 평균 볼점수 − λ·인기도 + μ·커뮤니티다양성 + 엔트로피보너스 − 필터위반 페널티
#   적응형 돌연변이: 개체군 다양성(고유 조합 비율)이 떨어지면 돌연변이율 ↑, 높으면 ↓
#   유전자 = 6개 번호 집합. 교차 = 두 부모의 합집합에서 6개 비복원 추출(진정 난수).
# =============================================================================
def entropy_bonus(combo) -> float:
    """번호 간격 분포의 Shannon 엔트로피(간격이 고르면 낮고, 다양하면 높음) — 단순 패턴 탈락용."""
    c = sorted(combo); gaps = np.diff(c)
    p = np.bincount(gaps)[gaps.min():].astype(float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum()) / math.log2(5)

def genetic_optimize(scores, prev, pop_fn, filter_fn, labels=None,
                     pop_size=300, generations=60, lam=1.0, mu=0.05, n_out=40, verbose=False):
    def fitness(c):
        ok, _ = filter_fn(c)
        f = scores[np.array(c) - 1].mean() - lam * pop_fn(c, prev)
        f += 0.05 * entropy_bonus(c)
        if labels is not None: f += mu * community_spread(c, labels)
        return f - (0.5 if not ok else 0.0)

    pool = [tuple(sorted(_rng.sample(range(1, 46), 6))) for _ in range(pop_size)]
    mut_rate, history = 0.10, []
    for g in range(generations):
        fit = np.array([fitness(c) for c in pool])
        order = np.argsort(-fit); pool = [pool[i] for i in order]; fit = fit[order]
        elite = pool[:pop_size // 10]                        # 상위 10% 엘리트 보존
        diversity = len(set(pool)) / pop_size
        mut_rate = float(np.clip(mut_rate * (1.3 if diversity < 0.6 else 0.85), 0.03, 0.4))  # 적응형
        history.append((g, float(fit[0]), diversity, mut_rate))
        children = list(elite)
        while len(children) < pop_size:
            a, b = (pool[min(_rng.randrange(pop_size), _rng.randrange(pop_size))] for _ in range(2))  # 토너먼트
            genes = list(set(a) | set(b))
            child = set(_rng.sample(genes, 6))
            if _rng.random() < mut_rate:                     # 돌연변이: 1~2개 교체
                for _ in range(_rng.choice([1, 1, 2])):
                    child.discard(_rng.choice(list(child)))
                    while len(child) < 6: child.add(_rng.randrange(1, 46))
            children.append(tuple(sorted(child)))
        pool = children
    fit = np.array([fitness(c) for c in pool]); order = np.argsort(-fit)
    seen, out = set(), []
    for i in order:
        c = pool[i]
        if c in seen or not filter_fn(c)[0]: continue
        seen.add(c); out.append(c)
        if len(out) >= n_out: break
    return out, history


# =============================================================================
# [TEST] 15종 무작위성 검정 배터리
#   각 검정은 (이름, p-value, 설명) 반환. 마지막에 Holm-Bonferroni로 가족오류율 보정.
#   핵심 교훈: 15개 검정 중 하나가 p<0.05인 건 우연히도 흔하다(≈54%). 보정 후 남는 것만 신호.
# =============================================================================
def _two_sided(sims, obs) -> float:
    """시뮬레이션 귀무분포 대비 양측 p (최소 1/len, 최대 1)."""
    sims = np.asarray(sims)
    return float(min(1.0, max(2 * min(np.mean(sims <= obs), np.mean(sims >= obs)), 1 / len(sims))))

def randomness_battery(hist: np.ndarray, n_sim: int = 3000) -> list[dict]:
    A = appear_matrix(hist); n = len(hist); flat = hist.ravel()
    sums = hist.sum(1); res = []
    sim = np.array([sorted(_rng.sample(range(1, 46), 6)) for _ in range(n_sim)])  # 이론분포 근사용

    # 1 χ² 균등성
    f = np.bincount(flat, minlength=46)[1:]
    res.append(("χ² 균등성", stats.chisquare(f).pvalue, "45개 번호 출현 횟수 균등"))
    # 2 런 검정(합계의 중앙값 상하 부호 런)
    s = (sums > np.median(sums)).astype(int); runs = 1 + (np.diff(s) != 0).sum()
    n1, n0 = s.sum(), len(s) - s.sum(); mu = 2*n1*n0/(n1+n0) + 1
    var = 2*n1*n0*(2*n1*n0 - n1 - n0) / ((n1+n0)**2 * (n1+n0-1))
    res.append(("런 검정", 2*stats.norm.sf(abs(runs-mu)/math.sqrt(var)), "합계 시계열의 런 수"))
    # 3 자기상관(합계, lag1)
    r1 = np.corrcoef(sums[:-1], sums[1:])[0, 1]
    res.append(("자기상관 lag1", 2*stats.norm.sf(abs(r1)*math.sqrt(n)), "직전 회차 합계와의 상관"))
    # 4 엔트로피(번호 분포) — 시뮬레이션 대조
    def ent(x): p = np.bincount(x, minlength=46)[1:]/len(x); p = p[p>0]; return -(p*np.log2(p)).sum()
    e_obs = ent(flat); e_sim = [ent(np.array([_rng.sample(range(1,46),6) for _ in range(n)]).ravel()) for _ in range(200)]
    res.append(("엔트로피", _two_sided(e_sim, e_obs), "출현분포 Shannon 엔트로피"))
    # 5 갭 분포 KS (각 번호 재출현 간격 ~ 기하분포 p=6/45)
    gaps = []
    for b in range(N_BALL):
        idx = np.where(A[:, b])[0]; gaps += list(np.diff(idx))
    gaps = np.array(gaps); edges = [1, 2, 3, 5, 8, 12, 17, 25, 40, 10**6]        # 구간화(기대빈도≥5 보장)
    obs_g = np.array([((gaps >= lo) & (gaps < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])])
    G = stats.geom(6/45); exp_g = np.array([(G.cdf(hi-1) - G.cdf(lo-1)) for lo, hi in zip(edges[:-1], edges[1:])]) * len(gaps)
    res.append(("갭 분포 χ²", stats.chisquare(obs_g, exp_g * obs_g.sum()/exp_g.sum()).pvalue, "재출현 간격 ~ 기하분포"))
    # 6 쌍빈도 χ²
    P = (A.T @ A)[np.triu_indices(N_BALL, 1)]
    res.append(("쌍빈도 χ²", stats.chisquare(P).pvalue, "990개 번호쌍 동시출현 균등"))
    # 7 이진행렬 순위(최근 45×45 블록)
    ranks = [np.linalg.matrix_rank(A[i:i+45].astype(float)) for i in range(0, n-45, 45)]
    rank_sim = [np.linalg.matrix_rank(appear_matrix(np.array([_rng.sample(range(1,46),6) for _ in range(45)])).astype(float)) for _ in range(300)]
    res.append(("행렬 순위", _two_sided(rank_sim, np.mean(ranks)) if len(ranks) > 2 else 1.0, "45×45 출현행렬 평균 순위(선형 종속성)"))
    # 8 LZ 복잡도(선형복잡도 대용) — 합계 부호열 압축률 vs 시뮬레이션
    def lz(seq):
        i, c, d = 0, 0, set()
        while i < len(seq):
            j = i+1
            while j <= len(seq) and seq[i:j] in d: j += 1
            d.add(seq[i:j]); c += 1; i = j
        return c
    obs = lz("".join(map(str, s))); sims = [lz("".join(str(_rng.randrange(2)) for _ in range(n))) for _ in range(200)]
    res.append(("LZ 복잡도", _two_sided(sims, obs), "합계 부호열의 압축 복잡도"))
    # 9 직렬 검정(2-gram of 홀짝 개수 4구간)
    q = np.minimum(hist.sum(1) % 2 + (hist % 2).sum(1), 6) // 2
    pairs = Counter(zip(q[:-1], q[1:])); obs_tab = np.zeros((4,4))
    for (a,b),v in pairs.items(): obs_tab[a,b]=v
    res.append(("직렬 검정", stats.chi2_contingency(obs_tab+0.5)[1], "연속 회차 상태 전이 독립성"))
    # 10 합계 분포 KS vs 시뮬레이션
    res.append(("합계 KS", stats.ks_2samp(sums, sim.sum(1)).pvalue, "6수 합 분포 = 이론 분포"))
    # 11 끝수 균등
    tail_n = np.array([sum(1 for v in range(1, 46) if v % 10 == d) for d in range(10)])   # 끝수별 번호 개수 4/5/5/5/5/5/4/4/4/4
    res.append(("끝수 균등", stats.chisquare(np.bincount(flat % 10, minlength=10), len(flat)*tail_n/45).pvalue, "0~9 끝수(구간크기 보정)"))
    # 12 구간 균등(10단위 5구간; 구간 크기 보정)
    dec = np.bincount((flat-1)//10, minlength=5); exp = len(flat)*np.array([10,10,10,10,5])/45
    res.append(("구간 균등", stats.chisquare(dec, exp).pvalue, "1-10/…/41-45 구간 출현"))
    # 13 연속쌍 빈도(회차당 연번 쌍 수)
    cons = (np.diff(hist, axis=1) == 1).sum(1); cons_sim = (np.diff(sim, axis=1)==1).sum(1)
    res.append(("연속쌍 빈도", stats.ks_2samp(cons, cons_sim).pvalue, "회차당 연번 쌍 수 분포"))
    # 14 홀수 개수 이항성
    odd = (hist % 2).sum(1); odd_sim = (sim % 2).sum(1)
    res.append(("홀짝 분포", stats.ks_2samp(odd, odd_sim).pvalue, "홀수 개수 분포"))
    # 15 정상성(ADF 대용: 합계 이동평균의 전후 반부 평균 차이 t-검정)
    res.append(("정상성", stats.ttest_ind(sums[:n//2], sums[n//2:]).pvalue, "합계 시계열 전·후반 평균 동일"))

    # Holm-Bonferroni
    pv = np.array([r[1] for r in res]); order = np.argsort(pv); m = len(pv)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pv[i] * (m - rank)); adj[i] = min(running, 1.0)
    return [dict(name=r[0], p=float(r[1]), p_adj=float(adj[i]), desc=r[2],
                 sig_raw=r[1] < 0.05, sig_adj=adj[i] < 0.05) for i, r in enumerate(res)]
