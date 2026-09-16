#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOTTO ULTIMATE — Step 3 확장 모듈: 방법론 4·5·6
  [SEED] 진정 난수 시드 — ANU QRNG(양자) / random.org(대기잡음) / USGS 지진파(지구물리) / OS CSPRNG 폴백
  [VRF]  블록체인식 검증가능 난수 — SHA-256 해시체인 + 거부샘플링(모듈로 바이어스 제거), 커밋-공개 구조
  [PSO]  입자 군집 최적화 — 45차원 연속공간 → 상위 6개 좌표를 번호로 해석
  [ACO]  개미 군집 최적화 — 45×45 페로몬 행렬 위 6노드 궤적, 증발·강화
  [SLIME] 점균류 관 네트워크 — 동반출현 그래프에서 유량이 굵어지는 6노드 추출(Tero 모델 단순화)

원칙: 진정 난수·VRF는 "예측"이 아니라 "인간·소프트웨어 편향 제거"가 목적이다.
      군집지능은 GA와 같은 적합도(볼점수 − λ인기도 + 다양성)를 다른 탐색 경로로 최적화한다.
      서로 다른 탐색기가 같은 조합에 수렴하면 그것은 적합도 지형의 진짜 봉우리다(앙상블 합의).
"""
from __future__ import annotations
import hashlib, json, math, secrets, time, urllib.request
import numpy as np

N_BALL, N_PICK = 45, 6
_rng = secrets.SystemRandom()


# =============================================================================
# [SEED] 진정 난수 시드 획득 — 모든 소스는 실패 시 OS CSPRNG로 폴백, 출처를 함께 반환
# =============================================================================
def _http_json(url: str, timeout: float = 6.0):
    req = urllib.request.Request(url, headers={"User-Agent": "lotto-ultimate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def seed_qrng(n_bytes: int = 32) -> bytes:
    """ANU 양자난수(진공 양자요동). 무료 엔드포인트는 변동이 잦아 실패하면 예외 → 상위에서 폴백."""
    d = _http_json(f"https://qrng.anu.edu.au/API/jsonI.php?length={n_bytes}&type=uint8")
    return bytes(d["data"])

def seed_random_org(n_bytes: int = 32) -> bytes:
    """random.org 대기 잡음 난수(quota 무료 구간)."""
    url = f"https://www.random.org/integers/?num={n_bytes}&min=0&max=255&col=1&base=10&format=plain&rnd=new"
    req = urllib.request.Request(url, headers={"User-Agent": "lotto-ultimate/1.0"})
    with urllib.request.urlopen(req, timeout=6) as r:
        return bytes(int(x) for x in r.read().decode().split())

def seed_usgs(n_bytes: int = 32) -> bytes:
    """USGS 최근 1시간 지진 피드: 규모·깊이·시각·좌표를 해시 → 지구물리학적 요동 시드."""
    d = _http_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson")
    blob = json.dumps([(f["properties"]["mag"], f["properties"]["time"], f["geometry"]["coordinates"])
                       for f in d["features"]], sort_keys=True).encode()
    return hashlib.sha256(blob).digest()[:n_bytes]

def get_true_seed(source: str = "auto") -> tuple[bytes, str]:
    """source: qrng | random_org | usgs | os | auto(순차 시도). 반환 (32바이트, 실제 사용 출처)."""
    order = {"auto": ["qrng", "random_org", "usgs"], "qrng": ["qrng"], "random_org": ["random_org"],
             "usgs": ["usgs"], "os": []}[source]
    fn = dict(qrng=seed_qrng, random_org=seed_random_org, usgs=seed_usgs)
    for name in order:
        try:
            b = fn[name]()
            if len(b) >= 16: return hashlib.sha256(b).digest(), name    # 소스 편향 제거용 해시 화이트닝
        except Exception:
            continue
    return secrets.token_bytes(32), "os_csprng"


# =============================================================================
# [VRF] 검증가능 난수 추출 — 블록체인 VRF/커밋-공개 아이디어의 오프라인 구현
#   1) 커밋: commit = SHA256(seed ‖ round ‖ salt) 를 추첨 전에 공개(prediction_log에 기록)
#   2) 추출: h_i = SHA256(seed ‖ round ‖ i) 를 반복, 각 바이트를 거부샘플링(<225 → mod 45)으로 번호화
#      → 225 = 45×5 이므로 mod 45가 완전 균등(모듈로 바이어스 0)
#   3) 검증: 누구든 seed와 round로 같은 6개 번호를 재생성 가능
# =============================================================================
def vrf_numbers(seed: bytes, round_no: int, set_id: int = 0) -> tuple:
    chosen, ctr = [], 0
    while len(chosen) < N_PICK:
        h = hashlib.sha256(seed + round_no.to_bytes(4, "big") + set_id.to_bytes(2, "big") + ctr.to_bytes(4, "big")).digest()
        for byte in h:
            if byte < 225:                                  # 거부 샘플링
                n = byte % 45 + 1
                if n not in chosen: chosen.append(n)
                if len(chosen) == N_PICK: break
        ctr += 1
    return tuple(sorted(chosen))

def vrf_commit(seed: bytes, round_no: int) -> str:
    return hashlib.sha256(seed + round_no.to_bytes(4, "big") + b"COMMIT").hexdigest()[:16]

def vrf_sets(seed: bytes, round_no: int, n: int, filter_fn) -> list[tuple]:
    """필터를 통과하는 VRF 조합 n개 (set_id를 올리며 결정론적으로 탐색 → 재현 가능)."""
    out, sid = [], 0
    while len(out) < n and sid < 5000:
        c = vrf_numbers(seed, round_no, sid); sid += 1
        if filter_fn(c)[0] and c not in out: out.append(c)
    return out


# =============================================================================
# 공통 적합도 (GA와 동일 — 탐색기 간 공정 비교를 위해 같은 지형을 쓴다)
# =============================================================================
def make_fitness(scores, prev, pop_fn, filter_fn, lam=1.0, labels=None, mu=0.02):
    def fit(c):
        c = tuple(sorted(c))
        f = scores[np.array(c) - 1].mean() - lam * pop_fn(c, prev)
        if labels is not None: f += mu * len({labels[n - 1] for n in c})
        return f - (0.5 if not filter_fn(c)[0] else 0.0)
    return fit


# =============================================================================
# [PSO] 입자 군집 최적화
#   입자 위치 x ∈ R^45 : 각 번호의 "선호 강도". 해석 = 상위 6개 인덱스 → 조합.
#   v = w·v + c1·r1·(pbest−x) + c2·r2·(gbest−x). 관성 w를 선형 감소(탐색→수렴).
#   연속→이산 해석이므로 지형이 계단형: 이를 완화하기 위해 시작 위치를 scores 근처에 배치.
# =============================================================================
def pso_optimize(scores, fitness, n_particles=60, iters=80, n_out=20):
    dim = N_BALL
    X = np.array([scores + 0.3 * np.array([_rng.gauss(0, 1) for _ in range(dim)]) for _ in range(n_particles)])
    V = np.zeros_like(X)
    decode = lambda x: tuple(sorted((np.argsort(-x)[:N_PICK] + 1).tolist()))
    pbest, pfit = X.copy(), np.array([fitness(decode(x)) for x in X])
    g = int(pfit.argmax()); gbest, gfit = X[g].copy(), pfit[g]
    seen = {decode(x): f for x, f in zip(X, pfit)}
    for it in range(iters):
        w = 0.9 - 0.5 * it / iters
        R1 = np.array([[_rng.random() for _ in range(dim)] for _ in range(n_particles)])
        R2 = np.array([[_rng.random() for _ in range(dim)] for _ in range(n_particles)])
        V = w * V + 1.5 * R1 * (pbest - X) + 1.5 * R2 * (gbest - X)
        X = X + V
        for i in range(n_particles):
            c = decode(X[i]); f = seen.get(c)
            if f is None: f = fitness(c); seen[c] = f
            if f > pfit[i]: pfit[i], pbest[i] = f, X[i].copy()
            if f > gfit: gfit, gbest = f, X[i].copy()
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    return [c for c, f in ranked if f > -0.1][:n_out], float(gfit)


# =============================================================================
# [ACO] 개미 군집 최적화
#   페로몬 τ(45×45) 초기값 = 동반출현 정규화. 개미는 현재 노드에서 τ^α·η^β 확률로 다음 노드 선택
#   (η = 후보 번호의 볼점수). 6노드 궤적 완성 후 적합도에 비례해 페로몬 강화, 매 세대 ρ 증발.
#   엘리트 개미(세대 최고)에 추가 강화 → 수렴 가속, ρ가 다양성 유지.
# =============================================================================
def aco_optimize(scores, fitness, cooc=None, n_ants=60, iters=60, alpha=1.0, beta=2.0, rho=0.15, n_out=20):
    tau = np.ones((N_BALL, N_BALL)) if cooc is None else (cooc / cooc.max() + 0.1)
    np.fill_diagonal(tau, 0)
    eta = scores + 0.05
    seen = {}
    for it in range(iters):
        trails = []
        for _ in range(n_ants):
            cur = _rng.randrange(N_BALL); path = [cur]
            while len(path) < N_PICK:
                p = (tau[cur] ** alpha) * (eta ** beta); p[path] = 0
                p = p / p.sum(); r = _rng.random(); cum = 0.0
                for j, pj in enumerate(p):
                    cum += pj
                    if r <= cum: cur = j; break
                path.append(cur)
            c = tuple(sorted(n + 1 for n in path))
            f = seen.get(c)
            if f is None: f = fitness(c); seen[c] = f
            trails.append((c, f))
        tau *= (1 - rho)                                     # 증발
        best_c, best_f = max(trails, key=lambda t: t[1])
        for c, f in trails:
            dep = max(f, 0) * (3.0 if c == best_c else 1.0)  # 엘리트 강화
            for a in c:
                for b in c:
                    if a != b: tau[a - 1, b - 1] += dep / 30
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    return [c for c, f in ranked if f > -0.1][:n_out], float(ranked[0][1])


# =============================================================================
# [SLIME] 점균류(Physarum) 관 네트워크 — Tero(2007) 모델 단순화
#   관 전도도 D_ij 는 흐르는 유량 |Q_ij| 에 비례해 굵어지고, 흐르지 않으면 가늘어진다.
#   출발/도착을 매 반복마다 무작위(진정난수)로 바꾸면 자주 쓰이는 간선만 살아남는다.
#   유량 Q = D·(p_i − p_j) 를 키르히호프 법칙으로 풀어 압력 p 를 구한다(선형계).
#   최종: 노드 강도(연결된 D 합) 상위에서 필터를 만족하는 6노드 조합을 추출.
# =============================================================================
def slime_optimize(scores, fitness, cooc, iters=120, n_out=20):
    W = cooc.astype(float); W = W / W.max()
    D = W * (0.5 + scores[:, None] + scores[None, :]) / 2 + 1e-3    # 초기 관 굵기 = 동반출현×볼점수
    np.fill_diagonal(D, 0)
    for _ in range(iters):
        src, dst = _rng.sample(range(N_BALL), 2)
        L = np.diag(D.sum(1)) - D                             # 라플라시안
        b = np.zeros(N_BALL); b[src], b[dst] = 1, -1
        L_r = L.copy(); L_r[dst] = 0; L_r[dst, dst] = 1; b[dst] = 0   # 기준 압력 고정
        p = np.linalg.solve(L_r + 1e-9 * np.eye(N_BALL), b)
        Q = np.abs(D * (p[:, None] - p[None, :]))
        D = 0.9 * D + 0.4 * Q / (Q.max() + 1e-12)              # 성장(유량) − 쇠퇴
        np.fill_diagonal(D, 0)
    strength = D.sum(1)
    # 굵은 노드 상위 14개에서 6개 조합을 탐색(적합도 순)
    top = (np.argsort(-strength)[:14] + 1).tolist()
    from itertools import combinations
    cands = sorted(((fitness(c), c) for c in combinations(top, 6)), reverse=True)
    return [c for f, c in cands if f > -0.1][:n_out], float(cands[0][0]), strength


# =============================================================================
# 앙상블 합의 — 서로 다른 탐색기(GA·PSO·ACO·SLIME)가 공통으로 찍은 번호를 집계
# =============================================================================
def consensus(cand_lists: dict[str, list[tuple]]) -> tuple[np.ndarray, list]:
    vote = np.zeros(N_BALL)
    for name, lst in cand_lists.items():
        for c in lst:
            for n in c: vote[n - 1] += 1 / max(len(lst), 1)
    common = {}
    for name, lst in cand_lists.items():
        for c in lst: common.setdefault(c, set()).add(name)
    agreed = sorted([(len(v), c, sorted(v)) for c, v in common.items() if len(v) >= 2], reverse=True)
    return vote / max(len(cand_lists), 1), agreed
