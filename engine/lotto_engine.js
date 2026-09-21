/* =============================================================================
   LOTTO ULTIMATE — 브라우저 JS 엔진 코어 (lotto_ultimate.py 1:1 이식)
   - 외부 라이브러리 없음. 난수는 crypto.getRandomValues (OS CSPRNG).
   - Python과 같은 연산 순서를 지켜 45볼 점수를 부동소수점 수준까지 일치시킨다.
   - 이 파일은 index.html에 인라인되며, Node에서도 그대로 실행되어 교차검증에 쓰인다.
   ============================================================================= */
const LU = (() => {
  const N_BALL = 45, N_PICK = 6;

  /* ---------- 진정 난수 (crypto) ---------- */
  const _crypto = (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues)
    ? globalThis.crypto : require('crypto').webcrypto;
  const _u32 = new Uint32Array(2);
  /** Python SystemRandom.random()과 같은 53비트 정밀도 [0,1) */
  const random = () => { _crypto.getRandomValues(_u32); return ((_u32[0] >>> 5) * 67108864 + (_u32[1] >>> 6)) / 9007199254740992; };
  /** 모듈로 바이어스 제거 정수 난수 [0,n) — 거부 샘플링 */
  const randrange = n => { const lim = Math.floor(4294967296 / n) * n; const a = new Uint32Array(1); let x; do { _crypto.getRandomValues(a); x = a[0]; } while (x >= lim); return x % n; };
  const sample = (arr, k) => { const a = arr.slice(); for (let i = 0; i < k; i++) { const j = i + randrange(a.length - i); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); };

  /* ---------- [LAYER 1] 로더: 인코딩 감지 + 3형식 인식 + 검증 ---------- */
  function decodeBytes(buf) {
    try { return new TextDecoder('utf-8', { fatal: true }).decode(buf); }       // UTF-8(-sig)
    catch { try { return new TextDecoder('euc-kr').decode(buf); } catch { return new TextDecoder('latin1').decode(buf); } }  // CP949
  }
  function parseCSV(text) {
    const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(l => l.trim().length);
    const rows = lines.map(l => l.split(',').map(s => s.trim()));
    const head = rows[0].map(s => s.toLowerCase().replace(/\s+/g, '_'));
    let out = [];
    const numIdx = head.map((h, i) => h.startsWith('num') ? i : -1).filter(i => i >= 0);
    if (numIdx.length === 6) {                                        // 형식A 표준
      const ri = head.indexOf('round') >= 0 ? head.indexOf('round') : 0;
      const di = head.indexOf('draw_date');
      for (const r of rows.slice(1)) out.push({ round: +r[ri], date: di >= 0 ? r[di] : '', nums: numIdx.map(i => +r[i]) });
    } else if (head.some(h => h.includes('회차'))) {                   // 형식B 동행복권 원본
      const ri = head.findIndex(h => h.includes('회차'));
      for (const r of rows.slice(1)) {
        const ints = r.map((v, i) => [i, +v]).filter(([i, v]) => i !== ri && Number.isInteger(v) && v >= 1 && v <= 45).map(([i]) => i).slice(0, 6);
        if (ints.length === 6) out.push({ round: +r[ri], date: '', nums: ints.map(i => +r[i]) });
      }
    } else {                                                          // 형식C 헤더 없음
      for (const r of rows) if (r.length >= 7) out.push({ round: +r[0], date: '', nums: r.slice(1, 7).map(Number) });
    }
    // 방어4~6: 범위·중복·정렬·연속성
    const issues = [];
    out = out.filter(d => {
      const ok = d.nums.every(n => Number.isInteger(n) && n >= 1 && n <= 45) && new Set(d.nums).size === 6 && Number.isInteger(d.round);
      if (!ok) issues.push(`이상 행 제거: 회차 ${d.round}`);
      return ok;
    });
    out.sort((a, b) => a.round - b.round);
    const seen = new Set(); out = out.filter(d => !seen.has(d.round) && seen.add(d.round));
    out.forEach(d => d.nums.sort((a, b) => a - b));
    for (let i = 1; i < out.length; i++) if (out[i].round - out[i - 1].round !== 1) issues.push(`회차 누락: ${out[i - 1].round}→${out[i].round}`);
    return { rows: out, issues };
  }
  const loadCSV = buf => parseCSV(typeof buf === 'string' ? buf : decodeBytes(buf));

  /* ---------- [LAYER 2] 8종 통계 스코어 ---------- */
  const norm = x => { const mn = Math.min(...x), mx = Math.max(...x), r = mx - mn; return r > 0 ? x.map(v => (v - mn) / r) : x.map(() => 0.5); };
  const appearMatrix = hist => hist.map(nums => { const row = new Uint8Array(N_BALL); for (const n of nums) row[n - 1] = 1; return row; });
  const colSum = (A, from = 0) => { const s = new Float64Array(N_BALL); for (let t = from; t < A.length; t++) for (let b = 0; b < N_BALL; b++) s[b] += A[t][b]; return Array.from(s); };
  const colMean = (A, from = 0) => colSum(A, from).map(v => v / (A.length - from));

  const scoreFrequency = A => norm(colSum(A));
  function scoreRecency(A, n = 50) {                                  // np.exp(np.linspace(-2,0,m)) 가중
    const m = Math.min(n, A.length), w = [], s = new Float64Array(N_BALL);
    for (let i = 0; i < m; i++) w.push(Math.exp(-2 + i * (2 / (m - 1))));
    for (let i = 0; i < m; i++) { const row = A[A.length - m + i]; for (let b = 0; b < N_BALL; b++) s[b] += row[b] * w[i]; }
    return norm(Array.from(s));
  }
  function scoreGap(A) {
    const last = new Array(N_BALL).fill(-1);
    for (let t = 0; t < A.length; t++) for (let b = 0; b < N_BALL; b++) if (A[t][b]) last[b] = t;
    return norm(last.map(l => A.length - 1 - l));
  }
  function scoreZscore(A) {                                           // 모집단 표준편차(ddof=0), 콜드 우선(부호 반전)
    const f = colSum(A); const mu = f.reduce((a, b) => a + b, 0) / N_BALL;
    const sd = Math.sqrt(f.reduce((a, v) => a + (v - mu) ** 2, 0) / N_BALL) || 1;
    return norm(f.map(v => -(v - mu) / sd));
  }
  function scoreMomentum(A, short = 20, long = 100) {
    const s = colMean(A, Math.max(0, A.length - short)), l = colMean(A, Math.max(0, A.length - long));
    return norm(s.map((v, i) => v / (l[i] + 1e-9)));
  }
  function cooc(A) {                                                  // A^T A (45×45), 대각 0
    const P = Array.from({ length: N_BALL }, () => new Float64Array(N_BALL));
    for (const row of A) { const idx = []; for (let b = 0; b < N_BALL; b++) if (row[b]) idx.push(b); for (const a of idx) for (const c of idx) if (a !== c) P[a][c] += 1; }
    return P;
  }
  const scorePair = P => norm(P.map(r => r.reduce((a, b) => a + b, 0)));
  function scoreMarkov(hist) {                                        // 라플라스 평활 전이확률, 마지막 회차 조건부 합
    const T = Array.from({ length: N_BALL }, () => new Float64Array(N_BALL).fill(1));
    for (let t = 0; t + 1 < hist.length; t++) for (const a of hist[t]) for (const n of hist[t + 1]) T[a - 1][n - 1] += 1;
    for (const r of T) { const s = r.reduce((a, b) => a + b, 0); for (let j = 0; j < N_BALL; j++) r[j] /= s; }
    const out = new Float64Array(N_BALL); for (const a of hist[hist.length - 1]) for (let j = 0; j < N_BALL; j++) out[j] += T[a - 1][j];
    return norm(Array.from(out));
  }
  function scoreLift(A, P) {                                          // P(A∩B)/(P(A)P(B))
    const n = A.length, p = colMean(A), out = [];
    for (let i = 0; i < N_BALL; i++) { let s = 0; for (let j = 0; j < N_BALL; j++) if (i !== j) s += (P[i][j] / n) / (p[i] * p[j] + 1e-12); out.push(s); }
    return norm(out);
  }
  function scoreSigmoidCold(A) {
    /** 시그모이드 콜드번호 재평가 — 장기 미출현에 비선형 상한. c=기대갭(7.5), k=0.3 */
    const last = new Array(N_BALL).fill(-1);
    for (let t = 0; t < A.length; t++) for (let b = 0; b < N_BALL; b++) if (A[t][b]) last[b] = t;
    const c = N_BALL / N_PICK;
    return norm(last.map(l => 1.0 / (1.0 + Math.exp(-0.3 * ((A.length - 1 - l) - c)))));
  }
  function scoreMultiWindow(hist, windows = [30, 50, 100]) {
    /** 다중 윈도우 민감도 — 복수 시간 범위 빈도의 정규화 평균. 안정 신호일수록 높음 */
    const A = appearMatrix(hist);
    const allW = [...windows, hist.length];
    const normed = allW.map(w => { const seg = A.slice(-Math.min(w, A.length)); return norm(colMean(seg)); });
    return norm(Array.from({ length: N_BALL }, (_, b) => normed.reduce((a, row) => a + row[b], 0) / normed.length));
  }
  function scoreDirichlet(hist, alpha0 = 1.0) {
    /** Dirichlet 사후확률 — 균등 사전(α₀) + 관측 빈도 → 사후 기대값 */
    const f = new Array(N_BALL).fill(0);
    for (const nums of hist) for (const n of nums) f[n - 1]++;
    const total = f.reduce((a, b) => a + b, 0);
    return norm(f.map(v => (v + alpha0) / (total + N_BALL * alpha0)));
  }
  const SCORE_WEIGHTS = { frequency: 1.0, recency: 1.2, gap: 1.0, zscore: 0.8, sigmoid_cold: 0.7,
                          momentum: 1.0, pair: 0.8, markov: 1.0, lift: 0.8,
                          multi_window: 0.9, dirichlet: 0.6 };
  const EXTRA_WEIGHTS = { ml: 1.5, network: 0.8 };
  function compositeScores(hist, extra = null) {
    const A = appearMatrix(hist), P = cooc(A);
    const parts = { frequency: scoreFrequency(A), recency: scoreRecency(A), gap: scoreGap(A), zscore: scoreZscore(A),
                    sigmoid_cold: scoreSigmoidCold(A), momentum: scoreMomentum(A),
                    pair: scorePair(P), markov: scoreMarkov(hist), lift: scoreLift(A, P),
                    multi_window: scoreMultiWindow(hist), dirichlet: scoreDirichlet(hist) };
    const w = { ...SCORE_WEIGHTS };
    if (extra) for (const k in extra) { parts[k] = extra[k]; w[k] = EXTRA_WEIGHTS[k] ?? 1.0; }
    const keys = Object.keys(parts), wsum = keys.reduce((a, k) => a + w[k], 0);
    const total = Array.from({ length: N_BALL }, (_, b) => keys.reduce((a, k) => a + w[k] * parts[k][b], 0) / wsum);
    return { total, parts };
  }

  /* ---------- [LAYER 3] 인기도 역산 ---------- */
  const POP_COEF = { sum: -0.049, same_row_max: -0.074, tail_repeat: 0.059, prev_overlap: -0.014 };
  const POP_STD = { sum: [138, 32], same_row_max: [2.1, 0.75], tail_repeat: [1.2, 0.9], prev_overlap: [0.8, 0.75] };
  const LUCKY = new Set([3, 7, 13, 21, 27]);
  function comboFeatures(c, prev) {
    const rows = {}, tails = {};
    for (const n of c) { const r = Math.floor((n - 1) / 7); rows[r] = (rows[r] || 0) + 1; const t = n % 10; tails[t] = (tails[t] || 0) + 1; }
    const pv = new Set(prev);
    return { sum: c.reduce((a, b) => a + b, 0), same_row_max: Math.max(...Object.values(rows)),
             tail_repeat: Object.values(tails).filter(v => v > 1).reduce((a, v) => a + v - 1, 0), prev_overlap: c.filter(n => pv.has(n)).length };
  }
  function popularityIndex(c, prev) {
    const f = comboFeatures(c, prev);
    let z = 0; for (const k in POP_COEF) z += POP_COEF[k] * (f[k] - POP_STD[k][0]) / POP_STD[k][1];
    z += 0.03 * c.filter(n => n <= 31).length - 0.03 * 3.9;
    z += 0.04 * c.filter(n => LUCKY.has(n)).length;
    return z;
  }

  /* ---------- [LAYER 4] 8종 필터 ---------- */
  function acValue(c) { const d = new Set(); for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) d.add(Math.abs(c[i] - c[j])); return d.size - 5; }
  function passesFilters(c) {
    c = c.slice().sort((a, b) => a - b);
    const s = c.reduce((a, b) => a + b, 0);
    if (s < 100 || s > 175) return [false, 'sum'];
    if (acValue(c) < 7) return [false, 'ac'];
    const odd = c.filter(n => n % 2).length; if ([0, 1, 5, 6].includes(odd)) return [false, 'oddeven'];
    const low = c.filter(n => n <= 22).length; if ([0, 1, 5, 6].includes(low)) return [false, 'lowhigh'];
    let runs = 0; for (let i = 0; i < 5; i++) if (c[i + 1] === c[i] + 1) runs++;
    if (runs >= 3) return [false, 'consec']; for (let i = 0; i < 4; i++) if (c[i + 2] === c[i] + 2) return [false, 'consec'];
    const tails = {}; for (const n of c) tails[n % 10] = (tails[n % 10] || 0) + 1; if (Math.max(...Object.values(tails)) >= 3) return [false, 'tail'];
    if (new Set(c.map(n => Math.floor((n - 1) / 10))).size < 3) return [false, 'decade'];
    const d = []; for (let i = 0; i < 5; i++) d.push(c[i + 1] - c[i]); if (new Set(d).size === 1) return [false, 'arith'];
    const rows = {}; for (const n of c) { const r = Math.floor((n - 1) / 7); rows[r] = (rows[r] || 0) + 1; } if (Math.max(...Object.values(rows)) >= 4) return [false, 'row'];
    if (c.every(n => n <= 31)) return [false, 'birthday'];
    return [true, ''];
  }

  /* ---------- [LAYER 5] 생성: 가중 샘플링 + KMeans + 탐욕 선택(제약 2종) ---------- */
  function weightedPick(scores, k = N_PICK, temperature = 0.6) {
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const p = scores.map(s => Math.exp((s - mean) / Math.max(temperature, 1e-6)));
    const avail = Array.from({ length: N_BALL }, (_, i) => i), chosen = [];
    for (let t = 0; t < k; t++) {
      const tot = avail.reduce((a, i) => a + p[i], 0); const r = random(); let cum = 0, pick = avail[avail.length - 1];
      for (const i of avail) { cum += p[i] / tot; if (r <= cum) { pick = i; break; } }
      chosen.push(pick + 1); avail.splice(avail.indexOf(pick), 1);
    }
    return chosen.sort((a, b) => a - b);
  }
  function kmeans(X, k, nInit = 5, iters = 25) {                     // k-means++ 초기화, 최소 관성 선택
    let best = null;
    for (let run = 0; run < nInit; run++) {
      const C = [X[randrange(X.length)].slice()];
      while (C.length < k) {                                          // ++ 시드
        const d2 = X.map(x => Math.min(...C.map(c => x.reduce((a, v, i) => a + (v - c[i]) ** 2, 0))));
        const tot = d2.reduce((a, b) => a + b, 0); let r = random() * tot, idx = 0;
        for (; idx < d2.length - 1; idx++) { r -= d2[idx]; if (r <= 0) break; } C.push(X[idx].slice());
      }
      let labels = new Array(X.length).fill(0);
      for (let it = 0; it < iters; it++) {
        const nl = X.map(x => { let bi = 0, bd = Infinity; C.forEach((c, j) => { const d = x.reduce((a, v, i) => a + (v - c[i]) ** 2, 0); if (d < bd) { bd = d; bi = j; } }); return bi; });
        if (nl.every((v, i) => v === labels[i])) break; labels = nl;
        C.forEach((c, j) => { const m = X.filter((_, i) => labels[i] === j); if (m.length) for (let i = 0; i < c.length; i++) c[i] = m.reduce((a, x) => a + x[i], 0) / m.length; });
      }
      const inertia = X.reduce((a, x, i) => a + x.reduce((s, v, d) => s + (v - C[labels[i]][d]) ** 2, 0), 0);
      if (!best || inertia < best.inertia) best = { labels, inertia };
    }
    return best.labels;
  }
  function generateSets(scores, prev, { nSets = 10, pool = 3000, popPenalty = 1.0, extraCands = [], labels = null, maxPerBall = null } = {}) {
    maxPerBall = maxPerBall || Math.max(2, Math.ceil(nSets * 0.4));
    const seen = new Set(), cands = [];
    const src = []; for (let i = 0; i < pool; i++) src.push(weightedPick(scores)); for (const c of extraCands) src.push(c);
    for (const c of src) {
      const key = c.join(','); if (seen.has(key)) continue;
      if (!passesFilters(c)[0]) continue; seen.add(key);
      const pop = popularityIndex(c, prev), ball = c.reduce((a, n) => a + scores[n - 1], 0) / 6;
      const comm = labels ? new Set(c.map(n => labels[n - 1])).size : 0;
      cands.push({ combo: c, ball, pop, comm, total: ball - popPenalty * pop + 0.02 * comm });
    }
    if (!cands.length) return [];
    const k = Math.min(nSets, cands.length);
    const X = cands.map(d => { const v = new Array(N_BALL).fill(0); for (const n of d.combo) v[n - 1] = 1; return v; });
    const clus = k > 1 ? kmeans(X, k) : new Array(cands.length).fill(0);
    const order = cands.map((_, i) => i).sort((a, b) => cands[b].total - cands[a].total);
    const picked = [], usedCl = new Set(), count = new Array(46).fill(0);
    for (const strict of [true, false]) for (const i of order) {
      if (picked.length >= nSets) break; const d = cands[i];
      if (picked.includes(d) || !d.combo.every(n => count[n] < maxPerBall)) continue;
      if (strict && usedCl.has(clus[i])) continue;
      picked.push(d); usedCl.add(clus[i]); for (const n of d.combo) count[n]++;
    }
    return picked;
  }

  /* ---------- 감사: 균등성 χ² (자유도 44) ---------- */
  function audit(rows) {
    const f = new Array(N_BALL).fill(0); for (const r of rows) for (const n of r.nums) f[n - 1]++;
    const exp = rows.length * 6 / 45, chi2 = f.reduce((a, v) => a + (v - exp) ** 2 / exp, 0);
    // χ²(44) 상단 확률: Wilson–Hilferty 정규 근사
    const z = ((chi2 / 44) ** (1 / 3) - (1 - 2 / (9 * 44))) / Math.sqrt(2 / (9 * 44));
    const p = 0.5 * erfc(z / Math.SQRT2);
    return { rounds: rows.length, first: rows[0].round, last: rows[rows.length - 1].round, lastDate: rows[rows.length - 1].date, freqMin: Math.min(...f), freqMax: Math.max(...f), chi2, p };
  }
  function erfc(x) { const t = 1 / (1 + 0.5 * Math.abs(x)); const r = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); return x >= 0 ? r : 2 - r; }

  return { N_BALL, random, randrange, sample, loadCSV, parseCSV, appearMatrix, compositeScores, popularityIndex, passesFilters,
           weightedPick, generateSets, audit, SCORE_WEIGHTS };
})();
if (typeof module !== 'undefined') module.exports = LU;
