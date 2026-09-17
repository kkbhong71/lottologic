/* =============================================================================
   LOTTO ULTIMATE — 브라우저 최적화 모듈 (lotto_ultimate_ml.py / _swarm.py 의 JS 이식)
   [GA]  유전 알고리즘 (적응형 돌연변이)          ← lotto_ultimate_ml.genetic_optimize
   [PSO] 입자 군집 최적화                        ← lotto_ultimate_swarm.pso_optimize
   [VRF] SHA-256 검증가능 난수 (crypto.subtle)    ← lotto_ultimate_swarm.vrf_numbers / vrf_commit
   [CONSENSUS] 탐색기 합의                        ← lotto_ultimate_swarm.consensus
   [TRACKER] localStorage 사전등록·사후채점       ← lotto_ultimate.register/score_predictions
   의존: LU (lotto_engine.js)
   ============================================================================= */
const LUX = (() => {
  const _LU = typeof LU !== 'undefined' ? LU : require('./lotto_engine.js');
  const N_BALL = 45, N_PICK = 6;
  const { random, randrange, sample, passesFilters, popularityIndex } = _LU;
  const key = c => c.join(',');
  const sorted = c => c.slice().sort((a, b) => a - b);
  /** 표준정규 난수(Box–Muller, crypto 기반) — Python random.gauss 대응 */
  const gauss = () => { let u = 0, v = 0; while (u === 0) u = random(); while (v === 0) v = random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  /* ---------- 공통 적합도 (Python make_fitness / GA fitness 와 동일 지형) ---------- */
  function entropyBonus(c) {                                          // 간격 분포 Shannon 엔트로피 / log2(5)
    const g = []; for (let i = 0; i < 5; i++) g.push(c[i + 1] - c[i]);
    const cnt = {}; for (const v of g) cnt[v] = (cnt[v] || 0) + 1;
    const p = Object.values(cnt).map(v => v / 5);
    return -p.reduce((a, v) => a + v * Math.log2(v), 0) / Math.log2(5);
  }
  const makeFitness = (scores, prev, { lam = 1.0, labels = null, mu = 0.05, entropy = true } = {}) => c => {
    c = sorted(c);
    let f = c.reduce((a, n) => a + scores[n - 1], 0) / 6 - lam * popularityIndex(c, prev);
    if (entropy) f += 0.05 * entropyBonus(c);
    if (labels) f += mu * new Set(c.map(n => labels[n - 1])).size;
    return f - (passesFilters(c)[0] ? 0 : 0.5);
  };

  /* ---------- [GA] ---------- */
  function geneticOptimize(scores, prev, { popSize = 300, generations = 60, lam = 1.0, labels = null, nOut = 40 } = {}) {
    const fitness = makeFitness(scores, prev, { lam, labels });
    let pool = Array.from({ length: popSize }, () => sorted(sample(Array.from({ length: 45 }, (_, i) => i + 1), 6)));
    let mut = 0.10; const history = [];
    for (let g = 0; g < generations; g++) {
      let fit = pool.map(fitness);
      const order = fit.map((f, i) => i).sort((a, b) => fit[b] - fit[a]);
      pool = order.map(i => pool[i]); fit = order.map(i => fit[i]);
      const elite = pool.slice(0, Math.floor(popSize / 10));
      const diversity = new Set(pool.map(key)).size / popSize;
      mut = Math.min(0.4, Math.max(0.03, mut * (diversity < 0.6 ? 1.3 : 0.85)));   // 적응형 돌연변이
      history.push({ g, best: fit[0], diversity, mut });
      const children = elite.slice();
      while (children.length < popSize) {
        const a = pool[Math.min(randrange(popSize), randrange(popSize))], b = pool[Math.min(randrange(popSize), randrange(popSize))];  // 토너먼트
        const genes = [...new Set([...a, ...b])];
        let child = new Set(sample(genes, 6));
        if (random() < mut) {
          const k = [1, 1, 2][randrange(3)];
          for (let i = 0; i < k; i++) { const arr = [...child]; child.delete(arr[randrange(arr.length)]); while (child.size < 6) child.add(randrange(45) + 1); }
        }
        children.push(sorted([...child]));
      }
      pool = children;
    }
    const fit = pool.map(fitness), order = fit.map((f, i) => i).sort((a, b) => fit[b] - fit[a]);
    const seen = new Set(), out = [];
    for (const i of order) { const c = pool[i]; if (seen.has(key(c)) || !passesFilters(c)[0]) continue; seen.add(key(c)); out.push(c); if (out.length >= nOut) break; }
    return { cands: out, history };
  }

  /* ---------- [PSO] ---------- */
  function psoOptimize(scores, prev, { nParticles = 60, iters = 80, lam = 1.0, labels = null, nOut = 20 } = {}) {
    const fitness = makeFitness(scores, prev, { lam, labels, mu: 0.02, entropy: false });
    const decode = x => sorted(x.map((v, i) => [v, i + 1]).sort((a, b) => b[0] - a[0]).slice(0, 6).map(p => p[1]));
    let X = Array.from({ length: nParticles }, () => scores.map(s => s + 0.3 * gauss()));
    let V = X.map(() => new Array(N_BALL).fill(0));
    const seen = new Map(); const evalC = c => { const k = key(c); if (!seen.has(k)) seen.set(k, fitness(c)); return seen.get(k); };
    let pbest = X.map(x => x.slice()), pfit = X.map(x => evalC(decode(x)));
    let gi = pfit.indexOf(Math.max(...pfit)), gbest = X[gi].slice(), gfit = pfit[gi];
    for (let it = 0; it < iters; it++) {
      const w = 0.9 - 0.5 * it / iters;
      for (let i = 0; i < nParticles; i++) {
        for (let d = 0; d < N_BALL; d++) { V[i][d] = w * V[i][d] + 1.5 * random() * (pbest[i][d] - X[i][d]) + 1.5 * random() * (gbest[d] - X[i][d]); X[i][d] += V[i][d]; }
        const f = evalC(decode(X[i]));
        if (f > pfit[i]) { pfit[i] = f; pbest[i] = X[i].slice(); }
        if (f > gfit) { gfit = f; gbest = X[i].slice(); }
      }
    }
    const ranked = [...seen.entries()].sort((a, b) => b[1] - a[1]).filter(([, f]) => f > -0.1).slice(0, nOut);
    return { cands: ranked.map(([k]) => k.split(',').map(Number)), best: gfit };
  }

  /* ---------- [VRF] SHA-256 검증가능 난수 — Python vrf_numbers 와 바이트 단위 동일 ---------- */
  const subtle = (globalThis.crypto && globalThis.crypto.subtle) || require('crypto').webcrypto.subtle;
  const be = (n, bytes) => { const a = new Uint8Array(bytes); for (let i = bytes - 1; i >= 0; i--) { a[i] = n & 0xff; n = Math.floor(n / 256); } return a; };
  const cat = (...arrs) => { const out = new Uint8Array(arrs.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of arrs) { out.set(x, o); o += x.length; } return out; };
  const sha256 = async data => new Uint8Array(await subtle.digest('SHA-256', data));
  const hex = a => [...a].map(b => b.toString(16).padStart(2, '0')).join('');
  async function vrfNumbers(seed, roundNo, setId = 0) {
    const chosen = []; let ctr = 0;
    while (chosen.length < N_PICK) {
      const h = await sha256(cat(seed, be(roundNo, 4), be(setId, 2), be(ctr, 4)));
      for (const b of h) { if (b < 225) { const n = b % 45 + 1; if (!chosen.includes(n)) chosen.push(n); if (chosen.length === N_PICK) break; } }   // 거부 샘플링
      ctr++;
    }
    return sorted(chosen);
  }
  const vrfCommit = async (seed, roundNo) => hex(await sha256(cat(seed, be(roundNo, 4), new TextEncoder().encode('COMMIT')))).slice(0, 16);
  async function vrfSets(seed, roundNo, n) {
    const out = []; let sid = 0;
    while (out.length < n && sid < 5000) { const c = await vrfNumbers(seed, roundNo, sid++); if (passesFilters(c)[0] && !out.some(o => key(o) === key(c))) out.push(c); }
    return out;
  }
  /** 브라우저 시드: OS CSPRNG 32바이트 → SHA-256 화이트닝 (외부 QRNG는 CORS로 브라우저 직접 호출 불가 → Actions 쪽이 담당) */
  async function browserSeed() { const raw = new Uint8Array(32); (globalThis.crypto || require('crypto').webcrypto).getRandomValues(raw); return { seed: await sha256(raw), source: 'browser_csprng' }; }

  /* ---------- [CONSENSUS] ---------- */
  function consensus(lists) {
    const vote = new Array(N_BALL).fill(0), common = {};
    for (const name in lists) { const lst = lists[name]; for (const c of lst) { for (const n of c) vote[n - 1] += 1 / Math.max(lst.length, 1); (common[key(c)] = common[key(c)] || new Set()).add(name); } }
    const k = Object.keys(lists).length || 1;
    const agreed = Object.entries(common).filter(([, s]) => s.size >= 2).map(([c, s]) => ({ combo: c.split(',').map(Number), by: [...s].sort() })).sort((a, b) => b.by.length - a.by.length);
    return { vote: vote.map(v => v / k), agreed };
  }

  /* ---------- [TRACKER] localStorage 사전등록 · 사후채점 ---------- */
  const LS_KEY = 'lotto_ultimate_log';
  const store = (typeof localStorage !== 'undefined') ? localStorage : { _d: {}, getItem(k) { return this._d[k] ?? null; }, setItem(k, v) { this._d[k] = String(v); } };
  const readLog = () => { try { return JSON.parse(store.getItem(LS_KEY) || '[]'); } catch { return []; } };
  const writeLog = log => store.setItem(LS_KEY, JSON.stringify(log));
  /** 회차당 1회만 등록 (중복 방지) → 등록 여부 반환 */
  function registerPredictions(sets, targetRound, meta = {}) {
    const log = readLog();
    if (log.some(e => e.target_round === targetRound)) return false;
    sets.forEach((s, i) => log.push({ target_round: targetRound, set_id: i + 1, nums: s.combo, pop: +(s.pop ?? 0).toFixed(4), vrf: !!s.vrf, hits: null, scored: 0, ts: new Date().toISOString(), ...meta }));
    writeLog(log); return true;
  }
  /** 로드된 회차 데이터로 미채점 예측을 채점 */
  function scorePredictions(rows) {
    const actual = new Map(rows.map(r => [r.round, new Set(r.nums)]));
    const log = readLog(); let newly = 0;
    for (const e of log) if (!e.scored && actual.has(e.target_round)) { const a = actual.get(e.target_round); e.hits = e.nums.filter(n => a.has(n)).length; e.scored = 1; newly++; }
    writeLog(log);
    const done = log.filter(e => e.scored);
    return { n: done.length, newly, meanHits: done.length ? done.reduce((a, e) => a + e.hits, 0) / done.length : null, match3plus: done.filter(e => e.hits >= 3).length, log };
  }
  const exportLog = () => JSON.stringify(readLog(), null, 1);
  const importLog = json => { const inc = JSON.parse(json); const log = readLog(); const have = new Set(log.map(e => `${e.target_round}-${e.set_id}`)); for (const e of inc) if (!have.has(`${e.target_round}-${e.set_id}`)) log.push(e); log.sort((a, b) => a.target_round - b.target_round || a.set_id - b.set_id); writeLog(log); return log.length; };

  return { makeFitness, entropyBonus, geneticOptimize, psoOptimize, vrfNumbers, vrfCommit, vrfSets, browserSeed, hex, consensus,
           registerPredictions, scorePredictions, readLog, exportLog, importLog };
})();
if (typeof module !== 'undefined') module.exports = LUX;
