/* =============================================================================
   LOTTO ULTIMATE — 교육 모듈 (Phase 5)
   1교시: 인기도 자가진단 — 내 번호가 얼마나 대중적인지 즉시 측정
   2교시: 대수의 법칙 — 시행 횟수를 늘리면 빈도가 1/45에 수렴
   3교시: 다중비교 함정 — 15개 검정 중 하나가 p<0.05인 건 우연
   4~6교시: 다음 턴에서 추가
   의존: LU (lotto_engine.js)
   ============================================================================= */
const EDU = (() => {
  const _LU = typeof LU !== 'undefined' ? LU : require('./lotto_engine.js');
  const N_BALL = 45, N_PICK = 6;
  const { random, randrange, popularityIndex } = _LU;

  /* ── 1교시: 인기도 자가진단 ── */
  /** 사용자 번호 6개 + 직전 회차 번호 → 인기도 점수 + 피처 해설 반환 */
  function diagnosePopularity(userNums, prevNums) {
    const c = userNums.slice().sort((a, b) => a - b);
    const pop = popularityIndex(c, prevNums);
    const s = c.reduce((a, b) => a + b, 0);
    const birthday = c.filter(n => n <= 31).length;
    const lucky = c.filter(n => [3, 7, 13, 21, 27].includes(n)).length;
    const rows = {}; for (const n of c) { const r = Math.floor((n - 1) / 7); rows[r] = (rows[r] || 0) + 1; }
    const tails = {}; for (const n of c) { const t = n % 10; tails[t] = (tails[t] || 0) + 1; }
    const tailRepeat = Object.values(tails).filter(v => v > 1).reduce((a, v) => a + v - 1, 0);
    const prevOverlap = c.filter(n => prevNums.includes(n)).length;

    const factors = [];
    if (birthday >= 5) factors.push({ text: `6개 중 ${birthday}개가 생일 범위(1~31) — 대중이 가장 많이 고르는 구간`, impact: 'bad' });
    else if (birthday <= 2) factors.push({ text: `생일 범위(1~31) ${birthday}개만 포함 — 고번호(32~45) 활용 좋음`, impact: 'good' });
    if (lucky > 0) factors.push({ text: `행운수(3,7,13,21,27) ${lucky}개 포함 — 많은 사람이 선호하는 번호`, impact: 'bad' });
    if (s < 120) factors.push({ text: `합계 ${s} — 낮은 합계는 인기 구간(100~140)`, impact: 'bad' });
    else if (s > 165) factors.push({ text: `합계 ${s} — 높은 합계는 비인기 구간`, impact: 'good' });
    if (tailRepeat >= 2) factors.push({ text: `같은 끝수가 반복(${tailRepeat}쌍) — 사람들이 무의식적으로 피하는 패턴이라 비인기`, impact: 'good' });
    if (prevOverlap >= 2) factors.push({ text: `직전 회차와 ${prevOverlap}개 겹침 — "또 나올 리 없어"라고 피하는 사람이 많아 비인기`, impact: 'good' });
    if (Math.max(...Object.values(rows)) >= 3) factors.push({ text: `OMR 용지 같은 행에 ${Math.max(...Object.values(rows))}개 집중 — 한 줄 긋기 패턴, 비인기`, impact: 'good' });

    const grade = pop > 0.15 ? { label: '매우 인기', color: '#FF7272', emoji: '🔴' }
                : pop > 0.05 ? { label: '인기', color: '#FFC978', emoji: '🟡' }
                : pop > -0.05 ? { label: '보통', color: '#A7B4CF', emoji: '⚪' }
                : pop > -0.15 ? { label: '비인기', color: '#7FE0A8', emoji: '🟢' }
                : { label: '매우 비인기', color: '#69C8F2', emoji: '🔵' };

    return { pop: +pop.toFixed(4), grade, factors, detail: { sum: s, birthday, lucky, tailRepeat, prevOverlap, maxRow: Math.max(...Object.values(rows)) } };
  }

  /** 무작위 1000조합의 인기도 분포 → 사용자 번호 위치를 백분위로 표시 */
  function popPercentile(userPop, prevNums, n = 1000) {
    const pops = [];
    for (let i = 0; i < n; i++) {
      const c = []; const used = new Set();
      while (c.length < 6) { const r = randrange(45) + 1; if (!used.has(r)) { used.add(r); c.push(r); } }
      pops.push(popularityIndex(c.sort((a, b) => a - b), prevNums));
    }
    pops.sort((a, b) => a - b);
    const rank = pops.filter(p => p <= userPop).length;
    return { percentile: Math.round(rank / n * 100), distribution: pops };
  }

  /* ── 2교시: 대수의 법칙 ── */
  /** n회 추첨 시뮬레이션 → 45개 번호 빈도 배열 + 이론 기대값 + 최대 편차 */
  function lawOfLargeNumbers(n) {
    const freq = new Array(N_BALL).fill(0);
    for (let i = 0; i < n; i++) {
      const used = new Set();
      while (used.size < N_PICK) used.add(randrange(N_BALL));
      for (const b of used) freq[b]++;
    }
    const expected = n * N_PICK / N_BALL;  // 이론 기대값
    const rates = freq.map(f => f / n);
    const theoryRate = N_PICK / N_BALL;    // 6/45 ≈ 0.1333
    const maxDev = Math.max(...rates.map(r => Math.abs(r - theoryRate)));
    return { freq, expected, rates, theoryRate, maxDev, n };
  }

  /** 점진적 시뮬레이션: [10, 50, 100, 500, 1000, 5000, 10000] 각 시점의 최대 편차 */
  function convergenceDemo(steps = [10, 50, 100, 500, 1000, 5000, 10000]) {
    const freq = new Array(N_BALL).fill(0);
    const results = [];
    let done = 0;
    for (const target of steps) {
      while (done < target) {
        const used = new Set();
        while (used.size < N_PICK) used.add(randrange(N_BALL));
        for (const b of used) freq[b]++;
        done++;
      }
      const rates = freq.map(f => f / done);
      const theoryRate = N_PICK / N_BALL;
      const maxDev = Math.max(...rates.map(r => Math.abs(r - theoryRate)));
      const chi2 = freq.reduce((a, f) => a + (f - done * theoryRate) ** 2 / (done * theoryRate), 0);
      results.push({ n: done, maxDev: +maxDev.toFixed(6), chi2: +chi2.toFixed(2), rates: rates.slice() });
    }
    return results;
  }

  /* ── 3교시: 다중비교 함정 ── */
  /** k개 독립 검정(각각 진짜 p=uniform[0,1])에서 최소 하나가 α 이하일 확률 시뮬레이션 */
  function multipleComparisonSim(k = 15, alpha = 0.05, trials = 5000) {
    let falsePositive = 0;
    const minPs = [];
    for (let t = 0; t < trials; t++) {
      let minP = 1;
      for (let i = 0; i < k; i++) { const p = random(); if (p < minP) minP = p; }
      minPs.push(minP);
      if (minP < alpha) falsePositive++;
    }
    minPs.sort((a, b) => a - b);
    return {
      k, alpha, trials,
      falseRate: +(falsePositive / trials * 100).toFixed(1),       // % 거짓 양성
      theoryRate: +((1 - (1 - alpha) ** k) * 100).toFixed(1),      // 이론값: 1-(1-α)^k
      median_minP: +minPs[Math.floor(trials / 2)].toFixed(4),
    };
  }

  /** Holm-Bonferroni 보정 시연: 원 p-value 배열 → 보정 후 p-value + 유의 여부 */
  function holmDemo(rawPs, alpha = 0.05) {
    const m = rawPs.length;
    const indexed = rawPs.map((p, i) => ({ i, p })).sort((a, b) => a.p - b.p);
    const adj = new Array(m);
    let running = 0;
    for (let rank = 0; rank < m; rank++) {
      running = Math.max(running, indexed[rank].p * (m - rank));
      adj[indexed[rank].i] = Math.min(running, 1);
    }
    return rawPs.map((p, i) => ({
      raw: +p.toFixed(4),
      adjusted: +adj[i].toFixed(4),
      sigRaw: p < alpha,
      sigAdj: adj[i] < alpha,
    }));
  }

  /** "15개 검정 실험" 데모: 순수 무작위 데이터로 15개 p-value 생성 → 원/보정 비교 */
  function fakeTestBattery(k = 15) {
    const rawPs = Array.from({ length: k }, () => random());  // 귀무가설이 참 → p ~ Uniform(0,1)
    return { rawPs, holm: holmDemo(rawPs), sigRaw: rawPs.filter(p => p < 0.05).length };
  }

  return { diagnosePopularity, popPercentile, lawOfLargeNumbers, convergenceDemo,
           multipleComparisonSim, holmDemo, fakeTestBattery };
})();
if (typeof module !== 'undefined') module.exports = EDU;
