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

  /* ── 4교시: 승자의 저주 ── */
  /** nProg개 "프로그램"이 각각 nRounds회 예측. 실력은 전부 동일(무작위).
   *  1기에서 가장 잘 맞춘 프로그램을 "승자"로 선택 → 2기 성적을 비교.
   *  승자의 저주: 1기 성적은 부풀려져 있고, 2기에서는 평균으로 회귀. */
  function winnersCurse(nProg = 7, nRounds = 50, trials = 500) {
    const theory = N_PICK * N_PICK / N_BALL;  // 0.8
    let sumWin1 = 0, sumWin2 = 0, sumAvg1 = 0, sumAvg2 = 0;
    for (let t = 0; t < trials; t++) {
      const phase1 = [], phase2 = [];
      for (let p = 0; p < nProg; p++) {
        let s1 = 0, s2 = 0;
        for (let r = 0; r < nRounds; r++) {
          const pred = new Set(); while (pred.size < N_PICK) pred.add(randrange(N_BALL) + 1);
          const actual = new Set(); while (actual.size < N_PICK) actual.add(randrange(N_BALL) + 1);
          s1 += [...pred].filter(n => actual.has(n)).length;
        }
        for (let r = 0; r < nRounds; r++) {
          const pred = new Set(); while (pred.size < N_PICK) pred.add(randrange(N_BALL) + 1);
          const actual = new Set(); while (actual.size < N_PICK) actual.add(randrange(N_BALL) + 1);
          s2 += [...pred].filter(n => actual.has(n)).length;
        }
        phase1.push(s1 / nRounds); phase2.push(s2 / nRounds);
      }
      const winIdx = phase1.indexOf(Math.max(...phase1));
      sumWin1 += phase1[winIdx]; sumWin2 += phase2[winIdx];
      sumAvg1 += phase1.reduce((a, b) => a + b, 0) / nProg;
      sumAvg2 += phase2.reduce((a, b) => a + b, 0) / nProg;
    }
    return {
      nProg, nRounds, trials, theory,
      winner_phase1: +(sumWin1 / trials).toFixed(3),
      winner_phase2: +(sumWin2 / trials).toFixed(3),
      average_phase1: +(sumAvg1 / trials).toFixed(3),
      average_phase2: +(sumAvg2 / trials).toFixed(3),
      regression: +((sumWin1 - sumWin2) / trials).toFixed(3),
    };
  }

  /* ── 5교시: 외삽의 위험 ── */
  /** 과거 nTrain회로 빈도 점수 상위 k개를 뽑고, 이후 nTest회에서 일치율 측정.
   *  비교 대상: 무작위 k개. 과거 패턴이 미래에 이어지지 않음을 보여줌. */
  function extrapolationRisk(nTrain = 200, nTest = 100, k = 12, trials = 300) {
    let sumModel = 0, sumRandom = 0;
    for (let t = 0; t < trials; t++) {
      // 훈련: nTrain회 추첨 시뮬레이션
      const freq = new Array(N_BALL).fill(0);
      for (let r = 0; r < nTrain; r++) {
        const d = new Set(); while (d.size < N_PICK) d.add(randrange(N_BALL)); for (const b of d) freq[b]++;
      }
      const topK = freq.map((f, i) => [f, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(p => p[1]);
      const randK = []; const used = new Set();
      while (randK.length < k) { const r = randrange(N_BALL); if (!used.has(r)) { used.add(r); randK.push(r); } }
      // 테스트: nTest회에서 각 조합의 번호가 나온 비율
      let mHit = 0, rHit = 0;
      for (let r = 0; r < nTest; r++) {
        const d = new Set(); while (d.size < N_PICK) d.add(randrange(N_BALL));
        for (const b of topK) if (d.has(b)) mHit++;
        for (const b of randK) if (d.has(b)) rHit++;
      }
      sumModel += mHit / (nTest * k); sumRandom += rHit / (nTest * k);
    }
    return {
      nTrain, nTest, k, trials,
      theoryRate: +(N_PICK / N_BALL).toFixed(4),
      modelRate: +(sumModel / trials).toFixed(4),
      randomRate: +(sumRandom / trials).toFixed(4),
      difference: +((sumModel - sumRandom) / trials).toFixed(4),
    };
  }

  /* ── 6교시: 몬테카를로 생애 시뮬레이터 ── */
  /** years년간 매주 spend원씩 구매 시뮬레이션.
   *  당첨 확률: 5등(3개) 1/45×..., 4등(4개), 3등(5개), 2등(5+보너스), 1등(6개)
   *  반환: 총 지출, 총 당첨금, 순이익, 등수별 당첨 횟수 */
  function lifetimeSim(years = 30, weeklySpend = 10000, gamesPerWeek = 10) {
    const weeks = years * 52;
    const totalSpent = weeks * weeklySpend;
    let totalWon = 0;
    const wins = { '5등(3개)': 0, '4등(4개)': 0, '3등(5개)': 0, '2등(5+보너스)': 0, '1등(6개)': 0 };
    const prizes = { 3: 5000, 4: 50000, 5: 1500000, '5b': 30000000, 6: 2000000000 };

    for (let w = 0; w < weeks; w++) {
      // 매주 실제 추첨: 6개 + 보너스 1개
      const drum = []; for (let i = 1; i <= 45; i++) drum.push(i);
      for (let i = 44; i > 0; i--) { const j = randrange(i + 1); [drum[i], drum[j]] = [drum[j], drum[i]]; }
      const actual = new Set(drum.slice(0, 6));
      const bonus = drum[6];

      for (let g = 0; g < gamesPerWeek; g++) {
        const pick = new Set(); while (pick.size < 6) pick.add(randrange(45) + 1);
        const match = [...pick].filter(n => actual.has(n)).length;
        if (match === 6) { wins['1등(6개)']++; totalWon += prizes[6]; }
        else if (match === 5 && pick.has(bonus)) { wins['2등(5+보너스)']++; totalWon += prizes['5b']; }
        else if (match === 5) { wins['3등(5개)']++; totalWon += prizes[5]; }
        else if (match === 4) { wins['4등(4개)']++; totalWon += prizes[4]; }
        else if (match === 3) { wins['5등(3개)']++; totalWon += prizes[3]; }
      }
    }
    return {
      years, weeklySpend, gamesPerWeek, weeks,
      totalSpent, totalWon,
      netProfit: totalWon - totalSpent,
      returnRate: +((totalWon / totalSpent) * 100).toFixed(1),
      wins,
    };
  }

  /** 여러 번 생애 시뮬레이션 → 분포 요약 */
  function lifetimeDistribution(years = 30, weeklySpend = 10000, sims = 100) {
    const results = [];
    for (let i = 0; i < sims; i++) results.push(lifetimeSim(years, weeklySpend));
    const nets = results.map(r => r.netProfit).sort((a, b) => a - b);
    const returns = results.map(r => r.returnRate);
    const anyBig = results.filter(r => r.wins['1등(6개)'] > 0 || r.wins['2등(5+보너스)'] > 0).length;
    return {
      sims, years, weeklySpend,
      totalSpent: results[0].totalSpent,
      medianNet: nets[Math.floor(sims / 2)],
      worstNet: nets[0],
      bestNet: nets[nets.length - 1],
      meanReturn: +(returns.reduce((a, b) => a + b, 0) / sims).toFixed(1),
      medianReturn: +returns.sort((a, b) => a - b)[Math.floor(sims / 2)].toFixed(1),
      bigWinners: anyBig,
      p1st: +(1 - (1 - 1 / 8145060) ** (years * 52 * 10)).toFixed(6),
    };
  }

  return { diagnosePopularity, popPercentile, lawOfLargeNumbers, convergenceDemo,
           multipleComparisonSim, holmDemo, fakeTestBattery,
           winnersCurse, extrapolationRisk, lifetimeSim, lifetimeDistribution };
})();
if (typeof module !== 'undefined') module.exports = EDU;
