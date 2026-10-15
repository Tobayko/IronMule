//! Independent qualification: the online controller's strict A/A/B rule (protocol 1, CTRL7), ported
//! unchanged so the coordinator's verdicts mean the same thing. The policy under test cannot reach
//! this module's thresholds: they come from the frozen goal, and the evaluator only sees measured
//! costs (lower is better) and correctness flags.
//!
//! For each complete triplet (reference A, candidate B, reference AA; order alternated by the
//! caller): `R = B / mean(A, AA)`, `N = |A - AA| / mean(A, AA)`. Any correctness failure, any
//! `R > 1 + max_regression` or `N > max_regression` rejects at once. A window wins when
//! `R + N < 1 - min_gain`. At the fixed sample count the exact one-sided Clopper-Pearson upper bound
//! of the loss frequency (alpha = family_alpha / max_trials) must be below 0.5, and
//! `mean(R) + mean(N) < 1 - min_gain` must hold. No early promotion, no retries.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    pub min_gain: f64,
    pub max_regression: f64,
    pub pairs: u32,
    pub family_alpha: f64,
    pub max_trials: u32,
}

impl Rule {
    pub fn valid(&self) -> bool {
        self.min_gain.is_finite() && self.min_gain >= 0.0 && self.min_gain < 1.0
            && self.max_regression.is_finite() && self.max_regression > 0.0 && self.max_regression < 1.0
            && self.pairs >= 1 && self.family_alpha > 0.0 && self.family_alpha < 1.0 && self.max_trials >= 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triplet {
    pub a: f64,
    pub b: f64,
    pub aa: f64,
    pub correct: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Not enough complete triplets yet.
    Incomplete,
    Qualified { mean_ratio: f64, mean_noise: f64, upper_bound: f64 },
    Rejected { reason: String, mean_ratio: f64, mean_noise: f64, upper_bound: f64 },
}

pub fn binomial_cdf(failures: u32, n: u32, probability: f64) -> f64 {
    if failures >= n || probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return 0.0;
    }
    let mut log_term = f64::from(n) * (-probability).ln_1p();
    let mut log_sum = log_term;
    let log_odds = probability.ln() - (-probability).ln_1p();
    for k in 0..failures {
        log_term += f64::from(n - k).ln() - f64::from(k + 1).ln() + log_odds;
        let maximum = log_term.max(log_sum);
        log_sum = maximum + ((log_term - maximum).exp() + (log_sum - maximum).exp()).ln();
    }
    log_sum.exp().min(1.0)
}

pub fn failure_upper(failures: u32, n: u32, alpha: f64) -> f64 {
    if failures == n {
        return 1.0;
    }
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..64 {
        let middle = (low + high) * 0.5;
        if binomial_cdf(failures, n, middle) > alpha {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

pub fn judge(rule: &Rule, triplets: &[Triplet]) -> Verdict {
    let (mut rs, mut ns) = (Vec::new(), Vec::new());
    for t in triplets {
        let base = 0.5 * (t.a + t.aa);
        let finite = [t.a, t.b, t.aa].iter().all(|x| x.is_finite() && *x > 0.0);
        if !t.correct || !finite {
            return Verdict::Rejected { reason: if t.correct { "invalid measurement" } else { "correctness failed" }.into(),
                                       mean_ratio: f64::NAN, mean_noise: f64::NAN, upper_bound: 1.0 };
        }
        let (r, n) = (t.b / base, (t.a - t.aa).abs() / base);
        if r > 1.0 + rule.max_regression {
            return Verdict::Rejected { reason: format!("regression: R = {r:.4}"), mean_ratio: r, mean_noise: n, upper_bound: 1.0 };
        }
        if n > rule.max_regression {
            return Verdict::Rejected { reason: format!("noisy A/A: N = {n:.4}"), mean_ratio: r, mean_noise: n, upper_bound: 1.0 };
        }
        rs.push(r);
        ns.push(n);
    }
    if (rs.len() as u32) < rule.pairs {
        return Verdict::Incomplete;
    }
    let count = rule.pairs as usize;
    let (rs, ns) = (&rs[..count], &ns[..count]);
    let failures = rs.iter().zip(ns).filter(|(r, n)| *r + *n >= 1.0 - rule.min_gain).count() as u32;
    let mean_ratio = rs.iter().sum::<f64>() / count as f64;
    let mean_noise = ns.iter().sum::<f64>() / count as f64;
    let upper_bound = failure_upper(failures, rule.pairs, rule.family_alpha / f64::from(rule.max_trials));
    if upper_bound < 0.5 && mean_ratio + mean_noise < 1.0 - rule.min_gain {
        Verdict::Qualified { mean_ratio, mean_noise, upper_bound }
    } else {
        Verdict::Rejected { reason: "no qualified gain".into(), mean_ratio, mean_noise, upper_bound }
    }
}

/// Protocol 2 (backlog OPT4, explicit opt-in by `"protocol": 2` in a goal; protocol 1 above is
/// unchanged). Same triplets, correctness and regression vetoes, window rule and Clopper-Pearson
/// bound. Changed: a triplet's A/A noise no longer vetoes unless `N > 0.5` (a machine that changes
/// by half within one triplet); a noisy window simply has to win with its noise (`R + N`); the mean
/// condition is `mean(R) + 2 mean(N) < 1 - min_gain`; and a two-sided sign test of the material
/// `AA - A` differences (`|AA - A| / mean(A, AA) >= min_gain`; AA always runs last) over the judged
/// triplets with `p < 0.05` makes the verdict "invalid: drift" (a tiny steady order effect is not
/// drift). Judged on exactly the first `pairs` triplets.
///
/// Known structural limit (analytic, OPT4 review): a load that slows both reference runs of a
/// triplet alike and spares the candidate run is indistinguishable from a real gain for every rule
/// on these statistics (R falls, N and `AA - A` stay at zero); only the measurement design (load
/// independent of the arm) can exclude it.
pub fn judge_v2(rule: &Rule, triplets: &[Triplet]) -> Verdict {
    const HARD_NOISE: f64 = 0.5;
    if (triplets.len() as u32) < rule.pairs {
        // Vetoes before completion still apply, so a production cycle can stop early.
        for t in triplets {
            if let Some(v) = veto_v2(rule, t, HARD_NOISE) {
                return v;
            }
        }
        return Verdict::Incomplete;
    }
    let judged = &triplets[..rule.pairs as usize];
    let (mut rs, mut ns) = (Vec::new(), Vec::new());
    for t in judged {
        if let Some(v) = veto_v2(rule, t, HARD_NOISE) {
            return v;
        }
        let base = 0.5 * (t.a + t.aa);
        rs.push(t.b / base);
        ns.push((t.a - t.aa).abs() / base);
    }
    let count = judged.len();
    let mean_ratio = rs.iter().sum::<f64>() / count as f64;
    let mean_noise = ns.iter().sum::<f64>() / count as f64;
    let material = |t: &&Triplet| (t.aa - t.a).abs() / (0.5 * (t.a + t.aa)) >= rule.min_gain;
    let (up, down) = (judged.iter().filter(material).filter(|t| t.aa > t.a).count() as u32,
                      judged.iter().filter(material).filter(|t| t.aa < t.a).count() as u32);
    let drift_p = (2.0 * sign_test_p(up.max(down), up + down)).min(1.0);
    if drift_p < 0.05 {
        return Verdict::Rejected { reason: format!("invalid: drift (AA - A sign test p = {drift_p:.4})"), mean_ratio, mean_noise, upper_bound: 1.0 };
    }
    let failures = rs.iter().zip(&ns).filter(|(r, n)| *r + *n >= 1.0 - rule.min_gain).count() as u32;
    let upper_bound = failure_upper(failures, rule.pairs, rule.family_alpha / f64::from(rule.max_trials));
    if upper_bound < 0.5 && mean_ratio + 2.0 * mean_noise < 1.0 - rule.min_gain {
        Verdict::Qualified { mean_ratio, mean_noise, upper_bound }
    } else {
        Verdict::Rejected { reason: "no qualified gain".into(), mean_ratio, mean_noise, upper_bound }
    }
}

fn veto_v2(rule: &Rule, t: &Triplet, hard_noise: f64) -> Option<Verdict> {
    let base = 0.5 * (t.a + t.aa);
    let finite = [t.a, t.b, t.aa].iter().all(|x| x.is_finite() && *x > 0.0);
    if !t.correct || !finite {
        return Some(Verdict::Rejected { reason: if t.correct { "invalid measurement" } else { "correctness failed" }.into(),
                                        mean_ratio: f64::NAN, mean_noise: f64::NAN, upper_bound: 1.0 });
    }
    let (r, n) = (t.b / base, (t.a - t.aa).abs() / base);
    if r > 1.0 + rule.max_regression {
        return Some(Verdict::Rejected { reason: format!("regression: R = {r:.4}"), mean_ratio: r, mean_noise: n, upper_bound: 1.0 });
    }
    (n > hard_noise).then(|| Verdict::Rejected { reason: format!("unstable machine: N = {n:.4}"), mean_ratio: r, mean_noise: n, upper_bound: 1.0 })
}

/// The goal's protocol: 1 (default, unchanged) or 2; any other value is refused when the goal loads.
pub fn judge_protocol(protocol: u32, rule: &Rule, triplets: &[Triplet]) -> Verdict {
    match protocol {
        2 => judge_v2(rule, triplets),
        _ => judge(rule, triplets),
    }
}

/// One-sided exact sign test: P(at least `wins` of `n` | p = 1/2).
pub fn sign_test_p(wins: u32, n: u32) -> f64 {
    if wins == 0 {
        return 1.0;
    }
    1.0 - binomial_cdf(wins - 1, n, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: Rule = Rule { min_gain: 0.03, max_regression: 0.10, pairs: 8, family_alpha: 0.05, max_trials: 4 };

    fn t(a: f64, b: f64, aa: f64) -> Triplet {
        Triplet { a, b, aa, correct: true }
    }

    #[test]
    fn a_clear_gain_qualifies_and_everything_else_does_not() {
        let fast: Vec<Triplet> = (0..8).map(|_| t(10.0, 8.0, 10.1)).collect();
        assert!(matches!(judge(&RULE, &fast), Verdict::Qualified { .. }));
        assert_eq!(judge(&RULE, &fast[..5]), Verdict::Incomplete);
        let tie: Vec<Triplet> = (0..8).map(|_| t(10.0, 9.9, 10.0)).collect();
        assert!(matches!(judge(&RULE, &tie), Verdict::Rejected { .. }));
        let mut wrong = fast.clone();
        wrong[3].correct = false;
        assert!(matches!(judge(&RULE, &wrong), Verdict::Rejected { reason, .. } if reason == "correctness failed"));
        let mut slow = fast.clone();
        slow[0] = t(10.0, 11.5, 10.0);
        assert!(matches!(judge(&RULE, &slow), Verdict::Rejected { reason, .. } if reason.starts_with("regression")));
        let mut noisy = fast.clone();
        noisy[1] = t(10.0, 8.0, 12.0);
        assert!(matches!(judge(&RULE, &noisy), Verdict::Rejected { reason, .. } if reason.starts_with("noisy")));
    }

    // ---- OPT4 review: synthetic evaluator checks. Constructed numbers, not hardware evidence. ----

    fn both(ts: &[Triplet]) -> (Verdict, Verdict) {
        (judge(&RULE, ts), judge_v2(&RULE, ts))
    }

    fn qualified(v: &Verdict) -> bool {
        matches!(v, Verdict::Qualified { .. })
    }

    /// Structural limit of every rule on (A, B, AA): load that slows both reference runs alike and
    /// spares the candidate looks exactly like a 5 % gain (R 0.95, N 0, no AA - A difference).
    #[test]
    fn synthetic_a_uniform_reference_slowdown_passes_both_protocols_as_a_gain() {
        let ts: Vec<Triplet> = (0..8).map(|_| t(10.5, 10.0, 10.5)).collect();
        let (p1, p2) = both(&ts);
        assert!(qualified(&p1) && qualified(&p2), "{p1:?} {p2:?}");
    }

    /// OPT4's first sub-threshold arm ("1 % fewer chunks") changes the output: it is a correctness
    /// failure in both protocols and never reaches the gain rule. It cannot be marked correct.
    #[test]
    fn synthetic_a_fewer_chunks_arm_is_a_correctness_failure_not_a_sub_threshold_gain() {
        let ts: Vec<Triplet> = (0..8).map(|_| Triplet { a: 10.0, b: 9.9, aa: 10.0, correct: false }).collect();
        for v in [judge(&RULE, &ts), judge_v2(&RULE, &ts)] {
            assert!(matches!(&v, Verdict::Rejected { reason, .. } if reason == "correctness failed"), "{v:?}");
        }
        // the same 1 % with equal outputs (e.g. a padded reference) is below min_gain in both
        let ok: Vec<Triplet> = (0..8).map(|_| t(10.0, 9.9, 10.0)).collect();
        let (p1, p2) = both(&ok);
        assert!(!qualified(&p1) && !qualified(&p2));
    }

    /// The case OPT4 is about: a large gain with one noisy triplet (N 0.3) is vetoed by protocol 1
    /// and qualified by protocol 2, which still needs that window to win with its noise.
    #[test]
    fn synthetic_a_large_gain_with_one_noisy_triplet() {
        let mut ts: Vec<Triplet> = (0..8).map(|k| t(10.0, 2.0, 10.0 + 0.05 * (k % 2) as f64)).collect();
        ts[2] = t(10.0, 2.0, 13.5);
        let (p1, p2) = both(&ts);
        assert!(matches!(&p1, Verdict::Rejected { reason, .. } if reason.starts_with("noisy")), "{p1:?}");
        assert!(qualified(&p2), "{p2:?}");
        // a window whose noise eats the gain still loses: N 0.45 with R 0.6
        let mut lose = ts.clone();
        lose[2] = t(10.0, 7.0, 16.0);
        assert!(!qualified(&judge_v2(&RULE, &lose)));
        // above half, the machine is unstable
        ts[2] = t(10.0, 2.0, 18.0);
        assert!(matches!(judge_v2(&RULE, &ts), Verdict::Rejected { reason, .. } if reason.starts_with("unstable")));
    }

    /// A machine slowing 10 % per run, true effect null, protocol 1's alternating order: b-first
    /// windows show a false gain. Protocol 1 rejects (its noise veto); protocol 2 calls it drift.
    #[test]
    fn synthetic_slowing_drift_is_never_qualified() {
        let d = 0.10;
        let ts: Vec<Triplet> = (0..8).map(|k| if k % 2 == 0 { t(1.0, 1.0 + d, 1.0 + 2.0 * d) } else { t(1.0 + d, 1.0, 1.0 + 2.0 * d) }).collect();
        let (p1, p2) = both(&ts);
        assert!(!qualified(&p1));
        assert!(matches!(&p2, Verdict::Rejected { reason, .. } if reason.starts_with("invalid: drift")), "{p2:?}");
        // a tiny steady order effect (AA 0.5 % faster) is not drift: a large gain still qualifies
        let steady: Vec<Triplet> = (0..8).map(|_| t(10.0, 2.0, 9.95)).collect();
        assert!(qualified(&judge_v2(&RULE, &steady)));
        assert_eq!(judge_protocol(1, &RULE, &steady), judge(&RULE, &steady));
    }

    /// Protocol 2 judges exactly `pairs` triplets; an early veto still stops a production cycle.
    #[test]
    fn synthetic_protocol_two_judges_a_fixed_count_and_keeps_early_vetoes() {
        let mut ts: Vec<Triplet> = (0..8).map(|_| t(10.0, 2.0, 10.0)).collect();
        assert_eq!(judge_v2(&RULE, &ts[..5]), Verdict::Incomplete);
        ts[1].correct = false;
        assert!(matches!(judge_v2(&RULE, &ts[..2]), Verdict::Rejected { .. }));
        let mut extra: Vec<Triplet> = (0..8).map(|_| t(10.0, 2.0, 10.0)).collect();
        extra.push(t(10.0, 50.0, 10.0)); // a ninth triplet is not judged
        assert!(qualified(&judge_v2(&RULE, &extra)));
    }

    #[test]
    fn binomial_helpers_match_known_values() {
        assert!((binomial_cdf(2, 5, 0.5) - 0.5).abs() < 1e-12);
        assert!((sign_test_p(15, 20) - 0.020_694_732_666).abs() < 1e-9);
        assert!((failure_upper(0, 8, 0.0125) - (1.0 - 0.0125f64.powf(1.0 / 8.0))).abs() < 1e-9);
    }
}
