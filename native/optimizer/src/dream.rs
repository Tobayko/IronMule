//! Dream-RSI meta loop for the sequential learner (OPT1).
//!
//! Paper: Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds* (arXiv
//! 2609.14858, v2 2026-10-06). Its code is unreleased (github.com/zhengkid/Dream-RSI holds the paper
//! only, checked 2026-10-13), so this is a paper-based extension of `ironmule/dream.py`, not a port.
//!
//! What is kept from the paper: online exploration logs discovery trees; the stored trees become a
//! replay simulator; exploration-policy candidates are scored by replaying ("dreaming") instead of
//! new online runs; the chosen policy then runs online again and its new trees join the pool.
//!
//! Deviations, stated: (1) the trees are episodes of a sequential decision (each node a real slice:
//! state, action, reward, next state, cost, predecessor), not program-search trees; (2) the
//! meta-policy space is the learner's settings (step size, exploration schedule, initial value), not
//! an LLM-written exploration program; (3) replay is an empirical transition table: an action is
//! available in a replayed state only if that (state, action) pair was recorded there, and it yields
//! the recorded outcomes in turn. An unrecorded pair is unknown and masked, and every mask is
//! counted; nothing is interpolated and no learned world model predicts unrecorded outcomes;
//! (4) trees are split by episode seed into development (replay learning), validation (choice) and
//! the evaluation holdout, which this module never receives; (5) one comparison run selects once
//! (after half its budget) instead of iterating select -> run -> select; the trees it then records
//! join the journal for the next run's replay, so the loop closes across runs, not within one.
//! Replay scores are estimates for choosing what to run next; they never qualify or activate anything.

use crate::episode::Transition;
use crate::json::Value;
use crate::rl::{QConfig, QTable, State};
use std::collections::BTreeMap;

pub const START: State = State { level: 3, progress: 0, violated: false };
const MAX_STEPS: usize = 60;

#[derive(Clone, Debug, Default)]
pub struct ReplayModel {
    outcomes: BTreeMap<(u32, usize), Vec<(f64, Option<State>)>>,
    pub transitions: usize,
}

impl ReplayModel {
    pub fn build<'a>(episodes: impl IntoIterator<Item = &'a [Transition]>) -> ReplayModel {
        let mut m = ReplayModel::default();
        for ep in episodes {
            for t in ep {
                m.outcomes.entry((t.state.key(), t.action)).or_default().push((t.reward, t.next));
                m.transitions += 1;
            }
        }
        m
    }

    pub fn recorded(&self, s: State, actions: usize) -> Vec<usize> {
        (0..actions).filter(|a| self.outcomes.contains_key(&(s.key(), *a))).collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplayStats {
    /// Walks started.
    pub episodes: usize,
    /// Walks that reached a recorded episode end. Only these have a return.
    pub complete: usize,
    pub steps: usize,
    /// Unrecorded actions skipped, summed over every visited state.
    pub masked: usize,
    /// Walks stopped in a state where nothing was recorded.
    pub dead_ends: usize,
    /// Walks stopped at the step cap.
    pub capped: usize,
    /// Walks stopped because the evaluated policy's own choice was never recorded there: what follows
    /// would be the replay of another policy.
    pub off_data: usize,
}

impl ReplayStats {
    pub fn incomplete(&self) -> usize {
        self.episodes - self.complete
    }

    pub fn add(&mut self, o: &ReplayStats) {
        self.episodes += o.episodes;
        self.complete += o.complete;
        self.steps += o.steps;
        self.masked += o.masked;
        self.dead_ends += o.dead_ends;
        self.capped += o.capped;
        self.off_data += o.off_data;
    }

    pub fn to_json(&self) -> Value {
        Value::obj().with("episodes", self.episodes).with("complete", self.complete).with("incomplete", self.incomplete())
            .with("steps", self.steps).with("masked", self.masked).with("dead_ends", self.dead_ends)
            .with("capped", self.capped).with("off_data", self.off_data)
    }
}

/// Step through recorded outcomes: `pick(state, allowed)` chooses (None: its own choice is unrecorded);
/// outcomes cycle per (state, action). The return is known only for a walk that reaches a recorded
/// episode end: a dead end, the step cap or leaving the data is unknown, never a short (better) return.
fn walk(model: &ReplayModel, actions: usize, cursors: &mut BTreeMap<(u32, usize), usize>, stats: &mut ReplayStats,
        mut pick: impl FnMut(State, &[usize]) -> Option<usize>,
        mut learn: impl FnMut(State, usize, f64, Option<State>, &[usize])) -> Option<f64> {
    let (mut s, mut ret) = (START, 0.0);
    stats.episodes += 1;
    for _ in 0..MAX_STEPS {
        let allowed = model.recorded(s, actions);
        if allowed.is_empty() {
            stats.dead_ends += 1;
            return None;
        }
        stats.masked += actions - allowed.len();
        let Some(a) = pick(s, &allowed) else {
            stats.off_data += 1;
            return None;
        };
        let list = &model.outcomes[&(s.key(), a)];
        let c = cursors.entry((s.key(), a)).or_insert(0);
        let (r, next) = list[*c % list.len()];
        *c += 1;
        stats.steps += 1;
        let next_allowed = next.map(|n| model.recorded(n, actions)).unwrap_or_default();
        // A next state with nothing recorded has an unknown continuation: no target to learn from
        // (bootstrapping with 0 would be the most optimistic value); the walk dead-ends there next.
        if next.is_none() || !next_allowed.is_empty() {
            learn(s, a, r, next, &next_allowed);
        }
        ret += r;
        match next {
            Some(n) => s = n,
            None => {
                stats.complete += 1;
                return Some(ret);
            }
        }
    }
    stats.capped += 1;
    None
}

/// Dream: a fresh learner with `config` learns for `episodes` replayed episodes (exploring among the
/// recorded actions, bootstrapping over the next state's recorded actions).
pub fn replay_learn(config: QConfig, model: &ReplayModel, actions: usize, episodes: usize, seed: u64) -> (QTable, ReplayStats) {
    let mut table = QTable::new(config, actions, seed);
    let mut stats = ReplayStats::default();
    let mut cursors = BTreeMap::new();
    for _ in 0..episodes {
        let t = std::cell::RefCell::new(&mut table);
        walk(model, actions, &mut cursors, &mut stats,
             |s, allowed| Some(t.borrow_mut().choose(s, true, allowed)),
             |s, a, r, n, next_allowed| t.borrow_mut().update_within(s, a, r, n, next_allowed));
        table.end_episode();
    }
    (table, stats)
}

/// Replay estimate of a frozen greedy policy's return on `model`: the mean over walks, or None unless
/// every walk was complete (the policy's own greedy choice must have been recorded at every step).
pub fn replay_value(table: &QTable, model: &ReplayModel, actions: usize, episodes: usize) -> (Option<f64>, ReplayStats) {
    let mut stats = ReplayStats::default();
    let mut cursors = BTreeMap::new();
    let mut total = 0.0;
    for _ in 0..episodes {
        if let Some(r) = walk(model, actions, &mut cursors, &mut stats, |s, allowed| {
            let a = table.greedy(s);
            allowed.contains(&a).then_some(a)
        }, |_, _, _, _, _| {}) {
            total += r;
        }
    }
    let known = stats.incomplete() == 0 && episodes > 0;
    (known.then(|| total / episodes as f64), stats)
}

/// The meta-candidates (fixed in OPT1): step size x exploration start x initial value.
pub fn candidates() -> Vec<QConfig> {
    let mut out = Vec::new();
    for alpha in [0.1, 0.3, 0.6] {
        for epsilon_start in [0.3, 0.6] {
            for initial_q in [0.0, -10.0] {
                out.push(QConfig { alpha, epsilon_start, epsilon_end: 0.05, epsilon_episodes: 40.0, initial_q });
            }
        }
    }
    out
}

/// Choose learner settings: learn on development trees, score on validation trees, best mean wins.
/// Only fully supported candidates compete (every validation walk complete for every seed); if none is,
/// nothing is chosen (None) and the caller keeps its current settings. Support is in the report.
pub fn select(dev: &ReplayModel, val: &ReplayModel, actions: usize, learn_episodes: usize, value_episodes: usize,
              seeds: &[u64]) -> (Option<QConfig>, Value) {
    let mut rows = Vec::new();
    let mut best: Option<(QConfig, f64)> = None;
    for c in candidates() {
        let (mut learn_stats, mut value_stats) = (ReplayStats::default(), ReplayStats::default());
        let scores: Vec<Option<f64>> = seeds.iter().map(|&seed| {
            let (table, ls) = replay_learn(c, dev, actions, learn_episodes, seed);
            let (v, vs) = replay_value(&table, val, actions, value_episodes);
            learn_stats.add(&ls);
            value_stats.add(&vs);
            v
        }).collect();
        let supported = !scores.is_empty() && scores.iter().all(Option::is_some);
        let mean = supported.then(|| scores.iter().flatten().sum::<f64>() / scores.len() as f64);
        rows.push(Value::obj().with("config", c.to_json()).with("supported", supported).with("validation_replay_return", mean)
            .with("per_seed", scores).with("learn_replay", learn_stats.to_json()).with("value_replay", value_stats.to_json()));
        if let Some(m) = mean {
            if best.map_or(true, |(_, b)| m > b) {
                best = Some((c, m));
            }
        }
    }
    let supported = rows.iter().filter(|r| r.boolean("supported") == Some(true)).count();
    (best.map(|b| b.0), Value::obj().with("chosen", best.map(|b| b.0.to_json())).with("chosen_validation_replay_return", best.map(|b| b.1))
        .with("supported_candidates", supported).with("candidates", rows)
        .with("note", "replay estimates from recorded transitions only; not measurements; unsupported candidates cannot be chosen"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(state: State, action: usize, reward: f64, next: Option<State>) -> Transition {
        Transition { t: 0, state, action, reward, next, level: 0, p95_ms: None, service_p95_ms: None, violated: false, chunks: 1, slice_s: 0.5 }
    }

    #[test]
    fn replay_uses_only_recorded_outcomes_and_counts_masks() {
        let mid = State { level: 0, progress: 2, violated: false };
        // recorded: from START action 1 -> mid (-1), mid action 0 -> end (-1); action 0 from START -> end (-5)
        let ep1 = vec![tr(START, 1, -1.0, Some(mid)), tr(mid, 0, -1.0, None)];
        let ep2 = vec![tr(START, 0, -5.0, None)];
        let model = ReplayModel::build([&ep1[..], &ep2[..]]);
        assert_eq!(model.transitions, 3);
        assert_eq!(model.recorded(mid, 3), vec![0]);
        let (table, stats) = replay_learn(crate::rl::DEFAULT, &model, 3, 50, 11);
        assert!(stats.masked > 0, "action 2 was never recorded anywhere: it must be masked");
        assert_eq!(table.visits.values().map(|v| v[2]).sum::<u64>(), 0, "an unrecorded action is never taken");
        // optimistic Q0 = 0: the greedy choice in `mid` is an untried action, which replay cannot follow
        let (value, s) = replay_value(&table, &model, 3, 4);
        assert_eq!((value, s.off_data), (None, 4));
        let (table, _) = replay_learn(QConfig { initial_q: -10.0, ..crate::rl::DEFAULT }, &model, 3, 50, 11);
        let (value, _) = replay_value(&table, &model, 3, 4);
        assert_eq!(value, Some(-2.0)); // the learned greedy path START -1-> mid -1-> end
        // an unreachable region is a dead end with no return, not an invented (short) outcome
        let lone = ReplayModel::build([&ep1[..1]]);
        let (v, s) = replay_value(&table, &lone, 3, 1);
        assert_eq!((v, s.dead_ends, s.complete), (None, 1, 0));
        let (chosen, report) = select(&model, &model, 3, 30, 4, &[1, 2]);
        assert!(candidates().contains(&chosen.unwrap()));
        assert_eq!(report.arr("candidates").unwrap().len(), candidates().len());
    }

    /// OPT1 scored a walk that left the data by its partial return: -1 for one step beats -2 for a
    /// finished episode. A missing continuation is unknown, so that policy is unsupported, never better.
    #[test]
    fn a_missing_continuation_is_unknown_and_never_wins_by_a_shorter_return() {
        let mid = State { level: 0, progress: 2, violated: false };
        let other = State { level: 1, progress: 2, violated: false };
        let finished = vec![tr(START, 0, -1.0, Some(mid)), tr(mid, 0, -1.0, None)];
        let cut = vec![tr(START, 1, -1.0, Some(other))]; // `other` never recorded: no continuation
        let model = ReplayModel::build([&finished[..], &cut[..]]);
        let mut leaves = QTable::new(crate::rl::DEFAULT, 2, 1);
        leaves.update(START, 1, 10.0, None); // greedy action 1 at START, into the unrecorded region
        let (v, s) = replay_value(&leaves, &model, 2, 3);
        assert_eq!((v, s.dead_ends, s.incomplete()), (None, 3, 3));
        let mut stays = QTable::new(crate::rl::DEFAULT, 2, 1);
        stays.update(START, 0, 10.0, None);
        assert_eq!(replay_value(&stays, &model, 2, 3).0, Some(-2.0));
        // the policy's own choice unrecorded: off the data, unknown (OPT1 silently took the best recorded one)
        let mut off = QTable::new(crate::rl::DEFAULT, 3, 1);
        off.update(START, 2, 10.0, None);
        let (v, s) = replay_value(&off, &model, 3, 2);
        assert_eq!((v, s.off_data), (None, 2));
        // the step cap is unknown too: a self-loop never reaches an episode end
        let looped = vec![tr(START, 0, -1.0, Some(START))];
        let (v, s) = replay_value(&stays, &ReplayModel::build([&looped[..]]), 2, 1);
        assert_eq!((v, s.capped), (None, 1));
        // learning in replay: a step into a state with nothing recorded has no target (0 would be optimistic)
        let (learned, _) = replay_learn(crate::rl::DEFAULT, &model, 2, 40, 3);
        assert_eq!(learned.visits.get(&START.key()).map_or(0, |v| v[1]), 0, "{:?}", learned.visits);
        // with nothing supported, nothing is chosen
        let (chosen, report) = select(&model, &ReplayModel::build([&cut[..]]), 2, 10, 2, &[1]);
        assert_eq!((chosen, report.num("supported_candidates")), (None, Some(0.0)));
    }
}
