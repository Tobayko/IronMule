//! Sequential RL candidate (OPT1): tabular Q-learning, undiscounted, episodic.
//!
//! This is reinforcement learning, not a bandit: the update bootstraps from the next state's value,
//! so an action is credited with what it changes later (here: finishing the background job earlier
//! shortens its exposure to later foreground load). A contextual bandit would use the immediate
//! reward only. State, action, transition, episode end and return are fixed in backlog OPT1 before
//! any data: state = (foreground level seen in the previous slice: idle, light, heavy or none yet;
//! job progress in quarters; previous slice violated the foreground target), action = an index into
//! the bounded thread set, reward per slice = -1 - penalty x violation, return = sum (no discount).

use crate::json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct State {
    pub level: u8,    // 0 idle, 1 light, 2 heavy, 3 not yet observed
    pub progress: u8, // 0..=3 quarters done
    pub violated: bool,
}

impl State {
    pub fn key(self) -> u32 {
        (self.level as u32) * 8 + (self.progress as u32) * 2 + self.violated as u32
    }

    pub fn from_key(k: u32) -> State {
        State { level: (k / 8) as u8, progress: ((k / 2) % 4) as u8, violated: k % 2 == 1 }
    }

    pub fn to_json(self) -> Value {
        Value::obj().with("level", self.level as u64).with("progress", self.progress as u64).with("violated", self.violated)
    }

    pub fn from_json(v: &Value) -> Option<State> {
        Some(State { level: v.num("level")? as u8, progress: v.num("progress")? as u8, violated: v.boolean("violated")? })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QConfig {
    pub alpha: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    /// Episodes over which epsilon falls linearly from start to end.
    pub epsilon_episodes: f64,
    pub initial_q: f64,
}

impl QConfig {
    pub fn to_json(self) -> Value {
        Value::obj().with("alpha", self.alpha).with("epsilon_start", self.epsilon_start).with("epsilon_end", self.epsilon_end)
            .with("epsilon_episodes", self.epsilon_episodes).with("initial_q", self.initial_q)
    }

    pub fn from_json(v: &Value) -> Option<QConfig> {
        Some(QConfig { alpha: v.num("alpha")?, epsilon_start: v.num("epsilon_start")?, epsilon_end: v.num("epsilon_end")?,
                       epsilon_episodes: v.num("epsilon_episodes")?, initial_q: v.num("initial_q")? })
    }
}

/// splitmix64 finaliser: a bijection, so distinct seeds give distinct, well-mixed generator states
/// (OPT1 used `seed | 1`, which gave seeds 2k and 2k + 1 the same stream). Never 0 (xorshift's fixed point).
pub fn stream(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    match z ^ (z >> 31) { 0 => 0x2545_F491_4F6C_DD1D, x => x }
}

pub const DEFAULT: QConfig = QConfig { alpha: 0.3, epsilon_start: 0.5, epsilon_end: 0.05, epsilon_episodes: 40.0, initial_q: 0.0 };

#[derive(Clone, Debug, PartialEq)]
pub struct QTable {
    pub config: QConfig,
    pub actions: usize,
    pub q: BTreeMap<u32, Vec<f64>>,
    pub visits: BTreeMap<u32, Vec<u64>>,
    pub episodes: u64,
    pub updates: u64,
    rng: u64,
    /// OPT2: learn from the dense reward (`episode::dense_reward`) instead of the slice's sparse one.
    pub dense: bool,
    /// OPT2 warm start: initial values per action in every unvisited state (the sweep's mean sparse return
    /// per fixed action), so the sweep's best is greedy and the others sit on the same value scale.
    pub prior: Option<Vec<f64>>,
}

impl QTable {
    pub fn new(config: QConfig, actions: usize, seed: u64) -> QTable {
        QTable { config, actions, q: BTreeMap::new(), visits: BTreeMap::new(), episodes: 0, updates: 0, rng: stream(seed),
                 dense: false, prior: None }
    }

    /// The exploration generator's state (saved with each campaign episode).
    pub fn rng_state(&self) -> u64 {
        self.rng
    }

    /// Restart the exploration stream from `seed` (a campaign attempt gets its own stream).
    pub fn reseed(&mut self, seed: u64) {
        self.rng = stream(seed);
    }

    fn random(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn values(&self, s: State) -> Vec<f64> {
        self.q.get(&s.key()).cloned().unwrap_or_else(|| self.initial_row())
    }

    fn initial_row(&self) -> Vec<f64> {
        self.prior.clone().unwrap_or_else(|| vec![self.config.initial_q; self.actions])
    }

    pub fn epsilon(&self) -> f64 {
        let f = (self.episodes as f64 / self.config.epsilon_episodes.max(1.0)).min(1.0);
        self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * f
    }

    /// Greedy (ties to the lower index, i.e. fewer threads) or epsilon-greedy among `allowed`.
    pub fn choose(&mut self, s: State, explore: bool, allowed: &[usize]) -> usize {
        assert!(!allowed.is_empty());
        if explore && self.random() < self.epsilon() {
            let i = (self.random() * allowed.len() as f64) as usize;
            return allowed[i.min(allowed.len() - 1)];
        }
        let v = self.values(s);
        *allowed.iter().max_by(|&&a, &&b| v[a].partial_cmp(&v[b]).unwrap().then(b.cmp(&a))).unwrap()
    }

    pub fn greedy(&self, s: State) -> usize {
        let v = self.values(s);
        (0..self.actions).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap().then(b.cmp(&a))).unwrap()
    }

    /// Q-learning: Q(s,a) += alpha (r + max_a' Q(s',a') - Q(s,a)); terminal next state contributes 0.
    /// On real hardware every action is available in the next state.
    pub fn update(&mut self, s: State, a: usize, reward: f64, next: Option<State>) {
        let all: Vec<usize> = (0..self.actions).collect();
        self.update_within(s, a, reward, next, &all);
    }

    /// As `update`, bootstrapping only over `next_allowed` (in replay: the actions recorded in the next
    /// state; an unrecorded action's initial value is a prior, not an outcome). None allowed: no future.
    pub fn update_within(&mut self, s: State, a: usize, reward: f64, next: Option<State>, next_allowed: &[usize]) {
        let future = next.map_or(0.0, |n| {
            let v = self.values(n);
            next_allowed.iter().map(|&b| v[b]).fold(f64::NEG_INFINITY, f64::max)
        });
        let future = if future.is_finite() { future } else { 0.0 };
        let (alpha, n, init) = (self.config.alpha, self.actions, self.initial_row());
        let row = self.q.entry(s.key()).or_insert(init);
        row[a] += alpha * (reward + future - row[a]);
        self.visits.entry(s.key()).or_insert_with(|| vec![0; n])[a] += 1;
        self.updates += 1;
    }

    pub fn end_episode(&mut self) {
        self.episodes += 1;
    }

    pub fn to_json(&self) -> Value {
        let q: Vec<Value> = self.q.iter().map(|(k, v)| Value::obj().with("state", *k as u64).with("q", v.clone())
            .with("visits", self.visits.get(k).cloned().unwrap_or_default())).collect();
        let v = Value::obj().with("config", self.config.to_json()).with("actions", self.actions).with("episodes", self.episodes)
            .with("updates", self.updates).with("rng", format!("{}", self.rng)).with("table", q);
        // only when set, so the digests of OPT1/OPT3 learners stay what they were
        let v = if self.dense { v.with("dense", true) } else { v };
        match &self.prior { Some(p) => v.with("prior", p.clone()), None => v }
    }

    pub fn from_json(v: &Value) -> Option<QTable> {
        let mut t = QTable::new(QConfig::from_json(v.get("config")?)?, v.num("actions")? as usize, 1);
        t.episodes = v.num("episodes")? as u64;
        t.updates = v.num("updates")? as u64;
        t.rng = v.str("rng")?.parse().ok()?;
        t.dense = v.boolean("dense").unwrap_or(false);
        t.prior = match v.arr("prior") {
            Some(p) => Some(p.iter().map(|x| x.as_f64()).collect::<Option<Vec<f64>>>()?),
            None if v.get("prior").is_some() => return None,
            None => None,
        };
        if t.prior.as_ref().is_some_and(|p| p.len() != t.actions || p.iter().any(|x| !x.is_finite())) {
            return None;
        }
        for row in v.arr("table")? {
            let k = row.num("state")? as u32;
            let q: Vec<f64> = row.arr("q")?.iter().map(|x| x.as_f64()).collect::<Option<_>>()?;
            let n: Vec<u64> = row.arr("visits")?.iter().map(|x| x.as_f64().map(|f| f as u64)).collect::<Option<_>>()?;
            if q.len() != t.actions || n.len() != t.actions || q.iter().any(|x| !x.is_finite()) {
                return None;
            }
            t.q.insert(k, q);
            t.visits.insert(k, n);
        }
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-step chain where the immediately worse action pays off later: Q-learning must find it.
    #[test]
    fn q_learning_credits_a_delayed_payoff_a_myopic_choice_misses() {
        let start = State { level: 0, progress: 0, violated: false };
        let mid = State { level: 0, progress: 1, violated: false };
        let trap = State { level: 2, progress: 1, violated: false };
        let mut t = QTable::new(QConfig { epsilon_episodes: 200.0, ..DEFAULT }, 2, 7);
        for _ in 0..400 {
            let a = t.choose(start, true, &[0, 1]);
            // action 1 costs more now (-2) but leads to a cheap finish; action 0 is cheap now, expensive later
            let (r, s2) = if a == 1 { (-2.0, mid) } else { (-1.0, trap) };
            t.update(start, a, r, Some(s2));
            let b = t.choose(s2, true, &[0, 1]);
            t.update(s2, b, if s2 == mid { -1.0 } else { -6.0 }, None);
            t.end_episode();
        }
        assert_eq!(t.greedy(start), 1, "{:?}", t.values(start));
        let back = QTable::from_json(&crate::json::parse(&t.to_json().dump()).unwrap()).unwrap();
        assert_eq!(back, t);
        for k in 0..32 {
            assert_eq!(State::from_key(k).key(), k);
        }
    }

    /// OPT1 seeded with `seed | 1`: seeds 2 and 3 explored identically. Every distinct seed must differ.
    #[test]
    fn distinct_seeds_give_distinct_exploration_streams() {
        let draws = |seed| {
            let mut t = QTable::new(QConfig { epsilon_start: 1.0, epsilon_end: 1.0, ..DEFAULT }, 4, seed);
            (0..32).map(|_| t.choose(State::from_key(0), true, &[0, 1, 2, 3])).collect::<Vec<_>>()
        };
        for (a, b) in [(2, 3), (0, 1), (4242, 4243), (1, 3)] {
            assert_ne!(draws(a), draws(b), "seeds {a} and {b} share a stream");
        }
        let states: std::collections::BTreeSet<u64> = (0..10_000).map(stream).collect();
        assert_eq!(states.len(), 10_000);
        assert!(!states.contains(&0));
    }

    /// Replay may only bootstrap over actions recorded in the next state: an unrecorded action's
    /// optimistic initial value (0 here) must not lift the target.
    #[test]
    fn the_bootstrap_uses_only_the_allowed_follow_up_actions() {
        let (s, n) = (State::from_key(0), State::from_key(2));
        let mut t = QTable::new(QConfig { alpha: 1.0, ..DEFAULT }, 3, 1);
        t.update(n, 1, -5.0, None); // Q(n, 1) = -5; actions 0 and 2 never recorded, still 0
        t.update_within(s, 0, -1.0, Some(n), &[1]);
        assert_eq!(t.values(s)[0], -6.0);
        t.update(s, 2, -1.0, Some(n)); // over every action: the unrecorded 0 wins
        assert_eq!(t.values(s)[2], -1.0);
        t.update_within(s, 1, -1.0, Some(n), &[]);
        assert_eq!(t.values(s)[1], -1.0, "no allowed follow-up: no future term");
    }

    /// OPT2 warm start: unvisited states start from the sweep's mean returns, so its best action is greedy
    /// and an ordinary update does not hand the lead to an untried action (a prior of 0 against -1 would:
    /// any real reward pushes the prior below the untried ones). Dense and prior survive the JSON round
    /// trip, and a table without them keeps OPT3's form.
    #[test]
    fn a_warm_start_keeps_the_sweeps_best_until_learning_says_otherwise() {
        let s = State::from_key(5);
        let mut t = QTable::new(QConfig { alpha: 0.3, ..DEFAULT }, 4, 3);
        let cold = t.to_json().dump();
        assert!(!cold.contains("prior") && !cold.contains("dense"));
        t.prior = Some(vec![-16.0, -11.0, -10.0, -14.0]);
        t.dense = true;
        assert_eq!(t.greedy(s), 2);
        t.update(s, 2, -1.0, Some(State::from_key(6))); // a normal slice: -1 plus the next state's -10
        assert_eq!(t.greedy(s), 2, "one ordinary update keeps the sweep's best greedy");
        t.update(s, 2, -5.0, Some(State::from_key(6))); // violations: the prior action loses its lead in s only
        t.update(s, 2, -5.0, Some(State::from_key(6)));
        assert_eq!(t.greedy(s), 1);
        assert_eq!(t.greedy(State::from_key(7)), 2);
        let back = QTable::from_json(&crate::json::parse(&t.to_json().dump()).unwrap()).unwrap();
        assert_eq!(back, t);
        assert!(QTable::from_json(&t.to_json().with("prior", vec![0.0, 1.0])).is_none(), "a prior of another width is refused");
    }
}
