//! Small CPU-only online policy controller. The owner serializes all calls.
//! Operational labels train a working table; only prospective A/A/B trials
//! can publish a policy. No I/O, allocation, logging, or GPU work in decide.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub const ABI: u32 = 1;
const ACTIONS: usize = 4;
const CELLS: usize = 6;
const PENDING: usize = 256;
const MAX_PAIRS: usize = 256;
const UNIT: u64 = 1_000_000;
const OK: i32 = 0;
const INPUT: i32 = 1;
const TICKET: i32 = 2;
const UNAVAILABLE: i32 = 3;
const SMALL: i32 = 5;
const MAGIC: &[u8] = b"IMCSTATE01";
const MAGIC_ADVERSE: &[u8] = b"IMCSTATE02";
const MAGIC_SEQUENTIAL: &[u8] = b"IMCSTATE03";
pub const PROTOCOL_STRICT: u32 = 1;
pub const PROTOCOL_ADVERSE: u32 = 2;
/// Adverse scoring plus an anytime-valid e-process that may decide after every window.
pub const PROTOCOL_SEQUENTIAL: u32 = 3;
/// Sequential v3 with every window scored by its raw ratio; noise is still added to the
/// win criterion and the mean gate, but a noisy control is not a forced loss.
pub const PROTOCOL_NOISE: u32 = 4;
const MAGIC_NOISE: &[u8] = b"IMCSTATE04";

fn is_sequential(protocol: u32) -> bool {
    matches!(protocol, PROTOCOL_SEQUENTIAL | PROTOCOL_NOISE)
}
const SEQUENTIAL_MIN_WINDOWS: u32 = 8;
const SEQUENTIAL_BETS: [f64; 4] = [0.5, 1.0, 1.5, 1.9];

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub abi_version: u32,
    pub enabled: u32,
    pub identity: [u8; 32],
    pub action_digest: [u8; 32],
    pub schema_digest: [u8; 32],
    pub objective_digest: [u8; 32],
    pub action_count: u32,
    pub min_train: u32,
    pub freeze_every: u32,
    pub min_pairs: u32,
    pub max_trials: u32,
    pub exploration_ppm: u32,
    pub trial_budget: u32,
    pub min_gain: f64,
    pub max_regression: f64,
    pub learning_rate: f64,
    pub family_alpha: f64,
    pub seed: u64,
}

impl Config {
    fn valid(&self) -> bool {
        self.abi_version == ABI
            && self.enabled <= 1
            && self.identity != [0; 32]
            && self.action_digest != [0; 32]
            && self.schema_digest != [0; 32]
            && self.objective_digest != [0; 32]
            && (2..=4).contains(&self.action_count)
            && (1..=1_000_000).contains(&self.min_train)
            && (1..=1_000_000).contains(&self.freeze_every)
            && (8..=MAX_PAIRS as u32).contains(&self.min_pairs)
            && (1..=64).contains(&self.max_trials)
            && self.exploration_ppm <= 50_000
            && (2..=256).contains(&self.trial_budget)
            && self.min_gain.is_finite()
            && self.min_gain > 0.0
            && self.min_gain < 1.0
            && self.max_regression.is_finite()
            && (0.0..=1.0).contains(&self.max_regression)
            && self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.learning_rate <= 1.0
            && self.family_alpha.is_finite()
            && self.family_alpha > 0.0
            && self.family_alpha <= 0.1
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Context {
    pub abi_version: u32,
    pub feature_valid: u32,
    pub identity: [u8; 32],
    pub workload_digest: [u8; 32],
    pub eligible_mask: u32,
    pub request_count: u32,
    pub max_tokens: u32,
    pub phase: u32,
    pub now_ns: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Decision {
    pub ticket: u64,
    pub policy_version: u64,
    pub candidate_version: u64,
    pub action: u32,
    pub kind: u32,
    pub reason: u32,
    pub cell: u32,
    pub propensity: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Completion {
    pub ticket: u64,
    pub actual_action: u32,
    pub status: u32,
    pub correctness: u32,
    pub resource_ok: u32,
    pub cost: f64,
    pub completed_ns: u64,
}

/// Ephemeral authority for separately budgeted extra hardware executions.
/// Issued quota is consumed by reservations, never converted to live credits.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct TrainingBudgetStatus {
    pub granted: u32,
    pub used: u32,
    pub remaining: u32,
    pub active: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Status {
    pub active_version: u64,
    pub fallback_version: u64,
    pub candidate_version: u64,
    pub frozen_after: u64,
    pub decisions: u64,
    pub observations: u64,
    pub updates: u64,
    pub dropped: u64,
    pub overridden: u64,
    pub abandoned: u64,
    pub promoted: u64,
    pub rejected: u64,
    pub invalid_trials: u64,
    pub last_pair_id: u64,
    pub enabled: u32,
    pub killed: u32,
    pub fault_mask: u32,
    pub candidate_cell: u32,
    pub candidate_action: u32,
    pub pairs_completed: u32,
    pub min_pairs: u32,
    pub trials_started: u32,
    pub credits: u32,
    pub pending: u32,
    pub last_outcome: u32,
    pub next_order: u32,
    pub mean_ratio: f64,
    pub aa_noise: f64,
    pub upper_bound: f64,
    pub active_actions: [u32; CELLS],
    pub fallback_actions: [u32; CELLS],
    pub candidate_actions: [u32; CELLS],
    pub working_counts: [u64; CELLS * ACTIONS],
    pub working_means: [f64; CELLS * ACTIONS],
    pub last_update_ns: u64,
    pub model_bytes: u64,
    pub state_bytes: u64,
    pub runtime_bytes: u64,
}

#[derive(Clone, Copy, Default, Debug)]
struct Statistic {
    count: u64,
    mean: f64,
}

#[derive(Clone, Copy, Default, Debug)]
struct Policy {
    version: u64,
    actions: [u32; CELLS],
    table: [[Statistic; ACTIONS]; CELLS],
}

#[derive(Clone, Copy, Default)]
struct Pending {
    ticket: u64,
    decision: Decision,
    issued_ns: u64,
    pair_id: u64,
}

#[derive(Clone, Copy)]
struct Candidate {
    policy: Policy,
    cell: u32,
    action: u32,
    frozen_after: u64,
    count: u32,
    ratios: [f64; MAX_PAIRS],
    noises: [f64; MAX_PAIRS],
}

#[derive(Clone, Copy, Default)]
struct Triplet {
    id: u64,
    cell: u32,
    workload: [u8; 32],
    completed_mask: u32,
    costs: [f64; 3],
}

#[derive(Clone, Copy)]
struct Qualification {
    version: u64,
    reference_version: u64,
    frozen_after: u64,
    cell: u32,
    action: u32,
    count: u32,
    ratios: [f64; MAX_PAIRS],
    noises: [f64; MAX_PAIRS],
}

impl Default for Qualification {
    fn default() -> Self {
        Self { version: 0, reference_version: 0, frozen_after: 0, cell: 0, action: 0,
            count: 0, ratios: [0.0; MAX_PAIRS], noises: [0.0; MAX_PAIRS] }
    }
}

impl Qualification {
    fn acceptable(&self, config: Config, protocol: u32) -> bool {
        if self.version == 0 || self.action >= config.action_count || self.cell >= CELLS as u32 {
            return false;
        }
        comparison_summary(&self.ratios, &self.noises, self.count, config, protocol)
            .is_some_and(|summary| summary.passed)
    }
}

pub struct Controller {
    config: Config,
    protocol: u32,
    working: [[Statistic; ACTIONS]; CELLS],
    active: Policy,
    fallback: Policy,
    active_qualifications: [Qualification; CELLS],
    fallback_qualifications: [Qualification; CELLS],
    candidate: Option<Candidate>,
    triplet: Option<Triplet>,
    pending: [Pending; PENDING],
    drift_hits: [u32; CELLS],
    counters: [u64; 10], // decisions, observations, updates, dropped, overrides,
    // abandoned, promotions, rejections, invalid trials, last update time.
    next_ticket: u64,
    next_version: u64,
    random: u64,
    credit_units: u64,
    // Deliberately absent from Encoder/Decoder: restoration grants no authority.
    training_budget: TrainingBudgetStatus,
    last_freeze_update: u64,
    last_pair_id: u64,
    trials_started: u32,
    killed: bool,
    fault_mask: u32,
    last_outcome: u32,
    last_mean: f64,
    last_noise: f64,
    last_upper: f64,
    // sequential_v3 only: cells whose trial ended in the current freezing round.
    tried_cells: u32,
}

impl Controller {
    pub fn new(config: Config) -> Option<Self> {
        Self::new_with_protocol(config, PROTOCOL_STRICT)
    }

    pub fn new_with_protocol(config: Config, protocol: u32) -> Option<Self> {
        if !config.valid() || !matches!(protocol, PROTOCOL_STRICT | PROTOCOL_ADVERSE | PROTOCOL_SEQUENTIAL | PROTOCOL_NOISE) {
            return None;
        }
        Some(Self {
            config,
            protocol,
            working: [[Statistic::default(); ACTIONS]; CELLS],
            active: Policy::default(),
            fallback: Policy::default(),
            active_qualifications: [Qualification::default(); CELLS],
            fallback_qualifications: [Qualification::default(); CELLS],
            candidate: None,
            triplet: None,
            pending: [Pending::default(); PENDING],
            drift_hits: [0; CELLS],
            counters: [0; 10],
            next_ticket: 1,
            next_version: 1,
            random: config.seed.max(1),
            credit_units: 0,
            training_budget: TrainingBudgetStatus::default(),
            last_freeze_update: 0,
            last_pair_id: 0,
            trials_started: 0,
            killed: false,
            fault_mask: 0,
            last_outcome: 0,
            last_mean: 0.0,
            last_noise: 0.0,
            last_upper: 0.0,
            tried_cells: 0,
        })
    }

    fn context(&self, x: &Context) -> Option<usize> {
        let mask = (1 << self.config.action_count) - 1;
        if x.abi_version != ABI
            || x.feature_valid != 3
            || x.identity != self.config.identity
            || x.workload_digest == [0; 32]
            || x.eligible_mask & 1 == 0
            || x.eligible_mask & !mask != 0
            || !(1..=1_000_000).contains(&x.request_count)
            || !(1..=1_000_000).contains(&x.max_tokens)
            || x.phase != 1
        {
            return None;
        }
        Some((x.request_count.min(3) as usize - 1) * 2 + usize::from(x.max_tokens > 32))
    }

    fn mask(&self, x: &Context) -> u32 {
        x.eligible_mask & !self.fault_mask
    }

    pub fn predict(&self, x: &Context) -> Decision {
        let mut d = Decision {
            policy_version: self.active.version,
            candidate_version: self.candidate.as_ref().map_or(0, |c| c.policy.version),
            propensity: 1.0,
            ..Decision::default()
        };
        if self.config.enabled == 0 {
            d.reason = 1;
            return d;
        }
        if self.killed {
            d.reason = 2;
            return d;
        }
        let Some(cell) = self.context(x) else {
            d.reason = 3;
            return d;
        };
        d.cell = cell as u32;
        let action = self.active.actions[cell];
        if self.fault_mask & 1 != 0 {
            d.reason = 4;
        } else if self.mask(x) & (1 << action) == 0 {
            d.reason = 5;
        } else {
            d.action = action;
            d.kind = u32::from(action != 0);
        }
        d
    }

    fn rng(&mut self) -> u64 {
        let mut x = self.random;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.random = x;
        x
    }

    fn reserve(&mut self, mut decision: Decision, x: &Context, pair_id: u64) -> Decision {
        let Some(slot) = self.pending.iter().position(|p| p.ticket == 0) else {
            self.counters[3] = self.counters[3].saturating_add(1);
            decision.action = 0;
            decision.kind = 0;
            decision.propensity = 1.0;
            decision.reason = 6;
            return decision;
        };
        if self.next_ticket == u64::MAX {
            self.killed = true;
            self.training_end();
            decision.action = 0;
            decision.kind = 0;
            decision.reason = 2;
            return decision;
        }
        decision.ticket = self.next_ticket;
        self.next_ticket += 1;
        self.pending[slot] = Pending {
            ticket: decision.ticket,
            decision,
            issued_ns: x.now_ns,
            pair_id,
        };
        decision
    }

    pub fn decide(&mut self, x: &Context) -> Decision {
        let mut d = self.predict(x);
        self.counters[0] = self.counters[0].saturating_add(1);
        if d.reason != 0 {
            return d;
        }
        // Accrual is tied to real ordinary decisions, never holdout requests.
        self.credit_units = (self.credit_units + u64::from(self.config.exploration_ppm))
            .min(u64::from(self.config.trial_budget) * UNIT);
        if self.pending.iter().all(|p| p.ticket != 0) {
            self.counters[3] = self.counters[3].saturating_add(1);
            d.action = 0;
            d.kind = 0;
            d.reason = 6;
            return d;
        }
        let alternatives = self.mask(x) & !(1 << d.action);
        // A frozen candidate has already earned its independent comparison.
        // Keep its two extra executions funded before spending on more labels.
        // Other cells may still explore when a full extra credit is available.
        let exploration_budget = if self.candidate.is_some() { 3 * UNIT } else { UNIT };
        if alternatives != 0 && self.credit_units >= exploration_budget {
            // An exploration draw is made only after all technical filters.
            // Modulo bias is reflected in the logged exact draw probability.
            let threshold = u64::from(self.config.exploration_ppm);
            let draw = self.rng();
            let draw_probability = modulo_probability(threshold, UNIT);
            if draw % UNIT < threshold {
                let n = alternatives.count_ones();
                let chosen = self.rng() % u64::from(n);
                let action = (0..self.config.action_count)
                    .filter(|a| alternatives & (1 << a) != 0)
                    .nth(chosen as usize)
                    .unwrap_or(0);
                d.action = action;
                d.kind = 2;
                d.propensity = draw_probability * residue_probability(chosen, u64::from(n));
                self.credit_units -= UNIT;
            } else {
                d.propensity = 1.0 - draw_probability;
            }
        }
        self.reserve(d, x, 0)
    }

    /// Inside an active offline grant, collect missing labels deterministically:
    /// the least-observed eligible alternative below `min_train`, never ahead of
    /// the reference, debiting one execution from the grant. Otherwise `decide`.
    pub fn decide_training(&mut self, x: &Context) -> Decision {
        let mut d = self.predict(x);
        if d.reason != 0 || self.training_budget.active != 1 || self.training_budget.remaining == 0
            || self.pending.iter().all(|p| p.ticket != 0) {
            return self.decide(x);
        }
        let cell = d.cell as usize;
        let reference = self.working[cell][d.action as usize].count;
        let alternatives = self.mask(x) & !(1 << d.action);
        let Some(action) = (0..self.config.action_count)
            .filter(|a| alternatives & (1 << a) != 0)
            .filter(|&a| {
                let count = self.working[cell][a as usize].count;
                count < u64::from(self.config.min_train) && count < reference
            })
            .min_by_key(|&a| self.working[cell][a as usize].count) else {
            return self.decide(x);
        };
        self.counters[0] = self.counters[0].saturating_add(1);
        d.action = action;
        d.kind = 2;
        d.propensity = 1.0; // deterministic given the persisted counts
        self.training_budget.used += 1;
        self.training_budget.remaining -= 1;
        self.training_budget.active = u32::from(self.training_budget.remaining != 0);
        self.reserve(d, x, 0)
    }

    fn invalidate(&mut self) {
        if let Some(candidate) = self.candidate.take() {
            self.tried_cells |= 1 << candidate.cell;
            self.counters[8] = self.counters[8].saturating_add(1);
            self.last_outcome = 4;
        }
        self.triplet = None;
        for p in &mut self.pending {
            if p.pair_id != 0 {
                self.counters[5] = self.counters[5].saturating_add(1);
                *p = Pending::default();
            }
        }
    }

    pub fn fault(&mut self, kind: u32, action: u32) -> i32 {
        if !(1..=4).contains(&kind) || action >= self.config.action_count {
            return INPUT;
        }
        if kind == 4 {
            self.killed = true;
            self.training_end();
        } else if kind == 2 || kind == 3 {
            self.fault_mask |= 1 << action;
        }
        self.invalidate();
        self.active = self.fallback;
        self.active_qualifications = self.fallback_qualifications;
        for cell in 0..CELLS {
            if self.active.actions[cell] == action || self.fault_mask & (1 << self.active.actions[cell]) != 0
                || self.fault_mask & 1 != 0 {
                self.active.actions[cell] = 0;
                self.active_qualifications[cell] = Qualification::default();
            }
        }
        self.active.version = self.next_version;
        self.next_version = self.next_version.saturating_add(1);
        self.drift_hits = [0; CELLS];
        OK
    }

    fn freeze(&mut self) {
        if self.killed
            || self.fault_mask & 1 != 0
            || self.config.enabled == 0
            || self.candidate.is_some()
            || self.trials_started >= self.config.max_trials
            || self.counters[2].saturating_sub(self.last_freeze_update)
                < u64::from(self.config.freeze_every)
        {
            return;
        }
        let order = self.freeze_order();
        for cell in order.into_iter().flatten() {
            let reference = self.working[cell][self.active.actions[cell] as usize];
            if reference.count < u64::from(self.config.min_train) {
                continue;
            }
            let mut best = self.active.actions[cell];
            let mut cost = reference.mean;
            for action in 0..self.config.action_count {
                let statistic = self.working[cell][action as usize];
                if self.fault_mask & (1 << action) == 0
                    && statistic.count >= u64::from(self.config.min_train)
                    && statistic.mean < cost * (1.0 - self.config.min_gain)
                {
                    best = action;
                    cost = statistic.mean;
                }
            }
            if best == self.active.actions[cell] {
                continue;
            }
            let mut policy = self.active;
            policy.version = self.next_version;
            self.next_version = self.next_version.saturating_add(1);
            policy.actions[cell] = best;
            // Unchanged cells keep their qualified snapshot and monitoring
            // baseline. Ordinary observations cannot silently reset them.
            policy.table[cell] = self.working[cell];
            self.candidate = Some(Candidate {
                policy,
                cell: cell as u32,
                action: best,
                frozen_after: self.next_ticket - 1,
                count: 0,
                ratios: [0.0; MAX_PAIRS],
                noises: [0.0; MAX_PAIRS],
            });
            self.last_freeze_update = self.counters[2];
            self.trials_started += 1;
            self.last_outcome = 1;
            self.last_mean = 0.0;
            self.last_noise = 0.0;
            self.last_upper = 0.0;
            return;
        }
    }

    /// The best eligible action of a cell and its estimated relative gain, if any.
    fn cell_gain(&self, cell: usize) -> Option<f64> {
        let reference = self.working[cell][self.active.actions[cell] as usize];
        if reference.count < u64::from(self.config.min_train) {
            return None;
        }
        (0..self.config.action_count)
            .filter(|&a| self.fault_mask & (1 << a) == 0)
            .map(|a| self.working[cell][a as usize])
            .filter(|s| s.count >= u64::from(self.config.min_train)
                && s.mean < reference.mean * (1.0 - self.config.min_gain))
            .map(|s| 1.0 - s.mean / reference.mean)
            .fold(None, |best: Option<f64>, gain| Some(best.map_or(gain, |b| b.max(gain))))
    }

    /// Strict/adverse: index order. Sequential: untried cells by estimated gain;
    /// once every promising cell had a trial, a new round starts.
    fn freeze_order(&mut self) -> [Option<usize>; CELLS] {
        let mut order = [None; CELLS];
        if !is_sequential(self.protocol) {
            for (slot, cell) in order.iter_mut().zip(0..CELLS) { *slot = Some(cell); }
            return order;
        }
        let promising: Vec<(usize, f64)> = (0..CELLS).filter_map(|c| self.cell_gain(c).map(|g| (c, g))).collect();
        if !promising.is_empty() && promising.iter().all(|(c, _)| self.tried_cells & (1 << c) != 0) {
            self.tried_cells = 0;
        }
        let mut ranked: Vec<(usize, f64)> = promising.into_iter()
            .filter(|(c, _)| self.tried_cells & (1 << c) == 0).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (slot, (cell, _)) in order.iter_mut().zip(ranked) { *slot = Some(cell); }
        order
    }

    pub fn training_begin(&mut self, limit: u32) -> i32 {
        let maximum = (2 * self.config.min_pairs * self.config.max_trials).min(32768);
        if limit < 2 || limit > maximum || limit % 2 != 0 {
            return INPUT;
        }
        if self.training_budget.granted != 0 || self.killed || self.config.enabled == 0
            || self.triplet.is_some() || self.pending.iter().any(|p| p.ticket != 0)
        {
            return UNAVAILABLE;
        }
        self.training_budget = TrainingBudgetStatus { granted: limit, used: 0, remaining: limit, active: 1 };
        OK
    }

    pub fn training_end(&mut self) -> i32 {
        self.training_budget.active = 0;
        self.training_budget.remaining = 0;
        OK
    }

    pub fn training_status(&self) -> TrainingBudgetStatus {
        self.training_budget
    }

    pub fn begin_pair(&mut self, x: &Context, pair_id: u64) -> Result<([Decision; 3], u32), i32> {
        self.begin_funded_pair(x, pair_id, false)
    }

    pub fn begin_training_pair(&mut self, x: &Context, pair_id: u64) -> Result<([Decision; 3], u32), i32> {
        self.begin_funded_pair(x, pair_id, true)
    }

    fn begin_funded_pair(&mut self, x: &Context, pair_id: u64, training: bool) -> Result<([Decision; 3], u32), i32> {
        let Some(candidate) = self.candidate else {
            return Err(UNAVAILABLE);
        };
        let Some(cell) = self.context(x) else {
            return Err(INPUT);
        };
        let reference = self.active.actions[cell];
        let funded = if training {
            self.training_budget.active == 1 && self.training_budget.remaining >= 2
        } else {
            self.credit_units >= 2 * UNIT
        };
        if self.killed
            || self.config.enabled == 0
            || self.fault_mask & 1 != 0
            || self.triplet.is_some()
            || cell as u32 != candidate.cell
            || self.mask(x) & (1 << reference) == 0
            || self.mask(x) & (1 << candidate.action) == 0
            || !funded
            || self.pending.iter().filter(|p| p.ticket == 0).count() < 3
        {
            return Err(UNAVAILABLE);
        }
        if pair_id == 0 || pair_id <= self.last_pair_id || self.next_ticket > u64::MAX - 3 {
            return Err(INPUT);
        }
        let order = candidate.count % 2;
        let mut decisions = [Decision::default(); 3];
        for (arm, action) in [reference, reference, candidate.action].into_iter().enumerate() {
            let d = Decision {
                policy_version: self.active.version,
                candidate_version: candidate.policy.version,
                action,
                kind: arm as u32 + 3,
                cell: cell as u32,
                propensity: 1.0, // deterministic balanced comparative arm assignment
                ..Decision::default()
            };
            decisions[arm] = self.reserve(d, x, pair_id);
        }
        if training {
            self.training_budget.used += 2;
            self.training_budget.remaining -= 2;
            self.training_budget.active = u32::from(self.training_budget.remaining != 0);
        } else {
            self.credit_units -= 2 * UNIT;
        }
        self.last_pair_id = pair_id;
        self.triplet = Some(Triplet {
            id: pair_id,
            cell: cell as u32,
            workload: x.workload_digest,
            ..Triplet::default()
        });
        Ok((decisions, order))
    }

    pub fn complete(&mut self, x: &Completion) -> i32 {
        let Some(index) = self.pending.iter().position(|p| p.ticket != 0 && p.ticket == x.ticket) else {
            return TICKET;
        };
        let p = self.pending[index];
        // Consume even malformed feedback. Retrying the same ticket cannot train.
        self.pending[index] = Pending::default();
        self.counters[1] = self.counters[1].saturating_add(1);
        if x.status == 4 {
            self.counters[5] = self.counters[5].saturating_add(1);
            if x.actual_action != p.decision.action {
                self.counters[4] = self.counters[4].saturating_add(1);
            }
            if p.pair_id != 0 { self.invalidate(); }
            return OK;
        }
        if x.actual_action >= self.config.action_count
            || x.status > 4
            || x.correctness > 1
            || x.resource_ok > 1
            || !x.cost.is_finite()
            || x.cost <= 0.0
            || x.cost > 1e15
            || x.completed_ns < p.issued_ns
        {
            if p.pair_id != 0 {
                self.invalidate();
            }
            return INPUT;
        }
        if x.actual_action != p.decision.action {
            self.counters[4] = self.counters[4].saturating_add(1);
            if p.pair_id != 0 {
                self.invalidate();
            }
            return INPUT;
        }
        if x.correctness == 0 {
            self.fault(2, x.actual_action);
            return INPUT;
        }
        if x.resource_ok == 0 {
            if p.decision.policy_version == self.active.version
                && x.actual_action != 0
                && self.active.actions[p.decision.cell as usize] == x.actual_action
            {
                // Hard service/resource failure withdraws the deployed policy.
                self.fault(1, x.actual_action);
            } else if p.pair_id != 0 {
                self.invalidate();
            }
            return INPUT;
        }
        if p.pair_id != 0 {
            if x.status != 0 {
                self.invalidate();
                return INPUT;
            }
            return self.complete_holdout(p, x.cost);
        }
        if self.killed || self.fault_mask & (1 << x.actual_action) != 0 || x.status == 4 {
            self.counters[5] = self.counters[5].saturating_add(1);
            return OK;
        }
        let cell = p.decision.cell as usize;
        let statistic = &mut self.working[cell][x.actual_action as usize];
        let next_mean = if statistic.count == 0 {
            x.cost
        } else {
            (1.0 - self.config.learning_rate) * statistic.mean + self.config.learning_rate * x.cost
        };
        if !next_mean.is_finite() || next_mean <= 0.0 {
            self.fault(3, x.actual_action);
            return INPUT;
        }
        statistic.mean = next_mean;
        statistic.count = statistic.count.saturating_add(1);
        self.counters[2] = self.counters[2].saturating_add(1);
        self.counters[9] = self.counters[9].max(x.completed_ns);
        let baseline = self.active.table[cell][x.actual_action as usize];
        if p.decision.policy_version == self.active.version && self.active.actions[cell] != 0
            && self.active.actions[cell] == x.actual_action
            && baseline.count > 0
            && statistic.mean > baseline.mean * (1.0 + self.config.max_regression)
        {
            self.drift_hits[cell] = self.drift_hits[cell].saturating_add(1);
            if self.drift_hits[cell] >= 8 {
                self.fault(1, x.actual_action);
            }
        } else if p.decision.policy_version == self.active.version {
            self.drift_hits[cell] = 0;
        }
        self.freeze();
        OK
    }

    fn complete_holdout(&mut self, p: Pending, cost: f64) -> i32 {
        let (Some(candidate), Some(mut triplet)) = (self.candidate, self.triplet) else {
            return TICKET;
        };
        if p.pair_id != triplet.id
            || p.decision.candidate_version != candidate.policy.version
            || p.ticket <= candidate.frozen_after
            || p.decision.cell != triplet.cell
        {
            self.invalidate();
            return INPUT;
        }
        let arm = (p.decision.kind - 3) as usize;
        if arm >= 3 || triplet.completed_mask & (1 << arm) != 0 {
            self.invalidate();
            return INPUT;
        }
        triplet.costs[arm] = cost;
        triplet.completed_mask |= 1 << arm;
        self.triplet = Some(triplet);
        if triplet.completed_mask != 7 {
            return OK;
        }
        self.triplet = None;
        let reference = (triplet.costs[0] + triplet.costs[1]) * 0.5;
        let ratio = triplet.costs[2] / reference;
        let noise = (triplet.costs[0] - triplet.costs[1]).abs() / reference;
        if score_ratio(ratio, noise, self.config, self.protocol).is_none() {
            self.reject_trial();
            return OK;
        }
        let candidate = self.candidate.as_mut().expect("candidate checked");
        let index = candidate.count as usize;
        candidate.ratios[index] = ratio;
        candidate.noises[index] = noise;
        candidate.count += 1;
        let decides = if is_sequential(self.protocol) {
            candidate.count >= SEQUENTIAL_MIN_WINDOWS
        } else {
            candidate.count == self.config.min_pairs
        };
        if decides {
            let Some(summary) = comparison_summary(&candidate.ratios, &candidate.noises,
                candidate.count, self.config, self.protocol) else {
                self.reject_trial();
                return INPUT;
            };
            if !summary.passed && !summary.hopeless {
                return OK; // Sequential only: keep collecting windows.
            }
            self.last_mean = summary.mean_ratio;
            self.last_noise = summary.mean_noise;
            self.last_upper = summary.upper_bound;
            if summary.passed {
                self.fallback = self.active;
                self.fallback_qualifications = self.active_qualifications;
                self.active_qualifications[candidate.cell as usize] = Qualification {
                    version: candidate.policy.version,
                    reference_version: self.active.version,
                    frozen_after: candidate.frozen_after,
                    cell: candidate.cell,
                    action: candidate.action,
                    count: candidate.count,
                    ratios: candidate.ratios,
                    noises: candidate.noises,
                };
                self.active = candidate.policy;
                self.tried_cells |= 1 << candidate.cell;
                self.candidate = None;
                self.counters[6] = self.counters[6].saturating_add(1);
                self.last_outcome = 2;
                self.drift_hits = [0; CELLS];
            } else {
                self.reject_trial();
            }
        }
        OK
    }

    fn reject_trial(&mut self) {
        if let Some(candidate) = self.candidate.take() {
            self.tried_cells |= 1 << candidate.cell;
        }
        self.triplet = None;
        self.counters[7] = self.counters[7].saturating_add(1);
        self.last_outcome = 3;
    }

    pub fn status(&self) -> Status {
        let mut status = Status {
            active_version: self.active.version,
            fallback_version: self.fallback.version,
            candidate_version: self.candidate.map_or(0, |c| c.policy.version),
            frozen_after: self.candidate.map_or(0, |c| c.frozen_after),
            decisions: self.counters[0],
            observations: self.counters[1],
            updates: self.counters[2],
            dropped: self.counters[3],
            overridden: self.counters[4],
            abandoned: self.counters[5],
            promoted: self.counters[6],
            rejected: self.counters[7],
            invalid_trials: self.counters[8],
            last_pair_id: self.last_pair_id,
            enabled: self.config.enabled,
            killed: u32::from(self.killed),
            fault_mask: self.fault_mask,
            candidate_cell: self.candidate.map_or(u32::MAX, |c| c.cell),
            candidate_action: self.candidate.map_or(0, |c| c.action),
            pairs_completed: self.candidate.map_or(0, |c| c.count),
            min_pairs: self.config.min_pairs,
            trials_started: self.trials_started,
            credits: (self.credit_units / UNIT) as u32,
            pending: self.pending.iter().filter(|p| p.ticket != 0).count() as u32,
            last_outcome: self.last_outcome,
            next_order: self.candidate.map_or(0, |c| c.count % 2),
            mean_ratio: self.last_mean,
            aa_noise: self.last_noise,
            upper_bound: self.last_upper,
            active_actions: self.active.actions,
            fallback_actions: self.fallback.actions,
            candidate_actions: self.candidate.map_or([0; CELLS], |c| c.policy.actions),
            last_update_ns: self.counters[9],
            model_bytes: (size_of_val(&self.working)
                + size_of::<Policy>() * (2 + usize::from(self.candidate.is_some()))) as u64,
            state_bytes: self.checkpoint().len() as u64,
            runtime_bytes: size_of::<Self>() as u64,
            ..Status::default()
        };
        for cell in 0..CELLS {
            for action in 0..ACTIONS {
                let i = cell * ACTIONS + action;
                status.working_counts[i] = self.working[cell][action].count;
                status.working_means[i] = self.working[cell][action].mean;
            }
        }
        status
    }
}

// Exact probabilities for uniform u64 draws followed by modulo reduction.
fn modulo_probability(threshold: u64, modulus: u64) -> f64 {
    let total = (u64::MAX as u128) + 1;
    let quotient = total / modulus as u128;
    let remainder = total % modulus as u128;
    let favorable = quotient * threshold as u128 + remainder.min(threshold as u128);
    favorable as f64 / total as f64
}
fn residue_probability(residue: u64, modulus: u64) -> f64 {
    let total = (u64::MAX as u128) + 1;
    let favorable = total / modulus as u128 + u128::from((residue as u128) < total % modulus as u128);
    favorable as f64 / total as f64
}

fn binomial_cdf(failures: u32, n: u32, probability: f64) -> f64 {
    if failures >= n || probability <= 0.0 { return 1.0; }
    if probability >= 1.0 { return 0.0; }
    // Log-sum-exp keeps the calculation valid even close to probability 1.
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

fn failure_upper(failures: u32, n: u32, alpha: f64) -> f64 {
    if failures == n { return 1.0; }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..64 {
        let middle = (low + high) * 0.5;
        if binomial_cdf(failures, n, middle) > alpha { low = middle; }
        else { high = middle; }
    }
    high
}

struct ComparisonSummary {
    mean_ratio: f64,
    mean_noise: f64,
    upper_bound: f64,
    passed: bool,
    hopeless: bool,
}

fn score_ratio(ratio: f64, noise: f64, config: Config, protocol: u32) -> Option<f64> {
    if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 + config.max_regression
        || !noise.is_finite() || noise < 0.0 || noise > 2.0 {
        return None;
    }
    match protocol {
        PROTOCOL_STRICT => (noise <= config.max_regression).then_some(ratio),
        PROTOCOL_NOISE => Some(ratio),
        PROTOCOL_ADVERSE | PROTOCOL_SEQUENTIAL =>
            Some(if noise > config.max_regression { 1.0 + config.max_regression } else { ratio }),
        _ => None,
    }
}

fn comparison_summary(ratios: &[f64; MAX_PAIRS], noises: &[f64; MAX_PAIRS], count: u32,
    config: Config, protocol: u32) -> Option<ComparisonSummary> {
    let sequential = is_sequential(protocol);
    if if sequential { !(SEQUENTIAL_MIN_WINDOWS..=config.min_pairs).contains(&count) } else { count != config.min_pairs } {
        return None;
    }
    let mut ratio_sum = 0.0;
    let mut noise_sum = 0.0;
    let mut failures = 0;
    // One betting product per fixed bet l: prod(1 + l (win - 1/2)). Under the
    // null (win frequency at most 1/2) each is a nonnegative supermartingale.
    let mut wealth = [1.0; SEQUENTIAL_BETS.len()];
    for i in 0..MAX_PAIRS {
        if i >= count as usize {
            if ratios[i] != 0.0 || noises[i] != 0.0 { return None; }
            continue;
        }
        let ratio = score_ratio(ratios[i], noises[i], config, protocol)?;
        ratio_sum += ratio;
        noise_sum += noises[i]; // Full measured noise remains an adverse surcharge.
        let win = ratio + noises[i] < 1.0 - config.min_gain;
        if !win { failures += 1; }
        for (w, bet) in wealth.iter_mut().zip(SEQUENTIAL_BETS) {
            *w *= 1.0 + bet * if win { 0.5 } else { -0.5 };
        }
    }
    // Preserve strict's separate sums/divisions and final addition exactly.
    let mean_ratio = ratio_sum / f64::from(count);
    let mean_noise = noise_sum / f64::from(count);
    let mean_gate = mean_ratio + mean_noise < 1.0 - config.min_gain;
    let alpha = config.family_alpha / f64::from(config.max_trials);
    if !sequential {
        let upper_bound = failure_upper(failures, count, alpha);
        let passed = upper_bound < 0.5 && mean_gate;
        return Some(ComparisonSummary { mean_ratio, mean_noise, upper_bound, passed, hopeless: !passed });
    }
    // Half the nominal alpha for the mixture e-process (Ville: P(sup E >= 2/alpha)
    // <= alpha/2 at any stopping time), half for the unchanged fixed test at the cap.
    let threshold = 2.0 / alpha;
    let evidence = wealth.iter().sum::<f64>() / wealth.len() as f64;
    let remaining = config.min_pairs - count;
    let reachable = wealth.iter().zip(SEQUENTIAL_BETS)
        .map(|(w, bet)| w * (1.0 + 0.5 * bet).powi(remaining as i32))
        .sum::<f64>() / wealth.len() as f64 >= threshold;
    let final_upper = failure_upper(failures, config.min_pairs, alpha * 0.5);
    let upper_bound = if remaining == 0 { final_upper } else { failure_upper(failures, count, alpha * 0.5) };
    let passed = mean_gate && (evidence >= threshold || (remaining == 0 && final_upper < 0.5));
    Some(ComparisonSummary { mean_ratio, mean_noise, upper_bound, passed,
        hopeless: !passed && (remaining == 0 || (!reachable && final_upper >= 0.5)) })
}

#[derive(Default)]
struct Encoder(Vec<u8>);
impl Encoder {
    fn u32(&mut self, v: u32) { self.0.extend(v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.0.extend(v.to_le_bytes()); }
    fn f64(&mut self, v: f64) { self.u64(v.to_bits()); }
    fn statistic(&mut self, s: Statistic) { self.u64(s.count); self.f64(s.mean); }
    fn policy(&mut self, p: Policy) {
        self.u64(p.version);
        for a in p.actions { self.u32(a); }
        for row in p.table { for s in row { self.statistic(s); } }
    }
    fn config(&mut self, c: Config) {
        self.u32(c.abi_version); self.u32(c.enabled);
        self.0.extend(c.identity); self.0.extend(c.action_digest);
        self.0.extend(c.schema_digest); self.0.extend(c.objective_digest);
        for x in [c.action_count, c.min_train, c.freeze_every, c.min_pairs, c.max_trials,
            c.exploration_ppm, c.trial_budget] { self.u32(x); }
        for x in [c.min_gain, c.max_regression, c.learning_rate, c.family_alpha] { self.f64(x); }
        self.u64(c.seed);
    }
    fn decision(&mut self, d: Decision) {
        for x in [d.ticket, d.policy_version, d.candidate_version] { self.u64(x); }
        for x in [d.action, d.kind, d.reason, d.cell] { self.u32(x); }
        self.f64(d.propensity);
    }
    fn qualification(&mut self, q: Qualification) {
        self.u64(q.version);
        if q.version != 0 {
            self.u64(q.reference_version); self.u64(q.frozen_after);
            self.u32(q.cell); self.u32(q.action); self.u32(q.count);
            for x in q.ratios { self.f64(x); }
            for x in q.noises { self.f64(x); }
        }
    }
}

struct Decoder<'a> { bytes: &'a [u8], index: usize }
impl<'a> Decoder<'a> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.index.checked_add(N)?;
        let x = self.bytes.get(self.index..end)?.try_into().ok()?;
        self.index = end;
        Some(x)
    }
    fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take()?)) }
    fn u64(&mut self) -> Option<u64> { Some(u64::from_le_bytes(self.take()?)) }
    fn f64(&mut self) -> Option<f64> {
        let x = f64::from_bits(self.u64()?);
        x.is_finite().then_some(x)
    }
    fn statistic(&mut self) -> Option<Statistic> {
        let s = Statistic { count: self.u64()?, mean: self.f64()? };
        ((s.count == 0 && s.mean == 0.0) || (s.count > 0 && s.mean > 0.0 && s.mean <= 1e15)).then_some(s)
    }
    fn policy(&mut self, count: u32) -> Option<Policy> {
        let mut p = Policy { version: self.u64()?, ..Policy::default() };
        for a in &mut p.actions { *a = self.u32()?; if *a >= count { return None; } }
        for row in &mut p.table { for s in row { *s = self.statistic()?; } }
        Some(p)
    }
    fn decision(&mut self, count: u32) -> Option<Decision> {
        let d = Decision { ticket: self.u64()?, policy_version: self.u64()?, candidate_version: self.u64()?,
            action: self.u32()?, kind: self.u32()?, reason: self.u32()?, cell: self.u32()?, propensity: self.f64()? };
        (d.action < count && d.kind <= 5 && d.reason <= 7 && d.cell < CELLS as u32
            && d.propensity > 0.0 && d.propensity <= 1.0).then_some(d)
    }
    fn qualification(&mut self, config: Config, protocol: u32) -> Option<Qualification> {
        let version = self.u64()?;
        if version == 0 { return Some(Qualification::default()); }
        let mut q = Qualification { version, reference_version: self.u64()?, frozen_after: self.u64()?,
            cell: self.u32()?, action: self.u32()?, count: self.u32()?, ..Qualification::default() };
        for x in &mut q.ratios { *x = self.f64()?; }
        for x in &mut q.noises { *x = self.f64()?; }
        q.acceptable(config, protocol).then_some(q)
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    // Accidental-corruption check only. The Python envelope supplies SHA256.
    bytes.iter().fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3))
}

impl Controller {
    pub fn checkpoint(&self) -> Vec<u8> {
        let mut e = Encoder::default();
        e.0.extend(match self.protocol {
            PROTOCOL_ADVERSE => MAGIC_ADVERSE,
            PROTOCOL_SEQUENTIAL => MAGIC_SEQUENTIAL,
            PROTOCOL_NOISE => MAGIC_NOISE,
            _ => MAGIC,
        });
        e.config(self.config);
        for row in self.working { for s in row { e.statistic(s); } }
        e.policy(self.active); e.policy(self.fallback);
        for q in self.active_qualifications { e.qualification(q); }
        for q in self.fallback_qualifications { e.qualification(q); }
        e.u32(u32::from(self.candidate.is_some()));
        if let Some(c) = self.candidate {
            e.policy(c.policy); e.u32(c.cell); e.u32(c.action); e.u64(c.frozen_after); e.u32(c.count);
            for x in c.ratios { e.f64(x); } for x in c.noises { e.f64(x); }
        }
        e.u32(u32::from(self.triplet.is_some()));
        if let Some(t) = self.triplet {
            e.u64(t.id); e.u32(t.cell); e.0.extend(t.workload); e.u32(t.completed_mask);
            for x in t.costs { e.f64(x); }
        }
        for p in self.pending {
            e.u64(p.ticket);
            if p.ticket != 0 { e.decision(p.decision); e.u64(p.issued_ns); e.u64(p.pair_id); }
        }
        for x in self.drift_hits { e.u32(x); }
        for x in self.counters { e.u64(x); }
        for x in [self.next_ticket, self.next_version, self.random, self.credit_units,
            self.last_freeze_update, self.last_pair_id] { e.u64(x); }
        for x in [self.trials_started, u32::from(self.killed), self.fault_mask, self.last_outcome] { e.u32(x); }
        for x in [self.last_mean, self.last_noise, self.last_upper] { e.f64(x); }
        if is_sequential(self.protocol) { e.u32(self.tried_cells); }
        e.u64(checksum(&e.0));
        e.0
    }

    pub fn restore(config: Config, bytes: &[u8]) -> Option<Self> {
        Self::restore_with_protocol(config, PROTOCOL_STRICT, bytes)
    }

    pub fn restore_with_protocol(config: Config, protocol: u32, bytes: &[u8]) -> Option<Self> {
        let magic = match protocol {
            PROTOCOL_STRICT => MAGIC,
            PROTOCOL_ADVERSE => MAGIC_ADVERSE,
            PROTOCOL_SEQUENTIAL => MAGIC_SEQUENTIAL,
            PROTOCOL_NOISE => MAGIC_NOISE,
            _ => return None,
        };
        let body = bytes.get(..bytes.len().checked_sub(8)?)?;
        let checksum_bytes = bytes.get(body.len()..)?.try_into().ok()?;
        if bytes.len() > 100_000 || checksum(body) != u64::from_le_bytes(checksum_bytes) || !body.starts_with(magic) {
            return None;
        }
        let mut expected = Encoder::default(); expected.config(config);
        if body.get(MAGIC.len()..MAGIC.len() + expected.0.len())? != expected.0 { return None; }
        let mut d = Decoder { bytes: body, index: MAGIC.len() + expected.0.len() };
        let mut c = Self::new_with_protocol(config, protocol)?;
        for row in &mut c.working { for s in row { *s = d.statistic()?; } }
        c.active = d.policy(config.action_count)?;
        c.fallback = d.policy(config.action_count)?;
        for q in &mut c.active_qualifications { *q = d.qualification(config, protocol)?; }
        for q in &mut c.fallback_qualifications { *q = d.qualification(config, protocol)?; }
        for cell in 0..CELLS {
            for (policy, q) in [(c.active, c.active_qualifications[cell]), (c.fallback, c.fallback_qualifications[cell])] {
                if policy.actions[cell] != 0 && (q.version == 0 || q.action != policy.actions[cell]) { return None; }
                if q.version != 0 && (q.version > policy.version || q.reference_version >= q.version || q.action != policy.actions[cell]
                    || q.cell as usize != cell || policy.table[cell][q.action as usize].count < u64::from(config.min_train)) { return None; }
            }
        }
        match d.u32()? {
            0 => {},
            1 => {
                let mut candidate = Candidate { policy: d.policy(config.action_count)?, cell: d.u32()?, action: d.u32()?,
                    frozen_after: d.u64()?, count: d.u32()?, ratios: [0.0; MAX_PAIRS], noises: [0.0; MAX_PAIRS] };
                if candidate.cell >= CELLS as u32 || candidate.action >= config.action_count || candidate.count >= config.min_pairs
                    || candidate.policy.actions[candidate.cell as usize] != candidate.action { return None; }
                for x in &mut candidate.ratios { *x = d.f64()?; }
                for x in &mut candidate.noises { *x = d.f64()?; }
                for i in 0..MAX_PAIRS {
                    if i < candidate.count as usize {
                        score_ratio(candidate.ratios[i], candidate.noises[i], config, protocol)?;
                    } else if candidate.ratios[i] != 0.0 || candidate.noises[i] != 0.0 { return None; }
                }
                for cell in 0..CELLS { if cell as u32 != candidate.cell && candidate.policy.actions[cell] != c.active.actions[cell] { return None; } }
                c.candidate = Some(candidate);
            },
            _ => return None,
        }
        match d.u32()? {
            0 => {},
            1 => {
                let t = Triplet { id: d.u64()?, cell: d.u32()?, workload: d.take()?, completed_mask: d.u32()?,
                    costs: [d.f64()?, d.f64()?, d.f64()?] };
                if t.id == 0 || t.cell >= CELLS as u32 || t.workload == [0; 32] || t.completed_mask >= 7
                    || c.candidate.is_none() || c.candidate?.cell != t.cell { return None; }
                for arm in 0..3 { if t.completed_mask & (1 << arm) != 0 { if t.costs[arm] <= 0.0 || t.costs[arm] > 1e15 { return None; } }
                    else if t.costs[arm] != 0.0 { return None; } }
                c.triplet = Some(t);
            },
            _ => return None,
        }
        for i in 0..PENDING {
            let ticket = d.u64()?;
            if ticket != 0 {
                let p = Pending { ticket, decision: d.decision(config.action_count)?, issued_ns: d.u64()?, pair_id: d.u64()? };
                if p.decision.ticket != ticket || c.pending[..i].iter().any(|other| other.ticket == ticket) { return None; }
                if p.pair_id != 0 {
                    let t = c.triplet?;
                    let candidate = c.candidate?;
                    if p.pair_id != t.id || p.decision.kind < 3 || p.decision.cell != t.cell
                        || p.decision.candidate_version != candidate.policy.version || ticket <= candidate.frozen_after
                        || t.completed_mask & (1 << (p.decision.kind - 3)) != 0 { return None; }
                } else if p.decision.kind >= 3 { return None; }
                c.pending[i] = p;
            }
        }
        for x in &mut c.drift_hits { *x = d.u32()?; if *x >= 8 { return None; } }
        for x in &mut c.counters { *x = d.u64()?; }
        c.next_ticket = d.u64()?; c.next_version = d.u64()?; c.random = d.u64()?;
        c.credit_units = d.u64()?; c.last_freeze_update = d.u64()?; c.last_pair_id = d.u64()?;
        c.trials_started = d.u32()?;
        c.killed = match d.u32()? { 0 => false, 1 => true, _ => return None };
        c.fault_mask = d.u32()?; c.last_outcome = d.u32()?;
        c.last_mean = d.f64()?; c.last_noise = d.f64()?; c.last_upper = d.f64()?;
        if is_sequential(protocol) {
            c.tried_cells = d.u32()?;
            if c.tried_cells >= 1 << CELLS { return None; }
        }
        if d.index != body.len() || c.next_ticket == 0 || c.next_version == 0 || c.random == 0
            || c.credit_units > u64::from(config.trial_budget) * UNIT || c.trials_started > config.max_trials
            || c.fault_mask & !((1 << config.action_count) - 1) != 0 || c.last_outcome > 4
            || c.last_freeze_update > c.counters[2] || c.counters[2] > c.counters[1]
            || c.pending.iter().any(|p| p.ticket >= c.next_ticket)
            || c.active.version >= c.next_version || c.fallback.version >= c.next_version
            || c.candidate.is_some_and(|x| x.policy.version >= c.next_version || x.frozen_after >= c.next_ticket)
            || c.last_mean < 0.0 || c.last_noise < 0.0 || c.last_upper < 0.0 { return None; }
        for q in c.active_qualifications.into_iter().chain(c.fallback_qualifications) {
            if q.version != 0 && q.frozen_after >= c.next_ticket { return None; }
        }
        if let Some(t) = c.triplet {
            if t.id != c.last_pair_id || c.pending.iter().filter(|p| p.pair_id == t.id).count()
                != (3 - t.completed_mask.count_ones()) as usize { return None; }
        }
        // Never silently turn partly executed pre-restart work into evidence.
        if c.triplet.is_some() { c.invalidate(); }
        for p in &mut c.pending {
            if p.ticket != 0 { c.counters[5] = c.counters[5].saturating_add(1); *p = Pending::default(); }
        }
        Some(c)
    }
}

fn ffi(run: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(run)).unwrap_or(INPUT)
}

/// # Safety
/// `config` must point to a readable Config. The returned handle has one owner.
#[no_mangle]
pub unsafe extern "C" fn imc_create(config: *const Config) -> *mut Controller {
    imc_create_with_protocol(config, PROTOCOL_STRICT)
}
/// # Safety
/// Handle and pointers must be valid. Directed labels only inside an active grant.
#[no_mangle]
pub unsafe extern "C" fn imc_decide_training(handle: *mut Controller, input: *const Context, output: *mut Decision) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() { return INPUT; }
    ffi(|| { ptr::write_unaligned(output, (*handle).decide_training(&ptr::read_unaligned(input))); OK })
}
/// # Safety
/// `config` must be readable; protocol is fixed for the returned owned handle.
#[no_mangle]
pub unsafe extern "C" fn imc_create_with_protocol(config: *const Config, protocol: u32) -> *mut Controller {
    if config.is_null() { return ptr::null_mut(); }
    catch_unwind(AssertUnwindSafe(|| Controller::new_with_protocol(ptr::read_unaligned(config), protocol)
        .map_or(ptr::null_mut(), |c| Box::into_raw(Box::new(c))))).unwrap_or(ptr::null_mut())
}
/// # Safety
/// Handle must be null or a live uniquely owned handle from create/restore.
#[no_mangle]
pub unsafe extern "C" fn imc_free(handle: *mut Controller) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}
/// # Safety
/// All pointers must be valid; handle calls must be externally serialized.
#[no_mangle]
pub unsafe extern "C" fn imc_decide(handle: *mut Controller, input: *const Context, output: *mut Decision) -> i32 {
    ffi(|| { if handle.is_null() || input.is_null() || output.is_null() { return INPUT; }
        ptr::write_unaligned(output, (*handle).decide(&ptr::read_unaligned(input))); OK })
}
/// # Safety
/// All pointers must be valid; no call may overlap a mutation of this handle.
#[no_mangle]
pub unsafe extern "C" fn imc_predict(handle: *const Controller, input: *const Context, output: *mut Decision) -> i32 {
    ffi(|| { if handle.is_null() || input.is_null() || output.is_null() { return INPUT; }
        ptr::write_unaligned(output, (*handle).predict(&ptr::read_unaligned(input))); OK })
}
/// # Safety
/// All pointers must be valid; handle calls must be externally serialized.
#[no_mangle]
pub unsafe extern "C" fn imc_complete(handle: *mut Controller, input: *const Completion) -> i32 {
    ffi(|| { if handle.is_null() || input.is_null() { return INPUT; }
        (*handle).complete(&ptr::read_unaligned(input)) })
}
/// # Safety
/// All pointers must be valid; outputs must be separate writable Decisions.
#[no_mangle]
pub unsafe extern "C" fn imc_begin_pair(handle: *mut Controller, input: *const Context, pair_id: u64,
    a: *mut Decision, aa: *mut Decision, b: *mut Decision, order: *mut u32) -> i32 {
    ffi_begin_pair(handle, input, pair_id, a, aa, b, order, false)
}
/// # Safety
/// All pointers must be valid; outputs must be separate writable Decisions.
#[no_mangle]
pub unsafe extern "C" fn imc_begin_training_pair(handle: *mut Controller, input: *const Context, pair_id: u64,
    a: *mut Decision, aa: *mut Decision, b: *mut Decision, order: *mut u32) -> i32 {
    ffi_begin_pair(handle, input, pair_id, a, aa, b, order, true)
}
unsafe fn ffi_begin_pair(handle: *mut Controller, input: *const Context, pair_id: u64,
    a: *mut Decision, aa: *mut Decision, b: *mut Decision, order: *mut u32, training: bool) -> i32 {
    ffi(|| { if handle.is_null() || input.is_null() || a.is_null() || aa.is_null() || b.is_null() || order.is_null() { return INPUT; }
        match (*handle).begin_funded_pair(&ptr::read_unaligned(input), pair_id, training) {
            Ok((ds, ordering)) => { ptr::write_unaligned(a, ds[0]); ptr::write_unaligned(aa, ds[1]);
                ptr::write_unaligned(b, ds[2]); ptr::write_unaligned(order, ordering); OK },
            Err(code) => code,
        } })
}
/// # Safety
/// Handle must be valid and uniquely borrowed for this call.
#[no_mangle]
pub unsafe extern "C" fn imc_training_begin(handle: *mut Controller, limit: u32) -> i32 {
    ffi(|| { if handle.is_null() { return INPUT; } (*handle).training_begin(limit) })
}
/// # Safety
/// Handle must be valid and uniquely borrowed for this call.
#[no_mangle]
pub unsafe extern "C" fn imc_training_end(handle: *mut Controller) -> i32 {
    ffi(|| { if handle.is_null() { return INPUT; } (*handle).training_end() })
}
/// # Safety
/// Handle and output must be valid; no overlapping handle calls.
#[no_mangle]
pub unsafe extern "C" fn imc_training_status(handle: *const Controller, output: *mut TrainingBudgetStatus) -> i32 {
    ffi(|| { if handle.is_null() || output.is_null() { return INPUT; }
        ptr::write_unaligned(output, (*handle).training_status()); OK })
}
/// # Safety
/// Handle and output must be valid; no overlapping handle calls.
#[no_mangle]
pub unsafe extern "C" fn imc_status(handle: *const Controller, output: *mut Status) -> i32 {
    ffi(|| { if handle.is_null() || output.is_null() { return INPUT; }
        ptr::write_unaligned(output, (*handle).status()); OK })
}
/// # Safety
/// Handle must be valid and uniquely borrowed for this call.
#[no_mangle]
pub unsafe extern "C" fn imc_fault(handle: *mut Controller, kind: u32, action: u32) -> i32 {
    ffi(|| { if handle.is_null() { return INPUT; } (*handle).fault(kind, action) })
}
/// # Safety
/// Needed must be writable; output must have capacity bytes if non-null.
#[no_mangle]
pub unsafe extern "C" fn imc_checkpoint(handle: *const Controller, output: *mut u8, capacity: usize, needed: *mut usize) -> i32 {
    ffi(|| { if handle.is_null() || needed.is_null() { return INPUT; }
        let bytes = (*handle).checkpoint(); ptr::write_unaligned(needed, bytes.len());
        if output.is_null() || capacity < bytes.len() { return SMALL; }
        ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len()); OK })
}
/// # Safety
/// Config and input must be readable, with input containing length bytes.
#[no_mangle]
pub unsafe extern "C" fn imc_restore(config: *const Config, input: *const u8, length: usize) -> *mut Controller {
    imc_restore_with_protocol(config, PROTOCOL_STRICT, input, length)
}
/// # Safety
/// Config/input must be readable. Only the explicitly expected protocol is accepted.
#[no_mangle]
pub unsafe extern "C" fn imc_restore_with_protocol(config: *const Config, protocol: u32,
    input: *const u8, length: usize) -> *mut Controller {
    if config.is_null() || input.is_null() || length > 100_000 { return ptr::null_mut(); }
    catch_unwind(AssertUnwindSafe(|| Controller::restore_with_protocol(ptr::read_unaligned(config), protocol, std::slice::from_raw_parts(input, length))
        .map_or(ptr::null_mut(), |c| Box::into_raw(Box::new(c))))).unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    fn config() -> Config {
        Config { abi_version: 1, enabled: 1, identity: [1; 32], action_digest: [2; 32],
            schema_digest: [3; 32], objective_digest: [4; 32], action_count: 3,
            min_train: 2, freeze_every: 4, min_pairs: 8, max_trials: 4,
            exploration_ppm: 50_000, trial_budget: 256, min_gain: 0.05,
            max_regression: 0.1, learning_rate: 0.2, family_alpha: 0.05, seed: 7 }
    }

    #[test]
    fn exact_sign_test_matches_known_binomial_probabilities() {
        assert!((binomial_cdf(0, 8, 0.5) - 1.0 / 256.0).abs() < 1e-14);
        assert!((binomial_cdf(1, 8, 0.5) - 9.0 / 256.0).abs() < 1e-14);
        assert!((binomial_cdf(3, 4, 0.5) - 15.0 / 16.0).abs() < 1e-14);
        let upper = failure_upper(0, 8, 0.0125);
        assert!((upper - (1.0 - 0.0125_f64.powf(1.0 / 8.0))).abs() < 1e-14);
        assert!(upper < 0.5);
        assert!(failure_upper(1, 8, 0.0125) > 0.5);
        assert_eq!(failure_upper(8, 8, 0.0125), 1.0);
        assert!(binomial_cdf(252, 256, 0.99) > 0.2);
    }

    #[test]
    fn rehashed_checkpoint_still_refuses_unqualified_and_invalid_models() {
        let cfg = config();
        let mut c = Controller::new(cfg).unwrap();
        c.active.actions[0] = 1;
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
        c.active.actions[0] = 0;
        c.working[0][0] = Statistic { count: 1, mean: f64::NAN };
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
        c.working[0][0] = Statistic { count: 0, mean: 1.0 };
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
        c.working[0][0] = Statistic::default();
        c.credit_units = 257 * UNIT;
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
    }

    #[test]
    fn qualifications_bind_the_cell_and_revalidate_actual_trial_arrays() {
        let cfg = config();
        let mut c = Controller::new(cfg).unwrap();
        let mut q = Qualification { version: 1, reference_version: 0, frozen_after: 0,
            cell: 0, action: 1, count: cfg.min_pairs, ..Qualification::default() };
        q.ratios[..8].fill(0.5);
        c.active.version = 1;
        c.next_version = 2;
        c.active.actions[0] = 1;
        c.active.table[0][1] = Statistic { count: 2, mean: 50.0 };
        c.active_qualifications[0] = q;
        assert!(Controller::restore(cfg, &c.checkpoint()).is_some());
        c.active_qualifications[0].ratios[1] = 1.0;
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
        c.active_qualifications[0] = q;
        c.active.actions[1] = 1;
        c.active.table[1][1] = Statistic { count: 2, mean: 50.0 };
        c.active_qualifications[1] = q;
        assert!(Controller::restore(cfg, &c.checkpoint()).is_none());
    }

    #[test]
    fn maximum_fixed_state_fits_checkpoint_limit() {
        let cfg = config();
        let mut c = Controller::new(cfg).unwrap();
        let mut q = Qualification { version: 1, action: 1, count: cfg.min_pairs, ..Qualification::default() };
        q.ratios[..8].fill(0.5);
        for cell in 0..CELLS {
            q.cell = cell as u32;
            c.active_qualifications[cell] = q;
            c.fallback_qualifications[cell] = q;
        }
        for i in 0..PENDING {
            c.pending[i] = Pending { ticket: i as u64 + 1,
                decision: Decision { ticket: i as u64 + 1, propensity: 1.0, ..Decision::default() },
                ..Pending::default() };
        }
        c.candidate = Some(Candidate { policy: Policy::default(), cell: 0, action: 1,
            frozen_after: 0, count: 0, ratios: [0.0; MAX_PAIRS], noises: [0.0; MAX_PAIRS] });
        assert!(c.checkpoint().len() < 100_000);
        assert!(size_of::<Controller>() < 100_000);
    }

    #[test]
    fn ffi_checks_null_pointers_and_buffer_capacity() {
        unsafe {
            assert!(imc_create(ptr::null()).is_null());
            let handle = imc_create(&config());
            assert!(!handle.is_null());
            assert_eq!(imc_decide(handle, ptr::null(), ptr::null_mut()), INPUT);
            let mut needed = 0;
            assert_eq!(imc_checkpoint(handle, ptr::null_mut(), 0, &mut needed), SMALL);
            assert!(needed > 0);
            let mut data = vec![0; needed];
            assert_eq!(imc_checkpoint(handle, data.as_mut_ptr(), data.len(), &mut needed), OK);
            let restored = imc_restore(&config(), data.as_ptr(), data.len());
            assert!(!restored.is_null());
            imc_free(restored);
            imc_free(handle);
            imc_free(ptr::null_mut());
        }
    }

    #[test]
    fn abi_layout_matches_header_fixed_width_contract() {
        assert_eq!(size_of::<Config>(), 208);
        assert_eq!(size_of::<Context>(), 96);
        assert_eq!(size_of::<Decision>(), 48);
        assert_eq!(size_of::<Completion>(), 40);
        assert_eq!(size_of::<Status>(), 672);
    }

    fn deployed_gates(budget: u32) -> Config {
        let mut c = config();
        c.action_count = 4;
        c.min_train = 8;
        c.freeze_every = 32;
        c.min_pairs = 64;
        c.max_trials = 8;
        c.trial_budget = budget;
        c.max_regression = 0.05;
        c.learning_rate = 0.125;
        c.seed = 15;
        c
    }

    fn numerical_context(count: u32) -> Context {
        Context { abi_version: ABI, feature_valid: 3, identity: [1; 32], workload_digest: [9; 32],
            eligible_mask: if count == 1 { 5 } else { 15 }, request_count: count, max_tokens: 8,
            phase: 1, now_ns: 10 }
    }

    fn numerical_complete(c: &mut Controller, d: Decision) {
        assert_eq!(c.complete(&Completion { ticket: d.ticket, actual_action: d.action, status: 0,
            correctness: 1, resource_ok: 1, cost: if d.action == 0 { 0.3 } else { 0.1 },
            completed_ns: 20 }), OK);
    }

    fn frozen_candidate(budget: u32) -> Controller {
        frozen_candidate_with_protocol(budget, PROTOCOL_STRICT)
    }

    fn frozen_candidate_with_protocol(budget: u32, protocol: u32) -> Controller {
        let mut c = Controller::new_with_protocol(deployed_gates(budget), protocol).unwrap();
        for _ in 0..4096 {
            let d = c.decide(&numerical_context(1));
            numerical_complete(&mut c, d);
            if c.candidate.is_some() { return c; }
        }
        panic!("seeded operational labels did not freeze a candidate");
    }

    fn measured_pair(c: &mut Controller, id: u64, costs: [f64; 3]) {
        let (decisions, _) = c.begin_training_pair(&numerical_context(1), id).unwrap();
        for (d, cost) in decisions.into_iter().zip(costs) {
            assert_eq!(c.complete(&Completion { ticket: d.ticket, actual_action: d.action,
                status: 0, correctness: 1, resource_ok: 1, cost, completed_ns: 20 }), OK);
        }
    }

    #[test]
    fn strict_default_checkpoint_matches_pre_protocol_golden_bytes() {
        let default = Controller::new(config()).unwrap().checkpoint();
        assert_eq!(default.len(), 3782);
        assert_eq!(checksum(&default), 0x0446728d815b5226);
        assert_eq!(default, Controller::new_with_protocol(config(), PROTOCOL_STRICT).unwrap().checkpoint());
        assert!(default.starts_with(b"IMCSTATE01"));
    }

    #[test]
    fn adverse_score_is_pointwise_pessimistic_and_noisy_windows_are_losses() {
        let cfg = deployed_gates(6);
        for ratio in [0.01, 0.3, 0.94, 1.0, 1.05] {
            for noise in [0.0, 0.01, 0.05, 0.050001, 0.2, 1.5, 2.0] {
                let score = score_ratio(ratio, noise, cfg, PROTOCOL_ADVERSE).unwrap();
                assert!(score >= ratio);
                if noise <= cfg.max_regression {
                    assert_eq!(score, score_ratio(ratio, noise, cfg, PROTOCOL_STRICT).unwrap());
                } else {
                    assert_eq!(score, 1.0 + cfg.max_regression);
                    assert!(score + noise >= 1.0 - cfg.min_gain);
                    assert!(score_ratio(ratio, noise, cfg, PROTOCOL_STRICT).is_none());
                }
            }
        }
        for (ratio, noise) in [(1.050001, 0.0), (f64::NAN, 0.0), (0.0, 0.0),
            (0.5, -0.01), (0.5, f64::INFINITY), (0.5, 2.0001)] {
            assert!(score_ratio(ratio, noise, cfg, PROTOCOL_ADVERSE).is_none());
        }
    }

    #[test]
    fn adverse_qualification_counts_every_window_and_preserves_raw_proof() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
        let trained = c.working;
        assert_eq!(c.training_begin(128), OK);
        measured_pair(&mut c, 1, [0.3, 0.36, 0.1]);
        let first = c.candidate.unwrap();
        assert_eq!(first.count, 1);
        assert!((first.ratios[0] - 0.1 / 0.33).abs() < 1e-14);
        assert!((first.noises[0] - 0.06 / 0.33).abs() < 1e-14);
        for id in 2..=63 {
            measured_pair(&mut c, id, [0.3, 0.3, 0.1]);
            assert_eq!(c.candidate.unwrap().count, id as u32);
            assert_eq!(c.counters[6], 0, "no early qualification");
        }
        measured_pair(&mut c, 64, [0.3, 0.3, 0.1]);
        assert_eq!(c.counters[6], 1);
        assert!(c.candidate.is_none());
        assert_eq!(c.active.actions[0], 2);
        let q = c.active_qualifications[0];
        assert_eq!(q.count, 64);
        assert_eq!(q.ratios[0], first.ratios[0]);
        assert_eq!(q.noises[0], first.noises[0]);
        let summary = comparison_summary(&q.ratios, &q.noises, q.count, c.config, c.protocol).unwrap();
        assert!(summary.passed);
        assert!((c.last_mean - (1.05 + 63.0 / 3.0) / 64.0).abs() < 1e-14);
        assert!((c.last_noise - first.noises[0] / 64.0).abs() < 1e-14);
        assert_eq!(c.last_upper, failure_upper(1, 64, c.config.family_alpha / 8.0));
        for cell in 0..CELLS { for action in 0..ACTIONS {
            assert_eq!(c.working[cell][action].count, trained[cell][action].count);
            assert_eq!(c.working[cell][action].mean, trained[cell][action].mean);
        } }
        assert_eq!(c.training_status().used, 128);
        let bytes = c.checkpoint();
        let restored = Controller::restore_with_protocol(c.config, PROTOCOL_ADVERSE, &bytes).unwrap();
        assert_eq!(restored.checkpoint(), bytes);
        assert_eq!(restored.active.actions[0], 2);
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
    }

    #[test]
    fn adverse_noisy_only_and_slower_trials_fail_at_fixed_count() {
        for costs in [[0.3, 0.36, 0.1], [0.3, 0.3, 0.29]] {
            let mut c = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
            assert_eq!(c.training_begin(128), OK);
            for id in 1..=64 {
                measured_pair(&mut c, id, costs);
                if id < 64 { assert_eq!(c.candidate.unwrap().count, id as u32); }
            }
            assert!(c.candidate.is_none());
            assert_eq!(c.counters[6], 0);
            assert_eq!(c.counters[7], 1);
            assert_eq!(c.last_outcome, 3);
            assert_eq!(c.active.actions[0], 0);
            assert_eq!(c.training_status().used, 128);
            assert_eq!(c.last_upper, 1.0);
        }
    }

    #[test]
    fn strict_noise_and_adverse_raw_regression_remain_terminal() {
        for (protocol, costs) in [(PROTOCOL_STRICT, [0.3, 0.36, 0.1]),
            (PROTOCOL_ADVERSE, [0.3, 0.3, 0.316]) ] {
            let mut c = frozen_candidate_with_protocol(6, protocol);
            c.training_begin(128);
            measured_pair(&mut c, 1, costs);
            assert!(c.candidate.is_none());
            assert_eq!(c.counters[7], 1);
            assert_eq!(c.training_status().used, 2);
        }
    }

    #[test]
    fn adverse_execution_and_numerical_failures_remain_terminal() {
        for (status, correctness, resource_ok, cost) in [(1, 1, 1, 0.1),
            (2, 1, 1, 0.1), (3, 1, 1, 0.1), (4, 1, 1, 0.1),
            (0, 0, 1, 0.1), (0, 1, 0, 0.1), (0, 1, 1, f64::NAN)] {
            let mut c = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
            c.training_begin(128);
            let (ds, _) = c.begin_training_pair(&numerical_context(1), 1).unwrap();
            c.complete(&Completion { ticket: ds[2].ticket, actual_action: ds[2].action,
                status, correctness, resource_ok, cost, completed_ns: 20 });
            assert!(c.candidate.is_none());
            assert_eq!(c.active.actions[0], 0);
            assert_eq!(c.counters[6], 0);
            assert_eq!(c.training_status().used, 2);
        }
    }

    #[test]
    fn adverse_completed_partial_windows_survive_but_inflight_does_not() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
        c.training_begin(128);
        for id in 1..=19 { measured_pair(&mut c, id, [0.3, 0.36, 0.1]); }
        let candidate = c.candidate.unwrap();
        let bytes = c.checkpoint();
        assert!(bytes.starts_with(b"IMCSTATE02"));
        assert!(Controller::restore(c.config, &bytes).is_none());
        assert!(Controller::restore_with_protocol(c.config, PROTOCOL_ADVERSE,
            &Controller::new(c.config).unwrap().checkpoint()).is_none());
        let mut restored = Controller::restore_with_protocol(c.config, PROTOCOL_ADVERSE, &bytes).unwrap();
        assert_eq!(restored.candidate.unwrap().count, 19);
        assert_eq!(restored.candidate.unwrap().ratios, candidate.ratios);
        assert_eq!(restored.candidate.unwrap().noises, candidate.noises);
        assert_eq!(restored.counters[8], c.counters[8]);
        assert_eq!(restored.checkpoint(), bytes);
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
        restored.training_begin(2);
        let (ds, _) = restored.begin_training_pair(&numerical_context(1), 20).unwrap();
        numerical_complete(&mut restored, ds[0]);
        let restored = Controller::restore_with_protocol(c.config, PROTOCOL_ADVERSE, &restored.checkpoint()).unwrap();
        assert!(restored.candidate.is_none());
        assert_eq!(restored.counters[8], c.counters[8] + 1);
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
    }

    #[test]
    fn adverse_restore_recomputes_pessimistic_proof_and_rejects_invalid_raw_arrays() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
        c.training_begin(128);
        for id in 1..=64 { measured_pair(&mut c, id, [0.3, 0.3, 0.1]); }
        for noise in [0.2, 2.0001, -0.01, f64::NAN] {
            let mut q = c.active_qualifications[0];
            q.noises[..64].fill(noise);
            c.active_qualifications[0] = q;
            assert!(Controller::restore_with_protocol(c.config, PROTOCOL_ADVERSE, &c.checkpoint()).is_none());
        }
        let mut partial = frozen_candidate_with_protocol(6, PROTOCOL_ADVERSE);
        partial.training_begin(2);
        measured_pair(&mut partial, 1, [0.3, 0.36, 0.1]);
        for noise in [2.0001, -0.01, f64::NAN] {
            partial.candidate.as_mut().unwrap().noises[0] = noise;
            assert!(Controller::restore_with_protocol(partial.config, PROTOCOL_ADVERSE, &partial.checkpoint()).is_none());
        }
    }

    #[test]
    fn protocol_ffi_refuses_invalid_values_and_cross_restore() {
        unsafe {
            assert!(imc_create_with_protocol(ptr::null(), PROTOCOL_ADVERSE).is_null());
            for protocol in [0, 5, u32::MAX] {
                assert!(imc_create_with_protocol(&config(), protocol).is_null());
                let bytes = Controller::new(config()).unwrap().checkpoint();
                assert!(imc_restore_with_protocol(&config(), protocol, bytes.as_ptr(), bytes.len()).is_null());
            }
            let handle = imc_create_with_protocol(&config(), PROTOCOL_ADVERSE);
            assert!(!handle.is_null());
            let bytes = (*handle).checkpoint();
            assert!(imc_restore(&config(), bytes.as_ptr(), bytes.len()).is_null());
            assert!(imc_restore_with_protocol(&config(), PROTOCOL_STRICT, bytes.as_ptr(), bytes.len()).is_null());
            assert!(imc_restore_with_protocol(&config(), PROTOCOL_ADVERSE, ptr::null(), 0).is_null());
            let restored = imc_restore_with_protocol(&config(), PROTOCOL_ADVERSE, bytes.as_ptr(), bytes.len());
            assert!(!restored.is_null());
            imc_free(restored);
            imc_free(handle);
        }
    }

    #[test]
    fn sequential_null_false_promotion_stays_below_nominal_alpha() {
        // Worst null case: every window is a fair coin between a clear win and a
        // clear nonwin. Optional stopping after every window must stay valid.
        let mut cfg = deployed_gates(6);
        cfg.min_pairs = 32;
        cfg.max_trials = 1;
        cfg.family_alpha = 0.1;
        let (trials, mut promoted, mut rng) = (4000, 0, 0x9e3779b97f4a7c15_u64);
        for _ in 0..trials {
            let (mut ratios, noises) = ([0.0; MAX_PAIRS], [0.0; MAX_PAIRS]);
            for n in 1..=cfg.min_pairs {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                ratios[n as usize - 1] = if rng & 1 == 0 { 0.5 } else { 1.0 };
                if n < SEQUENTIAL_MIN_WINDOWS { continue; }
                let summary = comparison_summary(&ratios, &noises, n, cfg, PROTOCOL_SEQUENTIAL).unwrap();
                if summary.passed { promoted += 1; }
                if summary.passed || summary.hopeless { break; }
            }
        }
        // The mean gate also binds here; the bound is the e-process plus fixed test.
        assert!(f64::from(promoted) / f64::from(trials) <= cfg.family_alpha, "{promoted}");
    }

    #[test]
    fn sequential_clear_winner_promotes_after_eleven_windows_not_sixty_four() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_SEQUENTIAL);
        assert_eq!(c.training_begin(128), OK);
        for id in 1..=10 {
            measured_pair(&mut c, id, [0.3, 0.3, 0.1]);
            assert_eq!(c.counters[6], 0, "window {id}");
        }
        measured_pair(&mut c, 11, [0.3, 0.3, 0.1]);
        assert_eq!(c.counters[6], 1);
        assert_eq!(c.active.actions[0], 2);
        assert_eq!(c.active_qualifications[0].count, 11);
        assert_eq!(c.training_status().used, 22);
        let bytes = c.checkpoint();
        assert!(bytes.starts_with(b"IMCSTATE03"));
        let restored = Controller::restore_with_protocol(c.config, PROTOCOL_SEQUENTIAL, &bytes).unwrap();
        assert_eq!(restored.checkpoint(), bytes);
        assert_eq!(restored.active.actions[0], 2);
        for protocol in [PROTOCOL_STRICT, PROTOCOL_ADVERSE] {
            assert!(Controller::restore_with_protocol(c.config, protocol, &bytes).is_none());
        }
        // A short proof is valid only while it actually crosses the e-process threshold.
        c.active_qualifications[0].ratios[3] = 1.0;
        assert!(Controller::restore_with_protocol(c.config, PROTOCOL_SEQUENTIAL, &c.checkpoint()).is_none());
    }

    #[test]
    fn sequential_noisy_windows_count_as_losses_and_delay_promotion() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_SEQUENTIAL);
        c.training_begin(128);
        measured_pair(&mut c, 1, [0.3, 0.36, 0.1]); // noisy control: a nonwin, not dropped
        for id in 2..=11 { measured_pair(&mut c, id, [0.3, 0.3, 0.1]); }
        assert_eq!(c.counters[6], 0);
        let mut id = 12;
        while c.counters[6] == 0 { measured_pair(&mut c, id, [0.3, 0.3, 0.1]); id += 1; }
        assert!(c.active_qualifications[0].count > 11 && c.active_qualifications[0].count < 64);
        assert!(c.active_qualifications[0].noises[0] > c.config.max_regression);
    }

    #[test]
    fn sequential_loser_stops_early_and_keeps_hard_gates() {
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_SEQUENTIAL);
        c.training_begin(128);
        let mut id = 1;
        while c.candidate.is_some() { measured_pair(&mut c, id, [0.3, 0.3, 0.29]); id += 1; }
        assert!(id - 1 < 40, "futility after {} windows", id - 1);
        assert_eq!((c.counters[6], c.counters[7], c.last_outcome), (0, 1, 3));
        assert_eq!(c.active.actions[0], 0);
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_SEQUENTIAL);
        c.training_begin(128);
        measured_pair(&mut c, 1, [0.3, 0.3, 0.316]); // actual regression stays terminal
        assert!(c.candidate.is_none());
        assert_eq!(c.counters[7], 1);
    }

    #[test]
    fn directed_training_interleaves_missing_labels_and_debits_the_grant() {
        let mut c = Controller::new_with_protocol(deployed_gates(6), PROTOCOL_SEQUENTIAL).unwrap();
        let x = numerical_context(1); // eligible actions 0 and 2
        let live = c.decide(&x);
        assert_eq!((live.action, live.kind), (0, 0), "outside a grant decide is unchanged");
        numerical_complete(&mut c, live);
        assert_eq!(c.training_begin(64), OK);
        let mut actions = vec![];
        while c.candidate.is_none() {
            let d = c.decide_training(&x);
            actions.push(d.action);
            if d.kind == 2 && d.propensity == 1.0 { assert_eq!(d.action, 2); }
            numerical_complete(&mut c, d);
            assert!(actions.len() < 64, "candidate never froze");
        }
        assert_eq!(&actions[..4], &[2, 0, 2, 0], "alternative never runs ahead of the reference");
        assert!(c.working[0][2].count >= 8, "directed labels reach min_train");
        assert_eq!(actions.len(), 32 - 1, "first freeze after freeze_every updates");
        assert_eq!(c.training_status().used, 8);
        assert_eq!(c.counters[0], 32);
        // Faulted or ineligible alternatives are never directed.
        let mut f = Controller::new_with_protocol(deployed_gates(6), PROTOCOL_SEQUENTIAL).unwrap();
        f.training_begin(8);
        f.fault_mask = 1 << 2;
        assert_eq!(f.decide_training(&x).action, 0);
        let mut only = numerical_context(1);
        only.eligible_mask = 1;
        assert_eq!(f.decide_training(&only).action, 0);
        assert_eq!(f.training_status().used, 0);
    }

    fn labelled(c: &mut Controller, cell: usize, reference: f64, alternative: (usize, f64)) {
        c.working[cell][0] = Statistic { count: 8, mean: reference };
        c.working[cell][alternative.0] = Statistic { count: 8, mean: alternative.1 };
    }

    #[test]
    fn sequential_freezes_the_largest_untried_gain_and_rotates_cells() {
        let mut c = Controller::new_with_protocol(deployed_gates(6), PROTOCOL_SEQUENTIAL).unwrap();
        labelled(&mut c, 0, 1.0, (2, 0.88));  // singleton: 12 % estimated gain
        labelled(&mut c, 4, 4.0, (1, 3.0));   // four requests grouped: 25 %
        c.counters[2] = 32;
        c.freeze();
        assert_eq!((c.candidate.unwrap().cell, c.candidate.unwrap().action), (4, 1));
        c.reject_trial();
        c.counters[2] = 64;
        c.freeze();
        assert_eq!(c.candidate.unwrap().cell, 0, "a rejected cell waits for the others");
        c.invalidate();
        c.counters[2] = 96;
        c.freeze();
        assert_eq!(c.candidate.unwrap().cell, 4, "a new round starts once every cell had a trial");
        c.counters[1] = c.counters[2]; // consistent observation count for restore validation
        let bytes = c.checkpoint();
        let restored = Controller::restore_with_protocol(c.config, PROTOCOL_SEQUENTIAL, &bytes).unwrap();
        assert_eq!(restored.tried_cells, c.tried_cells);
        assert_eq!(restored.checkpoint(), bytes);
        // Strict and adverse keep the index order and the first cell.
        for protocol in [PROTOCOL_STRICT, PROTOCOL_ADVERSE] {
            let mut legacy = Controller::new_with_protocol(deployed_gates(6), protocol).unwrap();
            labelled(&mut legacy, 0, 1.0, (2, 0.88));
            labelled(&mut legacy, 4, 4.0, (1, 3.0));
            legacy.counters[2] = 32;
            legacy.freeze();
            assert_eq!(legacy.candidate.unwrap().cell, 0);
        }
    }

    #[test]
    fn noise_protocol_null_false_promotion_stays_below_alpha_with_noisy_arms() {
        // Exchangeable arms (no true effect) with 10 % lognormal-like jitter.
        let mut cfg = deployed_gates(6);
        cfg.min_pairs = 32;
        cfg.max_trials = 1;
        cfg.family_alpha = 0.1;
        let (trials, mut promoted, mut rng) = (3000, 0, 0x2545f4914f6cdd1d_u64);
        let mut draw = |rng: &mut u64| {
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            1.0 + 0.2 * ((*rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * 0.5
        };
        for _ in 0..trials {
            let (mut ratios, mut noises) = ([0.0; MAX_PAIRS], [0.0; MAX_PAIRS]);
            for n in 1..=cfg.min_pairs {
                let (a, aa, b) = (draw(&mut rng), draw(&mut rng), draw(&mut rng));
                let reference = (a + aa) / 2.0;
                ratios[n as usize - 1] = (b / reference).min(1.0 + cfg.max_regression);
                noises[n as usize - 1] = (a - aa).abs() / reference;
                if n < SEQUENTIAL_MIN_WINDOWS { continue; }
                let summary = comparison_summary(&ratios, &noises, n, cfg, PROTOCOL_NOISE).unwrap();
                if summary.passed { promoted += 1; }
                if summary.passed || summary.hopeless { break; }
            }
        }
        assert!(f64::from(promoted) / f64::from(trials) <= cfg.family_alpha, "{promoted}");
    }

    #[test]
    fn noise_protocol_keeps_noisy_wins_that_v3_forces_into_losses() {
        let cfg = deployed_gates(6);
        assert_eq!(score_ratio(0.8, 0.12, cfg, PROTOCOL_NOISE), Some(0.8));
        assert_eq!(score_ratio(0.8, 0.12, cfg, PROTOCOL_SEQUENTIAL), Some(1.05));
        assert!(score_ratio(1.06, 0.0, cfg, PROTOCOL_NOISE).is_none(), "hard ratio cap unchanged");
        let mut c = frozen_candidate_with_protocol(6, PROTOCOL_NOISE);
        c.training_begin(256);
        let mut id = 1;
        while c.counters[6] == 0 && id <= 64 {
            // ratio 0.8, noise 0.1: a win (0.9 < 0.95) here, a forced loss under v3
            measured_pair(&mut c, id, [0.3, 0.33, 0.252]);
            id += 1;
        }
        assert_eq!(c.counters[6], 1);
        assert_eq!(c.active_qualifications[0].count, 11);
        let bytes = c.checkpoint();
        assert!(bytes.starts_with(b"IMCSTATE04"));
        assert!(Controller::restore_with_protocol(c.config, PROTOCOL_NOISE, &bytes).is_some());
        assert!(Controller::restore_with_protocol(c.config, PROTOCOL_SEQUENTIAL, &bytes).is_none());
    }

    fn earn_reference_credits(c: &mut Controller, target: u64) {
        let mut x = numerical_context(1);
        x.eligible_mask = 1;
        while c.credit_units < target {
            let d = c.decide(&x);
            numerical_complete(c, d);
        }
    }

    #[test]
    fn frozen_candidate_reserves_comparison_credits_and_allows_extra_exploration() {
        let mut c = frozen_candidate(6);
        let (ds, _) = c.begin_pair(&numerical_context(1), 1).unwrap();
        for d in ds { numerical_complete(&mut c, d); }
        earn_reference_credits(&mut c, 2 * UNIT);
        let initial_rng = c.random;
        for _ in 0..10 {
            let before = c.credit_units;
            let d = c.decide(&numerical_context(4));
            assert_eq!(d.kind, 0);
            assert_eq!(d.propensity, 1.0);
            assert_eq!(c.credit_units, before + 50_000);
            numerical_complete(&mut c, d);
        }
        assert_eq!(c.random, initial_rng, "budget filtering precedes a random draw");
        earn_reference_credits(&mut c, 6 * UNIT);
        assert_eq!(c.begin_pair(&numerical_context(4), 2).unwrap_err(), UNAVAILABLE);
        let mut explored = false;
        for _ in 0..4096 {
            let before = c.credit_units;
            let d = c.decide(&numerical_context(4));
            let earned = (before + 50_000).min(6 * UNIT);
            assert_eq!(c.credit_units, earned - if d.kind == 2 { UNIT } else { 0 });
            assert!(c.credit_units >= 2 * UNIT);
            numerical_complete(&mut c, d);
            if d.kind == 2 { explored = true; break; }
        }
        assert!(explored, "another cell can still use credits above the reserve");
    }

    #[test]
    fn reserve_works_with_minimum_credit_cap_and_restored_candidate() {
        let original = frozen_candidate(2);
        let mut c = Controller::restore(deployed_gates(2), &original.checkpoint()).unwrap();
        let version = c.candidate.unwrap().policy.version;
        for _ in 0..100 {
            let before = c.credit_units;
            let d = c.decide(&numerical_context(4));
            assert_ne!(d.kind, 2);
            assert_eq!(d.propensity, 1.0);
            assert_eq!(c.credit_units, (before + 50_000).min(2 * UNIT));
            numerical_complete(&mut c, d);
        }
        assert_eq!(c.credit_units, 2 * UNIT);
        assert_eq!(c.candidate.unwrap().policy.version, version);
        let before = c.credit_units;
        let (ds, _) = c.begin_pair(&numerical_context(1), 1).unwrap();
        assert_eq!(c.credit_units, before - 2 * UNIT);
        for d in ds { numerical_complete(&mut c, d); }
        assert_eq!(c.candidate.unwrap().count, 1);
    }

    #[test]
    fn no_candidate_keeps_existing_exploration_budget_behavior() {
        let mut c = Controller::new(deployed_gates(6)).unwrap();
        let mut explored = false;
        for _ in 0..4096 {
            let before = c.credit_units;
            let d = c.decide(&numerical_context(1));
            assert_eq!(c.credit_units,
                (before + 50_000).min(6 * UNIT) - if d.kind == 2 { UNIT } else { 0 });
            if d.kind == 2 { explored = true; }
            assert!(c.candidate.is_none());
            if explored { break; }
        }
        assert!(explored);
    }

    #[test]
    fn reserved_credits_complete_unchanged_eight_and_sixty_four_gates() {
        let mut c = Controller::new(deployed_gates(6)).unwrap();
        let x = numerical_context(1);
        let mut ordinary = 0u64;
        let mut exploration = 0u64;
        let mut pairs = 0u64;
        for _ in 0..4096 {
            if c.candidate.is_some() && c.credit_units >= 2 * UNIT {
                let (ds, _) = c.begin_pair(&x, pairs + 1).unwrap();
                pairs += 1;
                for d in ds { numerical_complete(&mut c, d); }
            } else {
                let d = c.decide(&x);
                ordinary += 1;
                exploration += u64::from(d.kind == 2);
                numerical_complete(&mut c, d);
            }
            assert_eq!(c.credit_units, ordinary * 50_000 - exploration * UNIT - pairs * 2 * UNIT);
            if c.counters[6] != 0 { break; }
        }
        assert_eq!(c.config.min_train, 8);
        assert_eq!(c.config.min_pairs, 64);
        assert_eq!(pairs, 64);
        assert_eq!(exploration, 8);
        assert_eq!(c.counters[6], 1);
        assert_eq!(c.active.actions[0], 2);
        assert_eq!(c.working[0][2].count, 8, "holdout did not train the frozen model");
    }

    #[test]
    fn training_grant_is_bounded_once_and_requires_idle_enabled_handle() {
        let mut c = Controller::new(deployed_gates(6)).unwrap();
        for limit in [0, 1, 3, 1026, 32770] {
            assert_eq!(c.training_begin(limit), INPUT);
            assert_eq!(c.training_status(), TrainingBudgetStatus::default());
        }
        let ticket = c.decide(&numerical_context(1));
        assert_eq!(c.training_begin(2), UNAVAILABLE);
        numerical_complete(&mut c, ticket);
        assert_eq!(c.training_begin(1024), OK);
        assert_eq!(c.training_begin(2), UNAVAILABLE);
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 1024, used: 0, remaining: 1024, active: 1 });
        assert_eq!(c.training_end(), OK);
        assert_eq!(c.training_end(), OK);
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 1024, used: 0, remaining: 0, active: 0 });
        assert_eq!(c.training_begin(2), UNAVAILABLE);
        let mut disabled = deployed_gates(6);
        disabled.enabled = 0;
        assert_eq!(Controller::new(disabled).unwrap().training_begin(2), UNAVAILABLE);
        let mut killed = Controller::new(deployed_gates(6)).unwrap();
        killed.fault(4, 0);
        assert_eq!(killed.training_begin(2), UNAVAILABLE);
        let mut maximum = deployed_gates(6);
        maximum.min_pairs = 256;
        maximum.max_trials = 64;
        let mut c = Controller::new(maximum).unwrap();
        assert_eq!(c.training_begin(32768), OK);
    }

    #[test]
    fn offline_pairs_debit_only_offline_budget_after_complete_reservation() {
        let mut c = frozen_candidate(6);
        let (live, _) = c.begin_pair(&numerical_context(1), 1).unwrap();
        assert_eq!(c.training_begin(2), UNAVAILABLE);
        for d in live { numerical_complete(&mut c, d); }
        assert!(c.credit_units < 2 * UNIT);
        let checkpoint = c.checkpoint();
        assert_eq!(c.training_begin(2), OK);
        assert_eq!(c.checkpoint(), checkpoint, "a grant changes no persisted numerical/live state");
        assert_eq!(c.begin_pair(&numerical_context(1), 2).unwrap_err(), UNAVAILABLE);
        let granted = c.training_status();
        let mut foreign = numerical_context(1);
        foreign.identity = [8; 32];
        assert_eq!(c.begin_training_pair(&foreign, 2).unwrap_err(), INPUT);
        assert_eq!(c.begin_training_pair(&numerical_context(4), 2).unwrap_err(), UNAVAILABLE);
        assert_eq!(c.begin_training_pair(&numerical_context(1), 1).unwrap_err(), INPUT);
        assert_eq!(c.training_status(), granted);
        let credits = c.credit_units;
        let ordinary = c.decide(&numerical_context(1));
        numerical_complete(&mut c, ordinary);
        assert_eq!(c.credit_units, credits + 50_000);
        assert_eq!(c.training_status(), granted, "ordinary decisions never consume offline quota");
        let credits = c.credit_units;
        let (ds, _) = c.begin_training_pair(&numerical_context(1), 2).unwrap();
        assert_eq!(c.credit_units, credits);
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 2, used: 2, remaining: 0, active: 0 });
        assert_eq!(c.begin_training_pair(&numerical_context(1), 3).unwrap_err(), UNAVAILABLE);
        for d in ds { numerical_complete(&mut c, d); }
        assert_eq!(c.candidate.unwrap().count, 2);
        assert_eq!(c.training_begin(2), UNAVAILABLE);
    }

    #[test]
    fn live_pair_never_spends_an_active_offline_grant() {
        let mut c = frozen_candidate(6);
        assert_eq!(c.training_begin(4), OK);
        let grant = c.training_status();
        let live = c.credit_units;
        let trials = c.trials_started;
        let (ds, _) = c.begin_pair(&numerical_context(1), 1).unwrap();
        assert_eq!(c.credit_units, live - 2 * UNIT);
        assert_eq!(c.training_status(), grant);
        for d in ds { numerical_complete(&mut c, d); }
        assert_eq!(c.trials_started, trials);
        assert_eq!(c.begin_pair(&numerical_context(1), 2).unwrap_err(), UNAVAILABLE);
        assert_eq!(c.training_status(), grant);
        let (ds, _) = c.begin_training_pair(&numerical_context(1), 2).unwrap();
        for d in ds { numerical_complete(&mut c, d); }
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 4, used: 2, remaining: 2, active: 1 });
        assert_eq!(c.credit_units, live - 2 * UNIT);
        assert_eq!(c.trials_started, trials);
    }

    #[test]
    fn failed_offline_pair_retains_debit_and_kill_revokes_unused_quota() {
        let mut c = frozen_candidate(6);
        assert_eq!(c.training_begin(6), OK);
        let (ds, _) = c.begin_training_pair(&numerical_context(1), 1).unwrap();
        assert_eq!(c.complete(&Completion { ticket: ds[0].ticket, actual_action: ds[0].action,
            status: 4, correctness: 1, resource_ok: 1, cost: 0.3, completed_ns: 20 }), OK);
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 6, used: 2, remaining: 4, active: 1 });
        assert_eq!(c.counters[8], 1);
        assert!(c.candidate.is_none());
        c.fault(4, 0);
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 6, used: 2, remaining: 0, active: 0 });
        assert_eq!(c.training_begin(2), UNAVAILABLE);
        let restored = Controller::restore(deployed_gates(6), &c.checkpoint()).unwrap();
        assert!(restored.killed);
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
    }

    #[test]
    fn offline_authority_never_survives_restore_but_full_qualification_does() {
        let mut c = frozen_candidate(6);
        let counts = c.working[0][2].count;
        let credits = c.credit_units;
        let version = c.candidate.unwrap().policy.version;
        assert_eq!(c.training_begin(130), OK);
        let bytes = c.checkpoint();
        let mut restored = Controller::restore(deployed_gates(6), &bytes).unwrap();
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
        assert_eq!(restored.candidate.unwrap().policy.version, version);
        assert_eq!(restored.working[0][2].count, counts);
        assert_eq!(restored.begin_training_pair(&numerical_context(1), 1).unwrap_err(), UNAVAILABLE);
        for id in 1..=64 {
            let (ds, _) = c.begin_training_pair(&numerical_context(1), id).unwrap();
            for d in ds { numerical_complete(&mut c, d); }
            assert_eq!(c.credit_units, credits);
            assert_eq!(c.training_budget.used, id as u32 * 2);
            assert_eq!(c.training_budget.used + c.training_budget.remaining, 130);
        }
        assert_eq!(c.counters[6], 1);
        assert_eq!(c.active.actions[0], 2);
        assert_eq!(c.working[0][2].count, counts);
        let bytes = c.checkpoint();
        let restored = Controller::restore(deployed_gates(6), &bytes).unwrap();
        assert_eq!(restored.training_status(), TrainingBudgetStatus::default());
        assert_eq!(restored.checkpoint(), bytes);
        assert_eq!(restored.active.actions[0], 2);
        assert_eq!(restored.counters[6], 1);
        c.training_end();
        assert_eq!(c.training_status(), TrainingBudgetStatus { granted: 130, used: 128, remaining: 0, active: 0 });
    }

    #[test]
    fn new_training_ffi_is_optional_and_preserves_existing_struct_layouts() {
        assert_eq!(size_of::<TrainingBudgetStatus>(), 16);
        unsafe {
            assert_eq!(imc_training_begin(ptr::null_mut(), 2), INPUT);
            assert_eq!(imc_training_end(ptr::null_mut()), INPUT);
            assert_eq!(imc_training_status(ptr::null(), ptr::null_mut()), INPUT);
            let handle = imc_create(&deployed_gates(6));
            let mut status = TrainingBudgetStatus::default();
            assert_eq!(imc_training_begin(handle, 2), OK);
            assert_eq!(imc_training_status(handle, &mut status), OK);
            assert_eq!(status.remaining, 2);
            assert_eq!(imc_training_end(handle), OK);
            assert_eq!(imc_training_status(handle, &mut status), OK);
            assert_eq!(status, TrainingBudgetStatus { granted: 2, used: 0, remaining: 0, active: 0 });
            imc_free(handle);
        }
    }
}
