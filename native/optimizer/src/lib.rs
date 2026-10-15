//! OPT1: an opt-in optimization coordinator with a Rust core (backlog OPT1).
//!
//! Cycle: detect capabilities and identity -> measure the reference -> choose candidates within a
//! bounded action set -> run each isolated in a child process -> append results to a hash-chained
//! journal -> learn -> qualify through an evaluator the policy cannot modify -> activate a reversible
//! setting or keep the reference -> roll back. Adapters are executables speaking a JSON protocol
//! (`adapter`). Learning: the existing RSI1 planner as a bandit (`ironmule_experiment_planner`) and a
//! tabular Q-learning candidate for sequential decisions (`rl`); `dream` replays recorded discovery
//! trees to choose learner settings (a paper-based extension of Dream-RSI, see its module notes).
//! Std only, so it builds offline like the other native crates.

pub mod adapter;
pub mod calibrate;
pub mod campaign;
pub mod coordinator;
pub mod cpujob;
pub mod dream;
pub mod episode;
pub mod experiment;
pub mod identity;
pub mod json;
pub mod proc;
pub mod qualify;
pub mod rulecheck;
pub mod rl;
pub mod sha256;
pub mod store;
