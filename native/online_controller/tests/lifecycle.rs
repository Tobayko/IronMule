use ironmule_online_controller::{Completion, Config, Context, Controller, Decision};

fn config() -> Config {
    Config {
        abi_version: 1, enabled: 1, identity: [1; 32], action_digest: [2; 32],
        schema_digest: [3; 32], objective_digest: [4; 32], action_count: 3,
        min_train: 2, freeze_every: 4, min_pairs: 8, max_trials: 4,
        exploration_ppm: 50_000, trial_budget: 256, min_gain: 0.05,
        max_regression: 0.1, learning_rate: 0.2, family_alpha: 0.05, seed: 7,
    }
}
fn context() -> Context {
    Context { abi_version: 1, feature_valid: 3, identity: [1; 32], workload_digest: [9; 32],
        eligible_mask: 7, request_count: 4, max_tokens: 32, phase: 1, now_ns: 10 }
}
fn done(c: &mut Controller, d: Decision, cost: f64) -> i32 {
    c.complete(&Completion { ticket: d.ticket, actual_action: d.action, status: 0,
        correctness: 1, resource_ok: 1, cost, completed_ns: 20 })
}
fn train(c: &mut Controller) {
    for _ in 0..20_000 {
        let d = c.decide(&context());
        assert_ne!(d.ticket, 0);
        assert_eq!(done(c, d, if d.action == 1 { 50.0 } else { 100.0 }), 0);
        if c.status().candidate_version != 0 { return; }
    }
    panic!("bounded seeded exploration did not produce a candidate");
}
fn accrue(c: &mut Controller) {
    let mut x = context();
    x.eligible_mask = 1;
    while c.status().credits < 2 {
        let d = c.decide(&x);
        assert_eq!(done(c, d, 100.0), 0);
    }
}
fn pair(c: &mut Controller, id: u64, a: f64, aa: f64, b: f64) {
    accrue(c);
    let (ds, order) = c.begin_pair(&context(), id).unwrap();
    assert_eq!(order, ((id - 1) % 2) as u32);
    assert!(ds.iter().all(|d| d.ticket > c.status().frozen_after));
    // Completion order differs from submission order: ticket attribution is
    // independent of whichever policy is active when a result arrives.
    assert_eq!(done(c, ds[2], b), 0);
    assert_eq!(done(c, ds[0], a), 0);
    assert_eq!(done(c, ds[1], aa), 0);
}
fn promote(c: &mut Controller) {
    train(c);
    for id in 1..=8 { pair(c, id, 100.0, 100.0, 50.0); }
    assert_eq!(c.status().promoted, 1);
}

#[test]
fn config_validation_rejects_every_invalid_contract_axis() {
    let mut bad = config(); bad.identity = [0; 32]; assert!(Controller::new(bad).is_none());
    bad = config(); bad.action_count = 5; assert!(Controller::new(bad).is_none());
    bad = config(); bad.min_pairs = 7; assert!(Controller::new(bad).is_none());
    bad = config(); bad.min_pairs = 257; assert!(Controller::new(bad).is_none());
    bad = config(); bad.max_trials = 0; assert!(Controller::new(bad).is_none());
    bad = config(); bad.exploration_ppm = 50_001; assert!(Controller::new(bad).is_none());
    bad = config(); bad.min_gain = f64::NAN; assert!(Controller::new(bad).is_none());
    bad = config(); bad.learning_rate = 0.0; assert!(Controller::new(bad).is_none());
    bad = config(); bad.family_alpha = 0.11; assert!(Controller::new(bad).is_none());
    bad = config(); bad.enabled = 2; assert!(Controller::new(bad).is_none());
}

#[test]
fn features_identity_and_technical_mask_fail_closed() {
    let c = Controller::new(config()).unwrap();
    let mut x = context(); x.feature_valid = 1; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.identity[0] = 4; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.eligible_mask = 6; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.eligible_mask = 9; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.request_count = 0; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.max_tokens = 0; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.phase = 2; assert_eq!(c.predict(&x).reason, 3);
    x = context(); x.workload_digest = [0; 32]; assert_eq!(c.predict(&x).reason, 3);
}

#[test]
fn operational_results_train_but_never_qualify() {
    let mut c = Controller::new(config()).unwrap(); train(&mut c);
    let frozen = c.status().candidate_version;
    for _ in 0..100 { let d = c.decide(&context()); done(&mut c, d, 1.0); }
    let s = c.status();
    assert!(s.updates > 0); assert_eq!(s.active_version, 0);
    assert_eq!(s.promoted, 0); assert_eq!(s.candidate_version, frozen);
}

#[test]
fn prospective_holdout_is_fixed_and_separate_then_publishes_one_cell() {
    let mut c = Controller::new(config()).unwrap(); train(&mut c);
    accrue(&mut c);
    let updates = c.status().updates;
    let counts = c.status().working_counts;
    let (ds, _) = c.begin_pair(&context(), 1).unwrap();
    assert_eq!(done(&mut c, ds[0], 100.0), 0);
    assert_eq!(done(&mut c, ds[1], 100.0), 0);
    assert_eq!(done(&mut c, ds[2], 50.0), 0);
    assert_eq!(c.status().updates, updates);
    assert_eq!(c.status().working_counts, counts);
    assert_eq!(c.status().promoted, 0);
    for id in 2..=8 { pair(&mut c, id, 100.0, 100.0, 50.0); }
    let s = c.status();
    assert_eq!(s.promoted, 1); assert_eq!(s.active_actions, [0,0,0,0,1,0]);
    assert!(s.upper_bound < 0.5);
    assert_eq!(c.predict(&context()).action, 1);
    let mut x = context(); x.max_tokens = 33;
    assert_eq!(c.predict(&x).action, 0);
    x = context(); x.eligible_mask = 1;
    assert_eq!(c.predict(&x).action, 0);
}

#[test]
fn losing_noisy_or_uncertain_holdout_never_promotes() {
    for (a, aa, b) in [(100.0, 100.0, 120.0), (100.0, 130.0, 50.0), (100.0, 100.0, 96.0)] {
        let mut c = Controller::new(config()).unwrap(); train(&mut c);
        for id in 1..=8 {
            pair(&mut c, id, a, aa, b);
            if c.status().candidate_version == 0 { break; }
        }
        assert_eq!(c.status().promoted, 0);
        assert_eq!(c.status().rejected, 1);
        assert_eq!(c.predict(&context()).action, 0);
    }
}

#[test]
fn holdout_wrong_cell_foreign_context_and_reused_id_refuse() {
    let mut c = Controller::new(config()).unwrap(); train(&mut c); accrue(&mut c);
    let mut x = context(); x.max_tokens = 33;
    assert_eq!(c.begin_pair(&x, 1).unwrap_err(), 3);
    x = context(); x.identity = [5; 32];
    assert_eq!(c.begin_pair(&x, 1).unwrap_err(), 1);
    x = context();
    let (ds, _) = c.begin_pair(&x, 1).unwrap();
    assert_eq!(c.begin_pair(&x, 2).unwrap_err(), 3);
    for (d, cost) in ds.into_iter().zip([100.0,100.0,50.0]) { done(&mut c, d, cost); }
    accrue(&mut c);
    assert_eq!(c.begin_pair(&x, 1).unwrap_err(), 1);
}

#[test]
fn exploration_and_comparisons_share_a_real_bounded_budget() {
    let mut c = Controller::new(config()).unwrap();
    for i in 1..=2000 {
        let d = c.decide(&context());
        assert!(d.propensity > 0.0 && d.propensity <= 1.0);
        done(&mut c, d, if d.action == 1 {50.0} else {100.0});
        let s = c.status();
        let exploratory = s.working_counts[17] + s.working_counts[18];
        assert!(exploratory <= (i / 20) as u64);
    }
    let mut disabled = config(); disabled.exploration_ppm = 0;
    let mut c = Controller::new(disabled).unwrap();
    for _ in 0..1000 { let d = c.decide(&context()); assert_eq!(d.action, 0); done(&mut c,d,100.0); }
    assert_eq!(c.status().credits, 0);
}

#[test]
fn interrupted_mixed_override_and_duplicate_observations_have_no_labels() {
    let mut c = Controller::new(config()).unwrap();
    let d = c.decide(&context());
    let x = Completion { ticket: d.ticket, actual_action: u32::MAX, status: 4,
        correctness: 1, resource_ok: 1, cost: 0.0, completed_ns: 20 };
    assert_eq!(c.complete(&x), 0); assert_eq!(c.complete(&x), 2);
    let s = c.status(); assert_eq!(s.updates,0); assert_eq!(s.abandoned,1);
    assert_eq!(s.overridden,1); assert_eq!(s.pending,0);
    let d = c.decide(&context());
    assert_eq!(c.complete(&Completion { ticket:d.ticket, actual_action:2, status:0,
        correctness:1,resource_ok:1,cost:50.0,completed_ns:20 }),1);
    assert_eq!(c.status().updates,0);
}

#[test]
fn pending_overflow_returns_reference_without_blocking_or_charging_exploration() {
    let mut c = Controller::new(config()).unwrap();
    let mut x = context(); x.eligible_mask = 1;
    for _ in 0..256 { assert_ne!(c.decide(&x).ticket,0); }
    let d = c.decide(&x); assert_eq!(d.ticket,0); assert_eq!(d.action,0);
    assert_eq!(d.reason,6); assert_eq!(c.status().dropped,1);
    assert_eq!(c.status().pending,256);
}

#[test]
fn invalid_numerical_and_future_feature_feedback_is_consumed_once() {
    for cost in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1e16] {
        let mut c = Controller::new(config()).unwrap(); let d = c.decide(&context());
        assert_eq!(done(&mut c,d,cost),1); assert_eq!(done(&mut c,d,10.0),2);
        assert_eq!(c.status().updates,0);
    }
    let mut c = Controller::new(config()).unwrap(); let d = c.decide(&context());
    assert_eq!(c.complete(&Completion { ticket:d.ticket,actual_action:d.action,status:0,
        correctness:1,resource_ok:1,cost:10.0,completed_ns:9 }),1);
}

#[test]
fn holdout_failure_and_abandonment_abort_without_reuse() {
    for interrupted in [false,true] {
        let mut c = Controller::new(config()).unwrap(); train(&mut c); accrue(&mut c);
        let (ds,_) = c.begin_pair(&context(),1).unwrap();
        assert_eq!(c.complete(&Completion {ticket:ds[2].ticket,actual_action:ds[2].action,
            status:if interrupted {4} else {2},correctness:1,resource_ok:1,cost:200.0,completed_ns:20}),
            if interrupted {0} else {1});
        assert_eq!(c.status().candidate_version,0); assert_eq!(c.status().invalid_trials,1);
        assert_eq!(c.status().pending,0); assert_eq!(done(&mut c,ds[0],100.0),2);
    }
}

#[test]
fn restart_preserves_working_policy_rng_credits_kill_and_faults() {
    let mut c = Controller::new(config()).unwrap(); promote(&mut c);
    let frozen_before = c.checkpoint();
    let mut restored = Controller::restore(config(),&frozen_before).unwrap();
    assert_eq!(restored.checkpoint(),frozen_before);
    assert_eq!(restored.predict(&context()).action,1);
    assert_eq!(restored.decide(&context()),c.decide(&context()));
    c.fault(4,0); let saved = c.checkpoint();
    let c = Controller::restore(config(),&saved).unwrap();
    assert_eq!(c.predict(&context()).reason,2); assert_eq!(c.status().killed,1);
}

#[test]
fn checkpoint_corruption_truncation_trailing_unknown_identity_and_config_refuse() {
    let c = Controller::new(config()).unwrap(); let saved = c.checkpoint();
    for index in [0,10,100,saved.len()-1] {
        let mut b = saved.clone(); b[index] ^= 1;
        assert!(Controller::restore(config(),&b).is_none());
    }
    for cut in [0,10,100,saved.len()-1] { assert!(Controller::restore(config(),&saved[..cut]).is_none()); }
    let mut b=saved.clone(); b.push(0); assert!(Controller::restore(config(),&b).is_none());
    let mut cfg=config(); cfg.identity[0]=8; assert!(Controller::restore(cfg,&saved).is_none());
    cfg=config(); cfg.min_gain=0.01; assert!(Controller::restore(cfg,&saved).is_none());
}

#[test]
fn restart_abandons_inflight_observations_and_invalidates_partial_holdout() {
    let mut c=Controller::new(config()).unwrap(); train(&mut c); accrue(&mut c);
    let (ds,_)=c.begin_pair(&context(),1).unwrap(); done(&mut c,ds[0],100.0);
    let ordinary=c.decide(&context()); let before=c.status();
    let mut c=Controller::restore(config(),&c.checkpoint()).unwrap();
    assert_eq!(c.status().pending,0); assert_eq!(c.status().candidate_version,0);
    assert_eq!(c.status().invalid_trials,1); assert_eq!(c.status().abandoned,before.abandoned+3);
    assert_eq!(done(&mut c,ordinary,50.0),2); assert_eq!(done(&mut c,ds[1],100.0),2);
    assert_eq!(c.status().promoted,0);
}

#[test]
fn frozen_predict_has_no_learning_side_effects() {
    let mut c=Controller::new(config()).unwrap(); promote(&mut c);
    let before=c.checkpoint(); for _ in 0..100 { assert_eq!(c.predict(&context()).action,1); }
    assert_eq!(c.checkpoint(),before);
}

#[test]
fn drift_recovers_automatically_but_correctness_fault_cannot_requalify_action() {
    let mut c=Controller::new(config()).unwrap(); promote(&mut c);
    let mut x=context(); x.eligible_mask=3;
    for _ in 0..20 { let d=c.decide(&x); done(&mut c,d,300.0); }
    assert_eq!(c.predict(&x).action,0); assert_eq!(c.status().fault_mask,0);
    c.fault(2,1);
    for _ in 0..500 { let d=c.decide(&x); assert_eq!(d.action,0); done(&mut c,d,100.0); }
    assert_eq!(c.status().fault_mask,2);
    let c=Controller::restore(config(),&c.checkpoint()).unwrap();
    assert_eq!(c.status().fault_mask,2); assert_eq!(c.predict(&x).action,0);
}

#[test]
fn disabled_controller_is_inert_and_has_no_ticket() {
    let mut cfg=config(); cfg.enabled=0; let mut c=Controller::new(cfg).unwrap();
    let d=c.decide(&context()); assert_eq!(d.action,0); assert_eq!(d.ticket,0);
    assert_eq!(d.reason,1); assert_eq!(c.status().updates,0);
}

#[test]
fn delayed_old_policy_outcomes_train_their_cell_but_cannot_drift_new_policy() {
    let mut c=Controller::new(config()).unwrap(); train(&mut c);
    let mut delayed=Vec::new();
    for _ in 0..20_000 {
        let d=c.decide(&context());
        if d.action==1 { delayed.push(d); } else { done(&mut c,d,100.0); }
        if delayed.len()==8 { break; }
    }
    assert_eq!(delayed.len(),8);
    for id in 1..=8 { pair(&mut c,id,100.0,100.0,50.0); }
    assert_eq!(c.predict(&context()).action,1);
    let updates=c.status().updates;
    for d in delayed { assert_eq!(done(&mut c,d,300.0),0); }
    assert_eq!(c.status().updates,updates+8);
    assert_eq!(c.predict(&context()).action,1);
}

#[test]
fn fixed_trial_cap_survives_rejection_and_restart() {
    let mut cfg=config(); cfg.max_trials=1;
    let mut c=Controller::new(cfg).unwrap(); train(&mut c);
    pair(&mut c,1,100.0,100.0,200.0);
    assert_eq!(c.status().rejected,1);
    let mut c=Controller::restore(cfg,&c.checkpoint()).unwrap();
    for _ in 0..2000 { let d=c.decide(&context()); done(&mut c,d,if d.action==1 {50.0} else {100.0}); }
    assert_eq!(c.status().trials_started,1); assert_eq!(c.status().candidate_version,0);
    assert_eq!(c.status().promoted,0);
}

#[test]
fn current_policy_resource_failure_withdraws_but_old_epoch_failure_does_not() {
    let mut c=Controller::new(config()).unwrap(); train(&mut c);
    let delayed=loop {
        let d=c.decide(&context());
        if d.action==1 { break d; }
        done(&mut c,d,100.0);
    };
    for id in 1..=8 { pair(&mut c,id,100.0,100.0,50.0); }
    let updates=c.status().updates;
    assert_eq!(c.complete(&Completion {ticket:delayed.ticket,actual_action:1,status:0,
        correctness:1,resource_ok:0,cost:300.0,completed_ns:20}),1);
    assert_eq!(c.predict(&context()).action,1);
    let current=loop {
        let d=c.decide(&context());
        if d.action==1 { break d; }
        done(&mut c,d,100.0);
    };
    assert_eq!(c.complete(&Completion {ticket:current.ticket,actual_action:1,status:0,
        correctness:1,resource_ok:0,cost:300.0,completed_ns:20}),1);
    assert_eq!(c.predict(&context()).action,0);
    assert_eq!(c.status().fault_mask,0);
    assert_eq!(c.status().updates,updates);
}
