#ifndef IRONMULE_ONLINE_CONTROLLER_H
#define IRONMULE_ONLINE_CONTROLLER_H
#include <stddef.h>
#include <stdint.h>

/* ABI 1. Action 0 is the unchanged caller reference. The caller owns
 * serialization: calls on the same handle must never overlap. No strings,
 * Python objects, GPU handles, or file paths cross this boundary. */
#define IMC_ABI 1
#define IMC_MAX_ACTIONS 4
#define IMC_CELLS 6
#define IMC_REQUIRED_FEATURES 3
#define IMC_SAFE_SERVE 1
enum imc_protocol { IMC_PROTOCOL_STRICT=1, IMC_PROTOCOL_ADVERSE=2, IMC_PROTOCOL_SEQUENTIAL=3, IMC_PROTOCOL_NOISE=4 };

/* Return codes: 0 OK, 1 invalid input, 2 unknown/already completed ticket,
 * 3 unavailable/budget/busy, 4 incompatible/corrupt checkpoint,
 * 5 output buffer too small. Errors never qualify a policy. */
enum imc_kind { IMC_REFERENCE=0, IMC_ACTIVE=1, IMC_EXPLORE=2,
  IMC_HOLDOUT_A=3, IMC_HOLDOUT_AA=4, IMC_HOLDOUT_B=5 };
enum imc_reason { IMC_OK=0, IMC_DISABLED=1, IMC_KILLED=2,
  IMC_BAD_CONTEXT=3, IMC_FAULTED=4, IMC_INELIGIBLE=5,
  IMC_PENDING_FULL=6, IMC_BUDGET=7 };
/* Completion statuses: success 0; error 1; timeout 2; cancelled 3;
 * interrupted 4. Flags are strict 0/1. Failed ordinary work receives its
 * configured objective penalty but cannot qualify; failed holdout aborts. */
enum imc_fault_kind { IMC_DRIFT=1, IMC_CORRECTNESS=2,
  IMC_NUMERICAL=3, IMC_OPERATOR_KILL=4 };
enum imc_trial_outcome { IMC_NO_TRIAL=0, IMC_QUALIFYING=1,
  IMC_PROMOTED=2, IMC_REJECTED=3, IMC_INVALID=4 };

typedef struct {
  uint32_t abi_version, enabled;
  uint8_t identity[32], action_digest[32], schema_digest[32], objective_digest[32];
  uint32_t action_count, min_train, freeze_every, min_pairs, max_trials;
  uint32_t exploration_ppm, trial_budget;
  double min_gain, max_regression, learning_rate, family_alpha;
  uint64_t seed;
} ImcConfig;

typedef struct {
  uint32_t abi_version, feature_valid;
  uint8_t identity[32], workload_digest[32];
  uint32_t eligible_mask, request_count, max_tokens, phase;
  uint64_t now_ns;
} ImcContext;

typedef struct {
  uint64_t ticket, policy_version, candidate_version;
  uint32_t action, kind, reason, cell;
  double propensity;
} ImcDecision;

typedef struct {
  uint64_t ticket;
  uint32_t actual_action, status, correctness, resource_ok;
  double cost;
  uint64_t completed_ns;
} ImcCompletion;

/* Extra-execution authority only, never serialized or used by live paths.
 * used counts successfully reserved extra executions, including later failures.
 * End/kill discard unspent remaining quota while retaining granted/used. */
typedef struct {
  uint32_t granted, used, remaining, active;
} ImcTrainingBudgetStatus;

typedef struct {
  uint64_t active_version, fallback_version, candidate_version, frozen_after;
  uint64_t decisions, observations, updates, dropped, overridden, abandoned;
  uint64_t promoted, rejected, invalid_trials, last_pair_id;
  uint32_t enabled, killed, fault_mask, candidate_cell, candidate_action;
  uint32_t pairs_completed, min_pairs, trials_started, credits, pending;
  uint32_t last_outcome, next_order;
  /* upper_bound: exact one-sided binomial failure-frequency upper bound,
   * NOT a raw-ratio confidence interval. Requires independent paired groups. */
  /* Protocol 2 mean_ratio includes adverse scores for noisy A/A windows;
   * aa_noise remains the complete raw mean noise in both protocols. */
  double mean_ratio, aa_noise, upper_bound;
  uint32_t active_actions[6];
  uint32_t fallback_actions[6], candidate_actions[6];
  uint64_t working_counts[24];
  double working_means[24];
  uint64_t last_update_ns, model_bytes, state_bytes, runtime_bytes;
} ImcStatus;

void *imc_create(const ImcConfig *config);
/* Optional: explicit immutable protocol; invalid values return NULL.
 * Existing create/restore exclusively select strict protocol 1. */
void *imc_create_with_protocol(const ImcConfig *config, uint32_t protocol);
void imc_free(void *handle);
int32_t imc_decide(void *handle, const ImcContext *context, ImcDecision *output);
/* Inside an active offline grant: deterministic least-observed alternative below
 * min_train (never ahead of the reference), one execution debited; else imc_decide. */
int32_t imc_decide_training(void *handle, const ImcContext *context, ImcDecision *output);
/* Read-only frozen-policy decision: no tickets, credits, RNG or learning. */
int32_t imc_predict(void *handle, const ImcContext *context, ImcDecision *output);
int32_t imc_complete(void *handle, const ImcCompletion *completion);
/* Produces A, AA, B tickets. order=0 means A,AA,B; order=1 means B,AA,A.
 * Only one triplet may be outstanding. IDs must increase; no reuse. */
int32_t imc_begin_pair(void *handle, const ImcContext *context, uint64_t pair_id,
  ImcDecision *a, ImcDecision *aa, ImcDecision *b, uint32_t *order);
/* One successful grant per handle. Even limit: 2..min(32768,2*min_pairs*max_trials).
 * INPUT1 for invalid limit; UNAVAILABLE3 for repeated grant, killed/disabled,
 * or any pending work. Exhaustion sets active=0; no refunds or trial resets. */
int32_t imc_training_begin(void *handle, uint32_t limit);
int32_t imc_training_end(void *handle);
int32_t imc_training_status(void *handle, ImcTrainingBudgetStatus *output);
/* Identical prospective A/AA/B tickets and gates; only this optional entrypoint
 * consumes two offline units after successful reservation. It uses no live credits. */
int32_t imc_begin_training_pair(void *handle, const ImcContext *context, uint64_t pair_id,
  ImcDecision *a, ImcDecision *aa, ImcDecision *b, uint32_t *order);
int32_t imc_status(void *handle, ImcStatus *output);
int32_t imc_fault(void *handle, uint32_t kind, uint32_t action);
/* Checkpoint includes fixed config and all numerical/controller state.
 * Encoding uses explicit little-endian fields, never raw struct padding.
 * Python must verify its canonical SHA256 envelope before imc_restore.
 * Restore accounts for and abandons all outstanding work; a partly executed
 * comparison is invalidated. It is never resumed as independent evidence. */
int32_t imc_checkpoint(void *handle, uint8_t *output, size_t capacity, size_t *needed);
void *imc_restore(const ImcConfig *config, const uint8_t *input, size_t length);
/* Strict checkpoints retain IMCSTATE01; adverse IMCSTATE02; sequential IMCSTATE03; noise IMCSTATE04.
 * A different expected protocol is rejected, never silently migrated. */
void *imc_restore_with_protocol(const ImcConfig *config, uint32_t protocol,
  const uint8_t *input, size_t length);
#endif
