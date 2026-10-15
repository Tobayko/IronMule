#ifndef IRONMULE_SERVING_LEARNER_H
#define IRONMULE_SERVING_LEARNER_H
#include <stddef.h>
#include <stdint.h>

/* ABI 3 (backlog RSI3-LIVE; 2 added waiting metadata and the decomposed first-content model, 3 first_known). Action 0 is the caller's reference schedule. The caller owns
 * serialization: calls on one handle never overlap. No strings, Python objects, GPU handles,
 * prompts or paths cross this boundary. Return codes: 0 OK, 1 invalid input, 3 nothing to do,
 * 5 output buffer too small. */
#define ISL_ABI 3
#define ISL_WAITING 8
#define ISL_MAX_ACTIONS 12
#define ISL_STATES 30

typedef struct {
  uint32_t abi_version, action_count;
  uint32_t widths[ISL_MAX_ACTIONS];     /* group width limit per action, 1..32; unused slots 0 */
  double gather_s[ISL_MAX_ACTIONS];     /* gather wait per action, 0..2 s; unused slots 0 */
  uint32_t max_pending, min_support, ring_capacity, horizon;
  double tau_s, alpha_min, epsilon, synthetic_weight;
  double ttft_weight, failure_penalty_s, reward_scale_s, max_rel_std;
  uint64_t seed;
} IslConfig;

/* Metadata known before a decision; never prompt text. age_cum/chars_cum[k-1]: summed age (s) and
 * prompt characters of the k oldest waiting requests; entries past the queue repeat the last. */
typedef struct {
  double queue, arrival_rate, prompt_chars, max_tokens;
  double age_cum[ISL_WAITING], chars_cum[ISL_WAITING];
} IslState;

/* One completed real decision interval. action is the EXECUTED action. */
typedef struct {
  IslState state, next;
  uint32_t action, served, ok, bad;  /* bad: failed, cancelled, timed out or rejected */
  uint32_t first_known;              /* 1: first_exec_s is a measured mean; 0: missing, not zero */
  double duration_s, service_s, ttft_sum_s;
  double first_exec_s; /* mean over successful members of first content - group dispatch */
  double area_s, unix_s;
} IslTransition;

typedef struct {
  uint32_t action, explored, state_bin, supported_mask;
  double q[ISL_MAX_ACTIONS];
  double predicted_service_s, predicted_service_std;
  double predicted_first_s, predicted_first_std;    /* dispatch -> a member's first content */
  double expected_members, known_wait_s;            /* known at the decision */
  double predicted_ttft_sum_s, predicted_ttft_std;  /* known_wait + members x first */
  uint32_t supported;                               /* 0: below min_support, not usable */
} IslDecision;

typedef struct {
  uint32_t rollouts, steps, stopped_unsupported, stopped_uncertain, stopped_idle, stopped_horizon, budget_exhausted;
  uint64_t elapsed_ns;
} IslDreamResult;

typedef struct {
  uint64_t ring_len, real_updates, synthetic_updates, rollouts, rollout_steps, stopped_unsupported, stopped_uncertain;
  uint64_t decisions, explorations, policy_changes, drift_resets, pruned, model_observations;
  double mean_abs_rel_error[2], within_2sigma, mean_reward, last_td_error;
  uint64_t state_bytes;
  uint64_t depth_hist[8];  /* rollouts by synthetic steps taken: 0..6, 7 = seven or more */
} IslStatus;

/* status 0 incomplete, 1 qualified, 2 rejected; reason 0 none, 1 correctness, 2 regression,
 * 3 noisy/unstable reference, 4 drift, 5 no qualified gain, 6 invalid measurement. */
typedef struct { uint32_t status, reason; double mean_ratio, mean_noise, upper_bound; } IslVerdict;

void *isl_create(const IslConfig *config);
void isl_free(void *handle);
/* explore=1 allows the configured epsilon of directed exploration among eligible actions. */
int32_t isl_decide(void *handle, const IslState *state, uint32_t eligible_mask, uint32_t explore, IslDecision *out);
/* Prediction for one action; no side effect. */
int32_t isl_predict(const void *handle, const IslState *state, uint32_t action, IslDecision *out);
int32_t isl_observe(void *handle, const IslTransition *transition);
/* Bounded Dyna rollouts through the learned model; returns 3 without stored experience. */
int32_t isl_dream(void *handle, uint32_t max_rollouts, uint64_t budget_ns, IslDreamResult *out);
/* Frozen candidate: best action with min_support real observations per state, else 0. */
int32_t isl_policy(const void *handle, uint32_t *out /* ISL_STATES */);
uint32_t isl_bin(const void *handle, const IslState *state);
int32_t isl_status(const void *handle, IslStatus *out);
uint32_t isl_prune(void *handle, double before_unix_s);
int32_t isl_checkpoint(const void *handle, uint8_t *output, size_t capacity, size_t *needed);
void *isl_restore(const IslConfig *config, const uint8_t *input, size_t length);
/* The optimizer's independent A/A/B rule (ironmule_optimizer::qualify), protocol 1 or 2. */
int32_t isl_judge(uint32_t protocol, double min_gain, double max_regression, uint32_t pairs, double family_alpha,
  uint32_t max_trials, const double *a, const double *b, const double *aa, const uint8_t *correct, uint32_t count,
  IslVerdict *out);
#endif
