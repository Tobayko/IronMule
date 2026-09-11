"""One decision layer over the paths that are already qualified.

`Runtime` takes a service mode from its caller and never changes it. That is right for
a library and wrong for a local user, who wants to load a model and get the fastest
route that this machine has actually earned. `AppleRuntime` is that user's entry point:
it wraps a `Runtime`, asks `ExecutionRouter` once per dispatch which of the established
paths applies, sets the mode, and records what it chose and why.

It is a router, not a second runtime. Every path it can name already exists and already
carries its own admission, correctness contract and fallback:

| route | mechanism | evidence |
| :--- | :--- | :--- |
| `interactive` | `InteractiveMode`, sequential batch-1 | `E15`, `E16` |
| `throughput` | `ThroughputMode`, grouped batch-1 | `E15`, `E16` |
| `paired_throughput` | `PairedThroughputMode` via `AutomaticMode` | `B45`-`B47`, `B50`, `B52` |
| tuned knobs | `Knobs` from the confirmed profile, incl. `k3840_matvec` | `B44`, autotuner |
| `reusable_session` | `ReusableSessionPlan` prefix reuse | `E9`, `E12` |

Four rules keep it honest.

**Only dispatch-time facts.** How many requests were handed over, how long the prompts
are, how many tokens were asked for, whether a session plan matches, the identity of
machine, model and libraries, and the caller's objective. Nothing about the answer a
request has not produced yet.

**The latency/throughput trade is named, not guessed.** `B55` measured it on this
machine: at two ready requests, grouping bought `+7.7%` aggregate tokens per second and
cost `+74.9%` median per-request latency; at four, `+24.4%` throughput for `+6.7%` median
latency. No dispatch-time fact says which of those a caller wants, so `objective` is a
parameter and `throughput` is its default — a caller who handed over several requests
together usually wants them all finished. `objective="latency"` keeps every dispatch on
the sequential path, and the decision record always says which one applied.

**A single request never waits for a partner.** One request is dispatched on the
sequential path immediately; pairing is only ever considered for requests the caller
handed over together.

**An unknown fingerprint falls back to the reference.** `load_profile` fails closed on a
foreign machine, model revision or library build, and a missing profile means the
sequential path plus baseline knobs — the closest thing here to stock mlx-lm. No route is
enabled from a model *name*.

**The router never switches an execution plan.** `E9` measured plans producing tokens up
to 4.31 logits apart, so a plan is a caller decision with visible output consequences.
The router reports the plan a request carries; it does not invent one. `session_plan()`
stays the explicit way to ask for prefix reuse.

## Mixed objectives (`B56`)

`objective` also travels on a `Request`, so one dispatch can carry both. The rule that
splits such a dispatch is fixed here, before it was measured, and it is deterministic:

1. Resolve each request's objective: the request's own, else the one given to `serve`,
   else the runtime's. An unspecified request therefore keeps doing exactly what it did
   before this existed.
2. Partition into one cohort per objective, caller order preserved inside each.
3. Order the cohorts by `(earliest arrival_ms in the cohort, latency before throughput)`.
4. Serve each cohort in that order with the route the router picks for it, sharing one
   dispatch timestamp so the wait between cohorts stays in the record.
5. Return results in the caller's original order.

Latency wins the tie because the harm is asymmetric. A latency request pulled into a
group pays the whole grouping cost (`B55`: `+74.9%` median latency at two requests); a
throughput request that waits for the latency cohort loses time it did not ask to
protect. Ordering by arrival first keeps that from becoming priority inversion: a cohort
whose earliest request has not arrived yet never leads.

**The bound this rule accepts, stated rather than hidden, and measured.** Cohorts run to
completion in order, so a trailing cohort's first token cannot precede the leading
cohort's last. It is bounded by the dispatch, every request is served, and no cohort
waits on a condition that may never occur -- so no starvation and no unbounded wait.

What that costs is `B56`'s one failed criterion. A `latency` request declared to arrive
*after* a throughput cohort has started does not lead, and waits for it: `2.10x` its own
latency with nothing foreign around it. A hand-written caller doing the same split
measured `2.10x` too, so this is the device and not the routing. The alternatives were
measured on the same arrival pattern and neither helps: letting the late request join the
group instead of waiting gives `2.05x` and throws away the latency path as well, while
dispatching it separately gives `1.00x` by construction.

So the rule is: **a latency request that becomes servable after work is already committed
cannot be protected inside that dispatch.** `last_decision["latency_protected"]` says
whether a dispatch could protect its latency requests, and a caller who receives a
request later should dispatch it later, which is what a server does anyway.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, dataclass, replace, field
from typing import Mapping, Any, Sequence

from .activation import ActivationContext, LearnedDispatchActivation, OFF
from .local_learner import IntakeContext, LocalLearner, default_state_path
from .monitoring import DriftMonitor, Observation, default_action_code_digest
from .plans import ExecutionPlan, StrictOneShotPlan, plan_kind
from .service import (AutomaticMode, InteractiveMode, PairedThroughputMode, Request,
                      Result, Runtime, ThroughputMode, paired_status)
from .silicon_profile import (RuntimeContext, match_silicon_parameter,
                              workload_class_for)
from .telemetry import Telemetry

ROUTER_VERSION = "ironmule.execution_router.v1"

#: Routes the router may name. `reference` is the unqualified safe path.
ROUTES = ("reference", "interactive", "throughput", "paired_throughput")
#: What a caller is optimising. Neither is free; `B55` measured what each costs.
OBJECTIVES = ("throughput", "latency")
#: The rule that splits a dispatch carrying more than one objective. Fixed before `B56`
#: measured it, and written into every record so a result names the rule it came from.
SCHEDULING_RULE = "ironmule.dispatch.objective_cohorts.v1"


@dataclass(frozen=True)
class RouteDecision:
    """What was chosen, on which facts, and which runs authorise it."""

    route: str
    reason: str
    objective: str
    plan_kinds: tuple[str, ...]
    requests: int
    max_prompt_tokens: int
    max_new_tokens: int
    session_plan_matches: int
    qualified_profile: bool
    request_ids: tuple[str, ...] = ()
    earliest_arrival_ms: float = 0.0
    evidence_run_ids: tuple[str, ...] = ()
    #: `B70`, shadow only. What a loaded silicon profile says about this dispatch, if
    #: anything. `route` and `reason` above are built without ever reading it, so this
    #: field can be wrong without the answer being wrong.
    silicon: Mapping[str, Any] | None = None
    #: The class name this dispatch was given, by `silicon_profile.workload_class_for`.
    #: Empty means the dispatch fits no named class, which matches nothing.
    workload_class: str = ""
    #: `B78`, shadow only. What this machine has learned locally about an action for this
    #: workload class, if anything. Like `silicon` it is filled in `annotate()` after the
    #: route exists, so `route` and `reason` cannot depend on it.
    local_learning: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["router_version"] = ROUTER_VERSION
        record["plan_kinds"] = list(self.plan_kinds)
        record["request_ids"] = list(self.request_ids)
        record["evidence_run_ids"] = list(self.evidence_run_ids)
        record["silicon"] = dict(self.silicon) if self.silicon else None
        record["local_learning"] = dict(self.local_learning) if self.local_learning else None
        return record


def _merge_telemetry(parts: Sequence[Telemetry], mode: str) -> Telemetry:
    """One telemetry for a dispatch that was served as several cohorts.

    Per-request metrics are already stamped against the dispatch, so they merge by
    concatenation. `wall_ns` is summed rather than maxed: the cohorts ran one after the
    other in one process, so their times add up and do not overlap.
    """
    if len(parts) == 1:
        return parts[0]
    merged = Telemetry(mode=mode)
    for part in parts:
        merged.plan_kinds.extend(part.plan_kinds)
        merged.requests.extend(part.requests)
        merged.realised_widths.extend(part.realised_widths)
        merged.wall_ns += part.wall_ns
        merged.fallbacks += part.fallbacks
        merged.fallback_reasons.extend(part.fallback_reasons)
        merged.correctness_errors += part.correctness_errors
        merged.correctness_check_performed |= part.correctness_check_performed
        merged.correctness_checked_requests += part.correctness_checked_requests
        merged.plan_switch_attempts += part.plan_switch_attempts
        merged.peak_memory_bytes = max(merged.peak_memory_bytes, part.peak_memory_bytes)
    return merged


class ExecutionRouter:
    """Pure choice: facts in, one route out. Constructs nothing and measures nothing."""

    def __init__(self, profile: dict[str, Any] | None, *, identity_sha256: str | None = None,
                 fingerprint: str | None = None, mlx: str = "", mlx_lm: str = "",
                 objective: str = "throughput",
                 silicon_profile: Any | None = None,
                 silicon_context: Any | None = None,
                 local_learner: Any | None = None):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.profile = profile
        self.identity_sha256 = identity_sha256
        self.fingerprint = fingerprint
        self.mlx = mlx
        self.mlx_lm = mlx_lm
        # `B70`, shadow only. Both may be None and usually are. Nothing below reads them
        # while choosing a route; they are read after the route exists, to annotate it.
        self.silicon_profile = silicon_profile
        self.silicon_context = silicon_context
        # `B78`, shadow only, and read in the same place for the same reason. May be None
        # and is on a fresh install.
        self.local_learner = local_learner

    @property
    def qualified(self) -> bool:
        """A profile is here at all. `load_profile` already refused a foreign one."""
        return bool(self.profile) and bool(self.identity_sha256) and bool(self.fingerprint)

    def objective_for(self, request: Request, default: str | None = None) -> str:
        """The request's own objective, else the dispatch's, else the runtime's."""
        return request.objective or default or self.objective

    def plan(self, requests: Sequence[Request],
             default_objective: str | None = None) -> tuple[RouteDecision, ...]:
        """The cohorts of one dispatch, in the order they will be served.

        Deterministic by construction: the partition is by resolved objective, order
        inside a cohort is the caller's, and order between cohorts is
        `(earliest arrival, latency before throughput)`. Same input, same plan.
        """
        if default_objective is not None and default_objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, "
                             f"got {default_objective!r}")
        requests = list(requests)
        if not requests:
            return (self.decide(requests, default_objective),)
        cohorts: dict[str, list[Request]] = {}
        for request in requests:
            cohorts.setdefault(self.objective_for(request, default_objective),
                               []).append(request)
        rank = {"latency": 0, "throughput": 1}
        order = sorted(cohorts, key=lambda name: (
            min(r.arrival_ms for r in cohorts[name]), rank[name]))
        return tuple(self.decide(cohorts[name], name) for name in order)

    def annotate(self, decision: RouteDecision) -> RouteDecision:
        """The diagnostic, computed once per dispatch and never inside `decide()`.

        `B70` measured why it lives here. An eager version inside `decide()` cost `78%` of
        the decision, and `63%` after optimising it, against an equivalence gate of `2%`.
        Both failures are recorded. A `4.3 us` decision cannot absorb a diagnostic that
        allocates anything, so the diagnostic moved to where a dispatch is recorded --
        once per dispatch rather than once per decision, and after the route already
        exists. `decide()` is therefore byte for byte what it was before this feature, and
        that is structural rather than a promise: it cannot read what it does not compute.
        """
        name = workload_class_for(requests=decision.requests,
                                  session_plan_matches=decision.session_plan_matches,
                                  max_new_tokens=decision.max_new_tokens,
                                  plan_kinds=decision.plan_kinds)
        # `B78`: one mapping lookup against a dictionary the learner built when its evidence
        # last changed. Nothing is estimated here and nothing is allocated.
        learned = (self.local_learner.shadow_for(name)
                   if self.local_learner is not None else None)
        if self.silicon_profile is None or self.silicon_context is None:
            return replace(decision, workload_class=name, local_learning=learned)
        context = self.silicon_context.for_dispatch(
            name, decision.objective, 1 if decision.route == "interactive" else 0)
        match = match_silicon_parameter(context, self.silicon_profile)
        return replace(decision, workload_class=name, silicon=match.as_dict(),
                       local_learning=learned)

    def decide(self, requests: Sequence[Request],
               objective: str | None = None) -> RouteDecision:
        objective = objective or self.objective
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        plans = tuple(plan_kind(r.plan) for r in requests)
        matches = sum(1 for r in requests
                      if getattr(r.plan, "matches", None) is not None
                      and r.plan.matches(r.prompt_ids))
        facts = {
            "objective": objective,
            "plan_kinds": plans,
            "requests": len(requests),
            "max_prompt_tokens": max((len(r.prompt_ids) for r in requests), default=0),
            "max_new_tokens": max((r.max_tokens for r in requests), default=0),
            "session_plan_matches": matches,
            "qualified_profile": self.qualified,
            "request_ids": tuple(r.rid for r in requests),
            "earliest_arrival_ms": min((r.arrival_ms for r in requests), default=0.0),
        }
        if not self.qualified:
            return RouteDecision(
                "reference",
                "no profile qualified for this machine, model revision and library build",
                **facts)
        if len(requests) <= 1:
            return RouteDecision(
                "interactive",
                "one request: the sequential path answers it now rather than waiting "
                "for a partner (E15, E16: grouping costs 26-31% median latency)",
                evidence_run_ids=("E15", "E16", "B55"), **facts)
        if objective == "latency":
            return RouteDecision(
                "interactive",
                f"{len(requests)} requests, objective 'latency': grouping would buy "
                "aggregate throughput at median per-request latency (B55 measured "
                "+74.9% at two requests, +6.7% at four), which this caller asked not to "
                "pay",
                evidence_run_ids=("E15", "E16", "B55"), **facts)
        return RouteDecision(
            "throughput",
            f"{len(requests)} requests dispatched together, objective 'throughput': "
            "grouped batch-1, and the profile's service-strategy record decides the "
            "paired path per round",
            evidence_run_ids=("E15", "E16", "B55"), **facts)


def monitor_path(state_path: Path) -> Path:
    """The monitor's own file, beside the controller's."""
    return Path(state_path).with_name(Path(state_path).stem + "_monitoring.json")


class AppleRuntime:
    """A loaded model that routes itself. Thin: it owns a `Runtime` and a decision.

    Use it exactly like `Runtime`, minus the mode:

        rt = AppleRuntime.load()
        out = rt.generate("...")                     # one request -> sequential
        results = rt.serve([...])                    # several -> grouped, paired if earned
        print(rt.last_decision)                      # route, reason, evidence

        rt = AppleRuntime.load(objective="latency")  # never trade latency for throughput

    One dispatch may carry both, and each request says which it wants:

        rt.serve([Request(prompt_ids=a, objective="latency"),
                  Request(prompt_ids=b, objective="throughput")])
        print(rt.last_decision["cohort_order"])      # ['latency', 'throughput']
        print(rt.last_decision["route_by_request"])  # per request, not per dispatch
    """

    def __init__(self, runtime: Runtime, router: ExecutionRouter,
                 activation: LearnedDispatchActivation | None = None,
                 monitor: DriftMonitor | None = None,
                 monitor_path: Path | None = None,
                 action_code_digest: str = ""):
        self.runtime = runtime
        self.router = router
        # `B79`. None means the reference, which is what every caller before B79 gets.
        self.activation = activation
        # `B80`, passive and off the hot path: an observation is built once per dispatch,
        # after the answer exists, in the same place the other diagnostics live.
        self.monitor = monitor
        self.monitor_path = monitor_path
        self.action_code_digest = action_code_digest
        self.decisions: list[dict[str, Any]] = []

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, model_id: str | None = None, revision: str | None = None,
             use_tuned_profile: bool = True, objective: str = "throughput",
             enable_local_learned_dispatch: bool = False,
             local_state_path: "Path | None" = None) -> "AppleRuntime":
        """Load once, with the tuned knobs this machine confirmed, and route from then on.

        The knob set — including `k3840_matvec`, which `Engine.admit_k3840` re-checks
        against the loaded weights — is chosen at load exactly as `Runtime.load` chooses
        it. Nothing here re-implements that; the router sits above it.
        """
        import mlx.core as mx
        import mlx_lm

        from .hw import fingerprint as hw_fingerprint
        from .tune import DEFAULT_MODEL, load_profile, resolve_local_model

        model_id = model_id or DEFAULT_MODEL
        resolved = resolve_local_model(model_id, revision)
        profile = (load_profile(model_id, revision=revision,
                                model_identity=resolved.identity)
                   if use_tuned_profile else None)
        runtime = Runtime.load(model_id, revision=revision,
                               use_tuned_profile=use_tuned_profile)
        # `B78`, shadow only. Restored fail-closed: missing, corrupt, or written for
        # another machine, model or library build all leave a learner that knows nothing,
        # and a learner that knows nothing annotates nothing.
        state_path = local_state_path or default_state_path()
        learner = LocalLearner.restore(
            state_path,
            IntakeContext(hardware_fingerprint=hw_fingerprint(),
                          mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                          model_id=model_id,
                          model_identity_sha256=resolved.identity.identity_sha256,
                          model_revision=getattr(resolved.identity, "revision", "") or ""))
        router = ExecutionRouter(profile,
                                 identity_sha256=resolved.identity.identity_sha256,
                                 fingerprint=hw_fingerprint(),
                                 mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                                 objective=objective,
                                 local_learner=learner)
        # `B79`. Off unless this caller asked for it *and* the controller qualifies exactly
        # this context. A profile cannot switch it on and neither can an environment
        # variable; a persisted kill record outranks the flag.
        quantisation = getattr(resolved.identity, "quantisation", None) or {}
        # `B80`, passive. Restored fail-closed like everything else: an unreadable monitor
        # state leaves one that watches from scratch, and a stored requalification survives.
        # The drift rule's material floor is derived from the controller, not chosen: the
        # smallest gain its own qualified interval supports. A slowdown below that leaves the
        # action still winning, so there is nothing to requalify.
        qualified = [learner.recommendation(action, workload_class)
                     for (action, workload_class) in learner._recommendations]
        intervals = [row.prediction_interval[1] for row in qualified
                     if row.prediction_interval is not None
                     and row.local_learning_state == "CANDIDATE_QUALIFIED"]
        material_floor = max(0.0, 1.0 - max(intervals)) if intervals else 0.0
        monitor = DriftMonitor.restore(monitor_path(state_path), state_path, material_floor)
        activation_context = ActivationContext(
                hardware_fingerprint=hw_fingerprint(),
                gpu_architecture=str(mx.device_info().get("architecture", "")),
                model_id=model_id,
                model_identity_sha256=resolved.identity.identity_sha256,
                model_revision=getattr(resolved.identity, "revision", "") or "",
                quantization_bits=int(quantisation.get("bits", 0)),
                quantization_group_size=int(quantisation.get("group_size", 0)),
                mlx=mx.__version__, mlx_lm=mlx_lm.__version__)
        activation = LearnedDispatchActivation(
            learner, activation_context,
            enabled=enable_local_learned_dispatch,
            state_path=state_path, monitor=monitor)
        if activation.enabled:
            activation.install_on(runtime.engine.model, resolved.identity)
        return cls(runtime, router, activation, monitor=monitor,
                   monitor_path=monitor_path(state_path),
                   action_code_digest=default_action_code_digest())

    # -- pass-through ---------------------------------------------------------
    def close(self) -> None:
        if self.monitor is not None and self.monitor_path is not None:
            self.monitor.save(self.monitor_path)
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False

    def encode(self, text: str) -> list[int]:
        return self.runtime.encode(text)

    def session_plan(self, shared_prefix: str, name: str = "session"):
        """Prefix reuse stays an explicit request, because it changes the output."""
        return self.runtime.session_plan(shared_prefix, name)

    @property
    def telemetry(self):
        return self.runtime.telemetry

    @property
    def last_decision(self) -> dict[str, Any] | None:
        return self.decisions[-1] if self.decisions else None

    # -- serving --------------------------------------------------------------
    def _mode_for(self, decision: RouteDecision):
        if decision.route == "throughput":
            # `AutomaticMode` is the shipped throughput path plus the profile's own
            # paired admission; it refuses back to throughput on its own if the model
            # declines. Re-deriving that split here would be a second mechanism.
            return AutomaticMode(self.router.profile, opt_in=True,
                                 identity_sha256=self.router.identity_sha256,
                                 fingerprint=self.router.fingerprint,
                                 mlx=self.router.mlx, mlx_lm=self.router.mlx_lm)
        return InteractiveMode()

    def _serve_cohort(self, decision: RouteDecision, requests: list[Request],
                      dispatch_ns: int) -> tuple[list[Result], dict[str, Any]]:
        """One cohort on one route. `dispatch_ns` keeps the wait before it in the record."""
        mode = self._mode_for(decision)
        self.runtime.mode = mode
        # `B79`, before a token is produced: the activation layer may substitute the
        # locally qualified candidate for the reference, and only for a workload class the
        # controller itself qualified. The route above is already chosen and is untouched.
        workload_class = workload_class_for(
            requests=decision.requests,
            session_plan_matches=decision.session_plan_matches,
            max_new_tokens=decision.max_new_tokens, plan_kinds=decision.plan_kinds)
        activation = (self.activation.for_dispatch(workload_class, decision.route)
                      if self.activation is not None else OFF)
        started = time.perf_counter_ns()
        try:
            results = self.runtime.serve(requests, dispatch_ns=dispatch_ns)
        except BaseException as error:
            # One direction out: a candidate dispatch that raised takes the candidate away
            # for every later request in this process, and the exception still propagates.
            if self.activation is not None and activation.effective_action == "candidate":
                self.activation.kill(f"an exception in the candidate path: "
                                     f"{type(error).__name__}")
            raise
        telemetry = self.runtime.telemetry
        if self.activation is not None:
            self.activation.after_dispatch(activation, telemetry)
        record = decision.as_dict()
        record["activation"] = activation.as_dict()
        record["mode"] = getattr(mode, "name", "unknown")
        record["mode_status"] = paired_status(mode)
        record["realised_strategy"] = record["mode_status"].get("strategy", record["mode"])
        record["cohort_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
        record["telemetry"] = telemetry.snapshot()
        return results, record

    def serve(self, requests: Sequence[Request],
              objective: str | None = None) -> list[Result]:
        """Serve one dispatch, splitting it by objective when it carries more than one.

        `objective` fills in for requests that named none; a request that named one keeps
        it. With a single objective in play this is exactly one `Runtime.serve`, which is
        what every call written before B56 does.
        """
        from .hw import swap_used_bytes

        requests = list(requests)
        # Cohorts address their members by request id, so two requests may not share one.
        # A caller sets `rid` by hand or lets it be generated; only the first can collide.
        by_id = {request.rid: request for request in requests}
        if len(by_id) != len(requests):
            raise ValueError("a dispatch cannot carry two requests with the same rid")
        cohorts = self.router.plan(requests, objective)
        swap_before = swap_used_bytes()
        dispatch_ns = time.perf_counter_ns()
        by_rid: dict[str, Result] = {}
        cohort_records: list[dict[str, Any]] = []
        telemetries = []
        try:
            for decision in cohorts:
                members = [by_id[rid] for rid in decision.request_ids]
                if not members:
                    continue
                results, record = self._serve_cohort(decision, members, dispatch_ns)
                cohort_records.append(record)
                telemetries.append(self.runtime.telemetry)
                by_rid.update({result.rid: result for result in results})
        finally:
            record = self._dispatch_record(cohorts, cohort_records, requests,
                                           dispatch_ns, swap_before)
            self.decisions.append(record)
        if telemetries:
            self.runtime.telemetry = _merge_telemetry(telemetries, record["mode"])
            self.runtime.telemetry.routing = dict(record)
            record["telemetry"] = self.runtime.telemetry.snapshot()
            record["fallbacks"] = self.runtime.telemetry.fallbacks
            record["fallback_reasons"] = list(self.runtime.telemetry.fallback_reasons[:10])
        return [by_rid[request.rid] for request in requests]

    def _dispatch_record(self, cohorts, cohort_records, requests, dispatch_ns,
                         swap_before) -> dict[str, Any]:
        """One record per dispatch, whatever it was split into."""
        from .hw import swap_used_bytes

        routes = {rid: decision.route
                  for decision in cohorts for rid in decision.request_ids}
        objectives = {rid: decision.objective
                      for decision in cohorts for rid in decision.request_ids}
        leading = cohort_records[0] if cohort_records else {}
        order = [decision.objective for decision in cohorts]
        # A latency request is protected when nothing foreign runs before it: either no
        # throughput cohort exists, or the latency cohort leads. B56 measured what the
        # other case costs (2.10x) and that no in-dispatch schedule avoids it.
        protected = ("latency" not in order or "throughput" not in order
                     or order.index("latency") < order.index("throughput"))
        # `B70`, shadow only, once per dispatch. The route is already chosen and every
        # cohort already served when this runs, so it annotates history.
        annotated = [self.router.annotate(decision) for decision in cohorts]
        monitoring = self._observe(annotated, cohort_records)
        return {
            "router_version": ROUTER_VERSION,
            "silicon": [{"request_ids": list(decision.request_ids),
                         "workload_class": decision.workload_class,
                         "match": decision.silicon} for decision in annotated],
            "local_learning": [{"request_ids": list(decision.request_ids),
                                "workload_class": decision.workload_class,
                                "shadow": decision.local_learning}
                               for decision in annotated],
            "activation": [row.get("activation") for row in cohort_records],
            "monitoring": monitoring,
            "dispatch_wall_ms": (time.perf_counter_ns() - dispatch_ns) / 1e6,
            "requests": len(requests),
            "mixed_objectives": len(cohorts) > 1,
            "cohort_order": order,
            "latency_protected": protected,
            "scheduling_rule": SCHEDULING_RULE,
            "route_by_request": routes,
            "objective_by_request": objectives,
            "cohorts": cohort_records,
            "swap_used_bytes_before": swap_before,
            "swap_used_bytes_after": swap_used_bytes(),
            # The leading cohort's own fields stay at the top level so a single-objective
            # dispatch reads exactly as it did before B56.
            "route": leading.get("route", "reference"),
            "reason": leading.get("reason", ""),
            "objective": leading.get("objective", self.router.objective),
            "mode": leading.get("mode", "unknown"),
            "realised_strategy": leading.get("realised_strategy", "unknown"),
            "plan_kinds": leading.get("plan_kinds", []),
            "session_plan_matches": sum(r.get("session_plan_matches", 0)
                                        for r in cohort_records),
            "qualified_profile": self.router.qualified,
            "evidence_run_ids": leading.get("evidence_run_ids", []),
            "fallbacks": 0,
            "fallback_reasons": [],
        }

    def _observe(self, annotated, cohort_records) -> list[dict[str, Any]]:
        """`B80`. One observation per cohort, built after the answer exists.

        This is observational evidence: what one real dispatch did, with no counterfactual.
        It can raise doubt about a qualified action and can never qualify one.
        """
        if self.monitor is None or not self.activation:
            return []
        from .hw import memory_pressure_level, swap_used_bytes

        context = getattr(self.activation, "context", None)
        if context is None:
            return []
        outcomes = []
        transitions_before = len(self.monitor.transitions)
        for decision, record in zip(annotated, cohort_records):
            activation = record.get("activation") or {}
            telemetry = record.get("telemetry") or {}
            if not decision.workload_class:
                continue
            observation = Observation(
                observed_at=datetime.now(timezone.utc).isoformat(),
                hardware_fingerprint=context.hardware_fingerprint,
                gpu_architecture=context.gpu_architecture,
                model_identity_sha256=context.model_identity_sha256,
                model_revision=context.model_revision,
                quantization_bits=context.quantization_bits,
                quantization_group_size=context.quantization_group_size,
                mlx=context.mlx, mlx_lm=context.mlx_lm,
                workload_class=decision.workload_class,
                action_id=self.activation.action_id,
                action_code_digest=self.action_code_digest or "unknown",
                effective_action=activation.get("effective_action", "reference"),
                end_to_end_ms=float(record.get("cohort_wall_ms") or 0.0) or 1e-6,
                service_ttft_ms=telemetry.get("service_ttft_p50_ms"),
                tokens_per_second=telemetry.get("aggregate_tokens_per_second"),
                new_tokens=int(decision.max_new_tokens),
                prompt_tokens=int(decision.max_prompt_tokens),
                fallbacks=int(telemetry.get("fallbacks") or 0),
                correctness_errors=int(telemetry.get("correctness_errors") or 0),
                memory_pressure_level=memory_pressure_level(),
                swap_used_bytes=swap_used_bytes(),
                controller_digest=activation.get("controller_digest") or "")
            outcomes.append(self.monitor.observe(observation))
        if len(self.monitor.transitions) != transitions_before and self.monitor_path:
            self.monitor.save(self.monitor_path)
        return outcomes

    def generate(self, prompt: str | None = None, *, prompt_ids: Sequence[int] | None = None,
                 plan: ExecutionPlan | None = None, max_tokens: int = 64,
                 objective: str | None = None) -> Result:
        ids = list(prompt_ids) if prompt_ids is not None else self.encode(prompt or "")
        request = Request(prompt_ids=ids, max_tokens=max_tokens,
                          plan=plan or StrictOneShotPlan(), objective=objective)
        return self.serve([request])[0]

    def status(self) -> dict[str, Any]:
        """One line's worth of truth about what this runtime will do next."""
        return {
            "router_version": ROUTER_VERSION,
            "objective": self.router.objective,
            "scheduling_rule": SCHEDULING_RULE,
            "qualified_profile": self.router.qualified,
            "local_learning": (self.router.local_learner.as_dict()
                               if self.router.local_learner is not None else None),
            "monitoring": (self.monitor.as_dict() if self.monitor is not None else None),
            "local_learned_dispatch": (self.activation.status()
                                       if self.activation is not None else
                                       {"enabled": False, "default_enabled": False,
                                        "disabled_reason": "no activation layer"}),
            "hardware_fingerprint": self.router.fingerprint,
            "model_identity_sha256": self.router.identity_sha256,
            "mlx": self.router.mlx, "mlx_lm": self.router.mlx_lm,
            "knobs": self.runtime.engine.knobs.as_dict(),
            "dispatches": len(self.decisions),
            "last_decision": self.last_decision,
        }


def _self_check() -> None:
    """The routing table itself, without a model. Every branch, once."""

    def request(n_prompt: int = 8, max_tokens: int = 16, plan=None) -> Request:
        return Request(prompt_ids=list(range(n_prompt)), max_tokens=max_tokens,
                       plan=plan or StrictOneShotPlan())

    unknown = ExecutionRouter(None)
    assert not unknown.qualified
    assert unknown.decide([request()]).route == "reference"
    assert unknown.decide([request(), request()]).route == "reference", \
        "an unknown fingerprint stays on the reference even under concurrency"

    known = ExecutionRouter({"knobs": {}}, identity_sha256="a" * 64,
                            fingerprint="m1max-64", mlx="0.32.0", mlx_lm="0.28.4")
    assert known.qualified
    solo = known.decide([request()])
    assert solo.route == "interactive" and solo.requests == 1
    assert "waiting" in solo.reason, "the single-request rule must say why it does not wait"

    pair = known.decide([request(), request()])
    assert pair.route == "throughput" and pair.requests == 2

    from .plans import ReusableSessionPlan
    session = ReusableSessionPlan([0, 1, 2], name="doc")
    with_session = known.decide([request(plan=session), request(plan=session)])
    assert with_session.session_plan_matches == 2
    assert with_session.plan_kinds == ("reusable_session", "reusable_session")
    assert with_session.route == "throughput", "a plan is reported, never chosen"

    latency = ExecutionRouter({"knobs": {}}, identity_sha256="a" * 64,
                              fingerprint="m1max-64", mlx="0.32.0", mlx_lm="0.28.4",
                              objective="latency")
    assert latency.decide([request(), request()]).route == "interactive", \
        "objective 'latency' must not buy throughput with median latency"
    assert latency.decide([request()]).route == "interactive"
    try:
        ExecutionRouter(None, objective="fastest")
    except ValueError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("an unknown objective must be refused")

    empty = known.decide([])
    assert empty.route == "interactive" and empty.max_prompt_tokens == 0

    # B56: one dispatch, two objectives.
    def tagged(objective, arrival=0.0):
        return Request(prompt_ids=[1, 2, 3], max_tokens=8, plan=StrictOneShotPlan(),
                       objective=objective, arrival_ms=arrival)

    mixed = known.plan([tagged("throughput"), tagged("latency")])
    assert [d.objective for d in mixed] == ["latency", "throughput"], \
        "a latency request must not be pulled into a foreign group"
    assert [d.route for d in mixed] == ["interactive", "interactive"], \
        "a lone throughput request has no partner, so it takes the sequential path too"
    assert sum(d.requests for d in mixed) == 2

    wider = known.plan([tagged("throughput"), tagged("latency"), tagged("throughput")])
    assert [(d.objective, d.route, d.requests) for d in wider] == [
        ("latency", "interactive", 1), ("throughput", "throughput", 2)], \
        "the throughput cohort groups among itself and leaves the latency one alone"

    later_latency = known.plan([tagged("throughput", 0.0), tagged("latency", 50.0)])
    assert [d.objective for d in later_latency] == ["throughput", "latency"], \
        "a cohort whose earliest request has not arrived must not lead"

    single = known.plan([tagged("throughput"), tagged("throughput")])
    assert len(single) == 1 and single[0].route == "throughput", \
        "one objective in play is one cohort, exactly as before B56"

    unspecified = known.plan([request(), request()])
    assert len(unspecified) == 1 and unspecified[0].objective == "throughput", \
        "an unspecified request inherits the runtime objective"
    assert known.plan([request(), request()], "latency")[0].objective == "latency"
    assert known.plan([tagged("throughput"), request()], "latency")[0].objective == \
        "latency", "a request's own objective wins over the dispatch default"
    assert len(known.plan([tagged("throughput"), request()], "latency")) == 2

    first = known.plan([tagged("throughput"), tagged("latency"), tagged("latency")])
    again = known.plan([tagged("throughput"), tagged("latency"), tagged("latency")])
    assert [d.objective for d in first] == [d.objective for d in again], \
        "the same dispatch must produce the same plan"

    unknown_mixed = ExecutionRouter(None).plan([tagged("latency"), tagged("throughput")])
    assert {d.route for d in unknown_mixed} == {"reference"}, \
        "an unqualified fingerprint routes nothing, whatever the objective says"

    record = solo.as_dict()
    assert record["router_version"] == ROUTER_VERSION
    assert record["evidence_run_ids"] == ["E15", "E16", "B55"]
    assert set(ROUTES) >= {d.route for d in (solo, pair, empty, with_session)}
    print("router self-check ok")


if __name__ == "__main__":
    _self_check()
