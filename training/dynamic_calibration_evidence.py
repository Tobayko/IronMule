"""Import the separately measured HTTP and ordered serving outcomes for one generated PK2 source."""
import hashlib

from lab.pk6 import calibration_grade as http
from lab.pk6 import ordered_grade as native
from lab.pk6 import startup_search
from training.code_author import read_request, read_response
from training.code_trial import read
from training.experience_archive import record


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def experiences(root,source,digest):
    folder=(root/source).parent
    order=read(folder/"order.json")
    request_id,request=read_request(folder/"author")
    reply=read_response(folder/"author",request_id)
    forecast=read(folder/"forecast.json")
    if request["order"]!=order or hashlib.sha256(startup_search.materialize(folder)).hexdigest()!=native.CANDIDATE:
        raise ValueError("different calibrated source proposal")
    group=order["order_id"]
    target={"hardware":forecast["hardware"],"base_commit":order["base_commit"],"model":forecast["gguf"],
            "quantization":"Q4_K_M","cost_reference":"lab/rp/results/campaign-costs.json"}
    def emit(kind,path,locator,**values):
        return record(kind,str(path.relative_to(root)),sha(path),locator,group,**target,**values)
    yield emit("local_calibration_forecast",folder/"forecast.json","rank",provenance="simulated",
               reason="conditional warmup and measurement cost model applied to two actual startup traces",
               metrics=forecast["candidates"],decision=forecast["selected"],authorship="IronMule selected the source site from observed local phase costs within a developer-defined family.")
    yield emit("local_calibration_author",folder/"author/response.json","generated_method",
               provenance="reported",reason=reply["rationale"],request_sha256=request_id,
               program_sha256=native.CANDIDATE,authorship="Separate hosted Codex author, in response to the generated order.")
    directory=folder/"http"
    rows=[read(directory/f"aa-{i}.json") for i in (0,1)]
    for r in rows:
        http.validated(r,"reference",98600,False)
    try:
        http.pair(*rows,98600,False,same=True)
    except ValueError as error:
        if read(directory/"result.json")["reason"]!=str(error):
            raise ValueError("HTTP outcome differs from returned work") from error
    else:
        raise ValueError("unchanged HTTP control rejection did not reproduce")
    yield emit("local_calibration_http_rejection",directory/"result.json","unchanged_reference",
               reason="the same warm reference changed one response",decision="reject",
               workload={"cold":32,"warm":32,"facts":18},references=[str((directory/f"aa-{i}.json").relative_to(root)) for i in (0,1)],
               uncertainty="The HTTP lane has no qualifying candidate pair.")
    diagnostic=directory/"diagnostic-candidate.json"
    row=read(diagnostic)
    http.validated(row,"candidate",98600,False)
    yield emit("local_calibration_http_diagnostic",diagnostic,"candidate",
               reason="single code screen following the failed HTTP control",metrics=[{"calls":row["pk2_calls"],"cold_s":row["waves"][0]["delivered_s"]}],
               quality={"facts_correct":18},decision="unqualified",uncertainty="No general serving label.")
    directory=folder/"ordered"
    if (directory/"aa-1.json").is_file():
        try:
            verdict=native.grade(directory)
        except ValueError as exc:
            verdict={"status":"rejected","reason":str(exc),"pairs":[]}
        for entry in verdict["pairs"]:
            seed=entry["seed"]
            name=directory/f"candidate-{seed}.json"
            row=read(name)
            yield emit("local_calibration_ordered_pair",name,str(seed),metrics=[entry],quality={"facts_correct":18,"identical_outputs":True},
                       workload={"seed":seed,"factual_first":entry["facts_first"],"warm":32,"cold":32},
                       label_eligible=verdict["status"]=="qualified",label_scope="time to first complete ordered batch on this physical binding",
                       family_fold="held-out-factual-first" if entry["facts_first"] else "training-mixed-first",
                       baseline_source=str((directory/f"reference-{seed}.json").relative_to(root)),program_sha256=row["source_sha256"],
                       reason="paired ordered worker, with target and exact-work validation",
                       uncertainty="Use whole workload-family holdouts. Each pair belongs to this same physical experiment.")
        if (directory/"outcome.json").is_file():
            observed=read(directory/"outcome.json")
            if verdict["status"]!=observed["status"]:
                raise ValueError("different ordered source verdict")
            yield emit("local_calibration_ordered_verdict",directory/"outcome.json","terminal",
                       reason=verdict.get("reason"),decision=verdict["status"],metrics=[verdict],
                       uncertainty="Authority applies only to the qualified physical target and ordered serving entry.")
    deployed=folder/"target-state/target-check.json"
    if deployed.is_file():
        from lab.pk6.ordered_quality import validate
        from lab.pk6.ordered_release import qualification

        control=read(folder/"target-state/control.json")
        actual=qualification(folder/"ordered")
        checked=read(deployed)
        if control["qualification"]!=actual or read(folder/"target-state/selection.json")["target_check_sha256"]!=sha(deployed):
            raise ValueError("different merged source or fresh selection")
        numeric=folder/"numerical/observed.json"
        quality=validate(read(numeric),actual["binding"]["runtime"]["settings"])
        if checked["metrics"]["numerical_quality"]!=quality:
            raise ValueError("different numerical evidence")
        yield emit("local_calibration_numerical_gate",numeric,"stock",quality=quality,
                   reason="On-device stock reference, repeat and selected candidate scored the same 18 questions and 16 likelihood chunks.")
        for p in checked["metrics"]["fresh"]:
            seed=p["seed"]
            rows=[read(folder/f"fresh/fresh-{a}-{seed}.json") for a in ("reference","candidate")]
            if native.price(*rows,seed,p["facts_first"])!=p:
                raise ValueError("different fresh measured delivery")
        yield emit("local_calibration_source_activation",deployed,"fresh_target",metrics=checked["metrics"]["fresh"],
                   quality=quality,decision="activated",program_sha256=native.CANDIDATE,
                   authorship="The target broker admitted the merged Codex source after two fresh measured workload families.",
                   reason="The exact hardware and runtime fingerprint and the two complete target comparisons passed.")
    if (folder/"proofs/recovery.json").is_file():
        from lab.pk6.replay_ordered import replay
        receipt=replay(folder)
        yield emit("local_calibration_live_pilot",folder/"proofs/trained.json","matched_live_training",metrics=[receipt],
                   decision="candidate", reason="The trained CPU head selected the generated source. Its matched random initialization selected the reference.",
                   uncertainty="The always-qualified simple rule tied the trained Pilot on the two independent factual-first holdouts.")
        yield emit("local_calibration_rollback",folder/"proofs/recovery.json","real_rollback",
                   decision="reference", reason="A persisted withdrawal restored the previous worker on an identical new batch.",
                   quality={"identical_answers":True})
