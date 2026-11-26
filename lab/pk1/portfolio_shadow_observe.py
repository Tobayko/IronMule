"""Bind two fresh complete original workers to the prospective portfolio study."""

from pathlib import Path

from lab.pk1 import post_restart_observe as base
from training.code_author import write_once
from training.code_trial import read

CONTRACT = base.CONTRACT.with_name("portfolio_shadow_contract.json")
WORKER = Path(__file__).with_name("portfolio_shadow_worker.py")
PROBE = WORKER.with_name("calibration_stability_probe.py")


def main():
    p = base.argument_parser()
    p.add_argument("--forecast", type=Path, required=True)
    a = p.parse_args()
    protocol = read(CONTRACT)
    forecast = read(a.forecast)
    if (
        a.out.exists()
        or forecast["schema"] != "ironmule.portfolio-shadow-forecast.v1"
        or forecast["contract_sha256"] != base.native.checksum(CONTRACT)
        or a.seed not in protocol["seeds"]
    ):
        raise ValueError("different forecast, occupied output or unregistered seed")
    context = protocol["contexts"][str(a.seed)]
    row = base.observe(
        a,
        contract_path=CONTRACT,
        worker=WORKER,
        extra_mounts=[(PROBE, "/kernel/calibration_stability_probe.py")],
        extra_env={"IRONMULE_PORTFOLIO_OBSERVATION": context},
    )
    if row["portfolio_context"] != context or row[
        "original_probe_sha256"
    ] != base.native.checksum(PROBE):
        raise ValueError("different original observer")
    write_once(a.out, {**row, "forecast_sha256": base.native.checksum(a.forecast)})


if __name__ == "__main__":
    main()
