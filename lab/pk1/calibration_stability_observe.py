"""Capture a single prospectively predicted holdout from the live selected A100 startup release."""
from pathlib import Path

from lab.pk1 import post_restart_observe as broker
from training.code_author import write_once
from training.code_trial import read

CONTRACT = broker.CONTRACT.with_name("calibration_stability_contract.json")
WORKER = Path(__file__).with_name("calibration_stability_probe.py")


def main():
    p = broker.argument_parser()
    p.add_argument("--prediction", type=Path, required=True)
    args = p.parse_args()
    forecast = read(args.prediction)
    if (args.out.exists() or forecast.get("schema") != "ironmule.calibration-stability-forecast.v1"
            or forecast["control_receipt_sha256"] != read(CONTRACT)["control_receipt_sha256"]
            or forecast["contract_sha256"] != broker.native.checksum(CONTRACT)
            or not forecast["select_information_probe"] or forecast["holdout_seeds"] != read(CONTRACT)["seeds"]):
        raise ValueError("missing bound prospective prediction or occupied seed")
    # Store the digest before running, with the independently verified publication and full worker binding.
    forecast_hash = broker.native.checksum(args.prediction)
    observation = broker.observe(args, contract_path=CONTRACT, worker=WORKER)
    if broker.native.checksum(args.prediction) != forecast_hash:
        raise ValueError("prediction changed while running the holdout")
    write_once(args.out, {**observation, "forecast_sha256": forecast_hash})


if __name__ == "__main__":
    main()
