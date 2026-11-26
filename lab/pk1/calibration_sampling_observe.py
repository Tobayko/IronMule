"""Run one registered blind full original worker after a sampling plan has been selected."""
from lab.pk1.calibration_stability_observe import CONTRACT, capture

if __name__ == "__main__":
    capture(CONTRACT.with_name("calibration_sampling_contract.json"), "ironmule.calibration-sampling-forecast.v1")
