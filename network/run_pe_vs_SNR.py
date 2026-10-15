from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pe import main


if __name__ == "__main__":
    main(default_methods=("GI-HT", "HT"), default_stage="train-evaluate", fixed_sweep="SNR")
