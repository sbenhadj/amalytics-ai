import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.config import TrainConfig
from amalytics_ml.models.training import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TrainConfig JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)
    cfg = TrainConfig(**cfg_dict)

    ckpt_path = train_model(cfg)
    print(f"Training finished. Checkpoint saved at: {ckpt_path}")

if __name__ == "__main__":
    main()
