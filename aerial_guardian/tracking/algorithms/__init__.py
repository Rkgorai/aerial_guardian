import os
from pathlib import Path
import yaml

from aerial_guardian.tracking.algorithms.byte_tracker import BYTETracker
from aerial_guardian.tracking.algorithms.bot_sort import BOTSORT

# Registry of available tracking algorithms
TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
}


def create_tracker(config_path, device="cpu"):
    """Factory function to load tracking configurations and instantiate tracking algorithms.
    
    config_path: Path to tracker config (.yaml) or a standard name like 'bytetrack.yaml' or 'botsort.yaml'
    device: device to run ReID networks on ('cpu', 'cuda')
    
    Returns: Tracker instance.
    """
    path = Path(config_path)
    
    # If the user supplied a standard shortname (e.g., 'bytetrack.yaml'), look up in our cfg package folder
    if not path.exists():
        default_cfg_dir = Path(__file__).parent / "cfg"
        alternate_path = default_cfg_dir / path.name
        if alternate_path.exists():
            path = alternate_path
        else:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}' or in default cfg directory '{default_cfg_dir}'.")

    # Read YAML
    print(f"Loading tracker configuration: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid tracker YAML configuration content: expected dict, got {type(cfg)}")

    # Add runtime variables to config
    cfg["device"] = device

    # Determine tracker type
    tracker_type = cfg.get("tracker_type", "bytetrack").lower()
    if tracker_type not in TRACKER_MAP:
        raise ValueError(f"Unknown tracker type '{tracker_type}' specified in config. Available types: {list(TRACKER_MAP.keys())}")

    # Instantiate
    tracker_class = TRACKER_MAP[tracker_type]
    print(f"Instantiating tracker: {tracker_class.__name__}")
    return tracker_class(cfg)
