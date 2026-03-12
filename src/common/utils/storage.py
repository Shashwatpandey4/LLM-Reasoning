import os
import json
from datetime import datetime
from typing import Dict, Any, List

def setup_results_dirs(base_dir: str = "results") -> Dict[str, str]:
    """Ensures the directory structure exists and returns paths."""
    dirs = {
        "raw": os.path.join(base_dir, "raw"),
        "summaries": os.path.join(base_dir, "summaries"),
        "plots": os.path.join(base_dir, "plots")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

class ExperimentTracker:
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        self.dirs = setup_results_dirs(base_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        
        self.raw_filepath = os.path.join(self.dirs["raw"], f"{self.run_id}.jsonl")
        self.summary_filepath = os.path.join(self.dirs["summaries"], f"{self.run_id}_metrics.json")
    
    def log_instance(self, data: Dict[str, Any]):
        """Appends a single structured output to a JSONL file."""
        with open(self.raw_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
            
    def save_summary(self, summary_data: Dict[str, Any]):
        """Saves final aggregate metrics to a JSON file."""
        with open(self.summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary saved to {self.summary_filepath}")

    def save_plot_manifest(self, manifest: Dict[str, Any]):
        plot_manifest_path = os.path.join(self.dirs["plots"], f"{self.run_id}_plots.json")
        with open(plot_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        return plot_manifest_path
