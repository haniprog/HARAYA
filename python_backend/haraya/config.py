from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarayaConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    num_labels: int = 3
    num_behavioral_features: int = 3
    label_names: tuple[str, str, str] = (
        "Safe Interaction",
        "Potential Harassment",
        "Harassment",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def default_model_dir(self) -> Path:
        return self.project_root / "python_backend" / "artifacts"


CONFIG = HarayaConfig()
