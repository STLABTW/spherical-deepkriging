from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeepKrigingModelConfig:
    input_dim: int
    output_type: str = "continuous"
    hidden_layers: List[int] = field(default_factory=lambda: [100, 100, 100])
    dropout_rate: float = 0.5
    activation: str = "relu"
    optimizer: str = "adam"
    loss: Optional[str] = None
    metrics: Optional[List[str]] = None
    epochs: int = 10
    batch_size: int = 32
    # for EarlyStopping; None = no early stopping
    patience: Optional[int] = None
    verbose: int = 1

    def __post_init__(self):
        if self.output_type == "continuous":
            self.loss = self.loss or "mse"
            self.metrics = self.metrics or ["mse", "mae"]
        elif self.output_type == "discrete":
            self.loss = self.loss or "binary_crossentropy"
            self.metrics = self.metrics or ["accuracy"]


@dataclass
class DeepKrigingDefaultConfig:
    """Config for Chen et al. (2024) DeepKriging, default: 3×100 Dense(ReLU) → Dropout(0.5) → BatchNorm, then output."""

    input_dim: int
    output_type: str = "continuous"
    hidden_units: int = 100
    num_hidden_layers: int = 3
    dropout_rate: float = 0.5
    activation: str = "relu"
    optimizer: str = "adam"
    loss: Optional[str] = None
    metrics: Optional[List[str]] = None
    epochs: int = 10
    batch_size: int = 32
    patience: Optional[int] = None
    verbose: int = 1

    def __post_init__(self):
        if self.output_type == "continuous":
            self.loss = self.loss or "mse"
            self.metrics = self.metrics or ["mse", "mae"]
        elif self.output_type == "discrete":
            self.loss = self.loss or "binary_crossentropy"
            self.metrics = self.metrics or ["accuracy"]
