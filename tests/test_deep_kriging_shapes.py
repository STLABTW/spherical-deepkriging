import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def test_deep_kriging_single_output_shape():
    from spherical_deepkriging.configs import DeepKrigingModelConfig
    from spherical_deepkriging.models.deep_kriging import DeepKrigingTrainer

    cfg = DeepKrigingModelConfig(
        input_dim=4,
        hidden_layers=[8, 8],
        output_type="continuous",
        dropout_rate=0.2,
    )

    trainer = DeepKrigingTrainer(cfg)
    x = np.random.randn(5, cfg.input_dim).astype(np.float32)
    y = trainer.model(x, training=False).numpy()

    assert y.shape == (5, 1)


def test_deep_kriging_default_dropout_count():
    from spherical_deepkriging.configs import DeepKrigingDefaultConfig
    from spherical_deepkriging.models.deep_kriging import DeepKrigingDefaultTrainer

    n = 4
    cfg = DeepKrigingDefaultConfig(
        input_dim=4,
        num_hidden_layers=n,
        hidden_units=8,
        dropout_rate=0.2,
        output_type="continuous",
    )
    trainer = DeepKrigingDefaultTrainer(cfg)

    dropout_layers = [
        l for l in trainer.model.layers if isinstance(l, tf.keras.layers.Dropout)
    ]
    # In deep_kriging.py: dropout is added only for i < n-1
    assert len(dropout_layers) == n - 1
