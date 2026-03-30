from types import SimpleNamespace

import numpy as np
import pytest

from spherical_deepkriging.configs import (
    DeepKrigingDefaultConfig,
    DeepKrigingModelConfig,
)

tf = pytest.importorskip("tensorflow")

from spherical_deepkriging.models.deep_kriging import (  # noqa: E402
    DeepKrigingDefaultTrainer,
    DeepKrigingTrainer,
)


def test_deep_kriging_discrete_output_uses_sigmoid():
    trainer = DeepKrigingTrainer(
        DeepKrigingModelConfig(
            input_dim=3,
            hidden_layers=[4, 2],
            output_type="discrete",
            dropout_rate=0.1,
        )
    )

    assert trainer.model.layers[-1].activation.__name__ == "sigmoid"


def test_deep_kriging_train_passes_validation_and_callbacks(monkeypatch, tmp_path):
    trainer = DeepKrigingTrainer(
        DeepKrigingModelConfig(
            input_dim=2,
            hidden_layers=[4],
            output_type="continuous",
            epochs=2,
            batch_size=4,
            verbose=0,
        )
    )

    compile_kwargs = {}
    fit_kwargs = {}

    def fake_compile(**kwargs):
        compile_kwargs.update(kwargs)

    def fake_fit(*args, **kwargs):
        fit_kwargs.update(kwargs)
        return SimpleNamespace(history={"loss": [1.0]})

    monkeypatch.setattr(trainer.model, "compile", fake_compile)
    monkeypatch.setattr(trainer.model, "fit", fake_fit)

    train_x = np.ones((6, 2), dtype=np.float32)
    train_y = np.ones((6, 1), dtype=np.float32)
    valid_x = np.zeros((3, 2), dtype=np.float32)
    valid_y = np.zeros((3, 1), dtype=np.float32)

    history = trainer.train(
        train_x,
        train_y,
        valid_features=valid_x,
        valid_labels=valid_y,
        log_dir=str(tmp_path),
    )

    assert history.history == {"loss": [1.0]}
    assert compile_kwargs == {
        "optimizer": trainer.config.optimizer,
        "loss": trainer.config.loss,
        "metrics": trainer.config.metrics,
    }
    assert fit_kwargs["validation_data"] == (valid_x, valid_y)
    assert fit_kwargs["epochs"] == 2
    assert fit_kwargs["batch_size"] == 4
    assert len(fit_kwargs["callbacks"]) == 1
    assert isinstance(fit_kwargs["callbacks"][0], tf.keras.callbacks.TensorBoard)


def test_default_trainer_discrete_output_and_optional_arguments(monkeypatch):
    trainer = DeepKrigingDefaultTrainer(
        DeepKrigingDefaultConfig(
            input_dim=2,
            hidden_units=4,
            num_hidden_layers=2,
            output_type="discrete",
            epochs=1,
            batch_size=2,
            verbose=0,
        )
    )

    fit_kwargs = {}
    monkeypatch.setattr(trainer.model, "compile", lambda **kwargs: None)
    monkeypatch.setattr(
        trainer.model,
        "fit",
        lambda *args, **kwargs: fit_kwargs.update(kwargs)
        or SimpleNamespace(history={}),
    )

    train_x = np.zeros((2, 2), dtype=np.float32)
    train_y = np.zeros((2, 1), dtype=np.float32)
    trainer.train(train_x, train_y)

    assert trainer.model.layers[-1].activation.__name__ == "sigmoid"
    assert fit_kwargs["validation_data"] is None
    assert fit_kwargs["callbacks"] == []
