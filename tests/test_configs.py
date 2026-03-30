import pytest


def test_deepkriging_model_config_continuous_defaults():
    from spherical_deepkriging.configs import DeepKrigingModelConfig

    cfg = DeepKrigingModelConfig(input_dim=4, output_type="continuous")
    assert cfg.loss == "mse"
    assert cfg.metrics == ["mse", "mae"]


def test_deepkriging_model_config_discrete_defaults():
    from spherical_deepkriging.configs import DeepKrigingModelConfig

    cfg = DeepKrigingModelConfig(input_dim=4, output_type="discrete")
    assert cfg.loss == "binary_crossentropy"
    assert cfg.metrics == ["accuracy"]


def test_deepkriging_default_config_continuous_defaults():
    from spherical_deepkriging.configs import DeepKrigingDefaultConfig

    cfg = DeepKrigingDefaultConfig(input_dim=4, output_type="continuous")
    assert cfg.loss == "mse"
    assert cfg.metrics == ["mse", "mae"]


def test_deepkriging_default_config_discrete_defaults():
    from spherical_deepkriging.configs import DeepKrigingDefaultConfig

    cfg = DeepKrigingDefaultConfig(input_dim=4, output_type="discrete")
    assert cfg.loss == "binary_crossentropy"
    assert cfg.metrics == ["accuracy"]

