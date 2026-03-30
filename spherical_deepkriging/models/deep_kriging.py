from typing import Optional

import numpy as np
import tensorflow as tf

from spherical_deepkriging.configs import (
    DeepKrigingDefaultConfig,
    DeepKrigingModelConfig,
)


class DeepKrigingTrainer:
    def __init__(self, config: DeepKrigingModelConfig) -> None:
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Dense(
                self.config.hidden_layers[0],
                activation=None,
                use_bias=False,
                kernel_initializer="he_normal",
                input_shape=(self.config.input_dim,),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(self.config.activation))
        model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

        for units in self.config.hidden_layers[1:]:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    use_bias=False,
                    kernel_initializer="he_normal",
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(self.config.activation))
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

        output_activation = (
            "linear" if self.config.output_type == "continuous" else "sigmoid"
        )
        model.add(tf.keras.layers.Dense(1, activation=output_activation))

        return model

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        valid_features: Optional[np.ndarray] = None,
        valid_labels: Optional[np.ndarray] = None,
        log_dir: Optional[str] = None,
    ) -> tf.keras.callbacks.History:
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)] if log_dir else []

        return self.model.fit(
            train_features,
            train_labels,
            validation_data=(
                (valid_features, valid_labels) if valid_features is not None else None
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            callbacks=callbacks,
        )


class DeepKrigingDefaultTrainer:
    """Chen et al. (2024) DeepKriging Default: 3×100, Dense → BatchNorm → ReLU → Dropout(0.5); no dropout after last hidden."""

    def __init__(self, config: DeepKrigingDefaultConfig) -> None:
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        n = self.config.num_hidden_layers
        units = self.config.hidden_units
        drop = self.config.dropout_rate
        act = self.config.activation

        for i in range(n):
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    use_bias=False,
                    kernel_initializer="he_normal",
                    input_shape=(self.config.input_dim,) if i == 0 else None,
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(act))
            if i < n - 1:
                model.add(tf.keras.layers.Dropout(drop))

        out_act = "linear" if self.config.output_type == "continuous" else "sigmoid"
        model.add(tf.keras.layers.Dense(1, activation=out_act))
        return model

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        valid_features: Optional[np.ndarray] = None,
        valid_labels: Optional[np.ndarray] = None,
        log_dir: Optional[str] = None,
    ) -> tf.keras.callbacks.History:
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)] if log_dir else []
        return self.model.fit(
            train_features,
            train_labels,
            validation_data=(
                (valid_features, valid_labels) if valid_features is not None else None
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            callbacks=callbacks,
        )
