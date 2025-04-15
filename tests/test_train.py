import tensorflow as tf
from agexp.train import build_model


def test_build_model():
    model = build_model(input_shape=(10,))
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 10)
