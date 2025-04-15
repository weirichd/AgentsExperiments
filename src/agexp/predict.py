import os
import tensorflow as tf


def load_model(model_path: str | None = None):
    """
    Loads a trained Keras model from disk.

    Args:
        model_path (str): Path to the model file.

    Returns:
        tf.keras.Model: Loaded model.
    """
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models/model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return tf.keras.models.load_model(model_path)


def predict(model: tf.keras.Model, input_data):
    """
    Runs prediction on input data.

    Args:
        model (tf.keras.Model): Loaded model.
        input_data: Array-like input data.

    Returns:
        Model predictions.
    """
    return model.predict(input_data)
