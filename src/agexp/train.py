import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow


def build_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        mlflow.log_param("epochs", 10)

        # Dummy data — replace with your real data loader
        X = tf.random.normal((1000, 10))
        y = tf.reduce_sum(X, axis=1)  # Fake regression target

        model = build_model(input_shape=(X.shape[1],))
        model.fit(X, y, epochs=10, validation_split=0.1)

        model_path = "models/model.keras"
        model.save(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
