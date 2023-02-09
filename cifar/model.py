import tensorflow as tf


def get_model(size):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(size, size, 3),
        include_preprocessing=True,
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(100, activation="softmax")(x)
    return base_model, tf.keras.Model(inputs=inputs, outputs=outputs)
