import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.mixed_precision import global_policy


@register_keras_serializable()
class SparseConv2D(Conv2D):
    def __init__(self, in_mask=1, out_mask=1, in_threshold=0, **kwargs):
        super().__init__(**kwargs)
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_threshold = in_threshold

    def call(self, inputs):
        masked_inputs = (
            tf.dtypes.cast(
                tf.math.abs(inputs) > self.in_threshold,
                global_policy().compute_dtype,
            )
            * inputs
            * self.in_mask
        )
        convolved = super().call(masked_inputs)
        masked_outputs = convolved * self.out_mask
        return masked_outputs

    def get_config(self):
        config = super().get_config()
        if hasattr(self.in_mask, "numpy"):
            config["in_mask"] = self.in_mask.numpy()
        else:
            config["in_mask"] = self.in_mask
        if hasattr(self.out_mask, "numpy"):
            config["out_mask"] = self.out_mask.numpy()
        else:
            config["out_mask"] = self.out_mask
        config["in_threshold"] = self.in_threshold
        return config

    @classmethod
    def from_config(cls, config):
        if "in_mask" in config:
            in_mask = tf.constant(config["in_mask"], global_policy().compute_dtype)
            config.pop("in_mask")
        else:
            in_mask = 1
        if "out_mask" in config:
            out_mask = tf.constant(config["out_mask"], global_policy().compute_dtype)
            config.pop("out_mask")
        else:
            out_mask = 1
        if "in_threshold" in config:
            in_threshold = config["in_threshold"]
            config.pop("in_threshold")
        else:
            in_threshold = 0
        layer = cls(
            in_mask=in_mask, out_mask=out_mask, in_threshold=in_threshold, **config
        )
        return layer


def clone_function(layer):
    if isinstance(layer, Conv2D):
        return SparseConv2D.from_config(layer.get_config())
    else:
        return layer.__class__.from_config(layer.get_config())
