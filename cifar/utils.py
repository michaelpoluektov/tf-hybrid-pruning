import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


def apply_mask(out, mask, weights):
    return out * tf.cast(tf.logical_not(mask), tf.float32) + weights * tf.cast(
        mask, tf.float32
    )


def call_wrapper(func, mask, weights):
    def wrapper(*args, **kwargs):
        return apply_mask(func(*args, **kwargs), mask, weights)

    return wrapper


def set_const(layer, mask, weights):
    if not "partial" in layer.name:
        layer.default_call = layer.call
        layer._name = f"partial_{layer.name}"
    layer._mask = mask
    layer.call = call_wrapper(layer.default_call, layer._mask, weights)


def get_conv_idx_out(model):
    return [
        i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.Conv2D)
    ]


def get_conv_idx_in(model):
    layers = []
    for l in enumerate(model.layers):
        outs = [n.outbound_layer for n in layer._outbound_nodes]
        for out in outs:
            if isinstance(out, tf.keras.layers.Conv2D):
                layers.append(model.layers.index(out))
    return layers


def get_activations(model, num_samples, ds):
    activations = []
    for j, (d, t) in enumerate(ds):
        if j == num_samples:
            break
        activations.append(model(d))
    return tf.concat(activations, 0)


def get_weight(layer: tf.keras.layers.Conv2D):
    return np.prod(np.array(layer.kernel)) * np.prod(layer.output_shape[1:-1])


def get_weights(conv_idx, model):
    weight_arr = np.array([get_weight(model.layers[i]) for i in conv_idx])
    return weight_arr / np.sum(weight_arr)


def get_score(accuracy_delta, sparsity, factor):
    return accuracy_delta + sparsity * factor


def prune_all(model, conv_idx, test_ds, val_ds, max_loss=0.2):
    # Assuming model is trained via transfer learning - base_model would be the original headless model
    base_model = model.layers[0]
    losses_per_layer = get_weights(conv_idx, base_model) * max_loss
    for l, i in tqdm(zip(losses_per_layer, conv_idx)):
        m = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[i].output)
        activations = get_activations(model, 50, test_ds)
        means = tf.math.reduce_mean(activations, 0)
        abs_means = tf.math.reduce_mean(tf.math.abs(activations), 0)
