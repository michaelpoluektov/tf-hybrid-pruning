# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Summary of modifications:
# Modified the 3 by 3 Conv2D to support sparse + tucker decompositions. See
# diagram in README.md. ResNet50 now takes the decomposition ranks as an
# argument.

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.utils import get_source_inputs


layers = None


def ResNet(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = tf.keras.layers
    if kwargs:
        raise ValueError(f"Unknown argument(s): {kwargs}")
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    # Determine proper input shape
    input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)

    if not preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(
            x
        )
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="post_bn")(x)
        x = layers.Activation("relu", name="post_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return x


# New function not present in original code
def tucker_conv2d(x, filters, rank, spar, kernel_size=3, name=None):
    if rank == 0:
        return layers.Conv2D(filters, kernel_size, padding="SAME", name=name)(x)
    t = layers.Conv2D(
        rank,
        1,
        padding="SAME",
        name=name + "_first",
        use_bias=False,
        kernel_initializer="zeros",
    )(x)
    t = layers.Conv2D(
        rank,
        kernel_size,
        padding="SAME",
        name=name + "_core",
        use_bias=False,
        kernel_initializer="zeros",
    )(t)
    t = layers.Conv2D(
        filters,
        1,
        padding="SAME",
        name=name + "_last",
        kernel_initializer="zeros",
        use_bias=False,
    )(t)
    if spar:
        s = layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_sp")(x)
        x = layers.Add()([s, t])
        return x
    else:
        return t


def block1(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    name=None,
    rank=1,
    spar=0,
):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            x
        )
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = tucker_conv2d(x, filters, rank, spar, kernel_size, name=name + "_2_conv")
    # x = layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, ranks, spars, stride1=2, name=None):
    x = block1(
        x, filters, stride=stride1, name=name + "_block1", rank=ranks[0], spar=spars[0]
    )
    for i in range(2, blocks + 1):
        x = block1(
            x,
            filters,
            conv_shortcut=False,
            name=name + "_block" + str(i),
            rank=ranks[i - 1],
            spar=spars[i - 1],
        )
    return x


def ResNet50(
    ranks,
    spars=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs,
):
    if not spars:
        spars = [0 for _ in ranks]

    def stack_fn(x):
        x = stack1(x, 64, 3, ranks[0:3], spars[0:3], stride1=1, name="conv2")
        x = stack1(x, 128, 4, ranks[3:7], spars[3:7], name="conv3")
        x = stack1(x, 256, 6, ranks[7:13], spars[7:13], name="conv4")
        return stack1(x, 512, 3, ranks[13:16], spars[13:16], name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
