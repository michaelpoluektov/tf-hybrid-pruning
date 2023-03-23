import tensorflow as tf
from sparse_conv2d import SparseConv2D, clone_function

def get_resnet(size):

    resnet_model = tf.keras.applications.ResNet50(
        include_top = False,
        weights = 'imagenet',
        input_shape = (224,224,3)
    )
    model=tf.keras.models.Sequential()
    model.add(UpSampling2D(size=(7, 7),interpolation='bilinear'))
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.25))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='softmax'))
    return resnet_model, model
