import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def Add_Convolutional_Layer(x, depth, size, stride = 1, batch_norm=True):
    x = layers.Conv2D(filters = depth, kernel_size = size, strides = (stride, stride), padding = 'same', use_bias = False)(x)

    if batch_norm:
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(0.1)(x)

    return x


def Add_Residual(x, depth, size):
    inp = x
    x = Add_Convolutional_Layer(x, depth//2, 1)
    x = Add_Convolutional_Layer(x, depth, size)
    x = layers.Add()([inp,x])
    return x


def Add_Residual_Set(x, out_depth, size, num_sets):
    for i in range(num_sets):
        x = Add_Residual(x, out_depth, size)

    return x


def Darknet():
    inputs = keras.Input(shape = (448,448,3))
    x = inputs
    x = Add_Convolutional_Layer(x, 32, 3)
    x = Add_Convolutional_Layer(x, 64, 3, 2)
    x = Add_Residual(x, 64, 3)

    x = Add_Convolutional_Layer(x, 128, 3, 2)
    x = Add_Residual_Set(x, 128, 3, 2)

    x = Add_Convolutional_Layer(x, 256, 3, 2)
    x = Add_Residual_Set(x, 256, 3, 8)

    x = Add_Convolutional_Layer(x, 512, 3, 2)
    x = Add_Residual_Set(x, 512, 3, 8)

    x = Add_Convolutional_Layer(x, 1024, 3, 2)
    x = Add_Residual_Set(x, 1024, 3, 4)

    model = keras.Model(inputs = inputs, outputs = x)
    return model

dn = Darknet()

keras.utils.plot_model(dn, 'darknet.png')
    
