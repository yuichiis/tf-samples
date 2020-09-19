import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

#x = np.array([
#    [-10.0,  1.0, 10.0],
#    [ 10.0,-10.0,  1.0],
#]);
#t = np.array([2,2])
#func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#x = np.array([
#[-2.0], [2.0] , [0.0]
#]);
#t = np.array([0.0, 1.0 , 0.0])
#func = keras.losses.BinaryCrossentropy(from_logits=True)

def numerical_gradient(
        f,
        *variables,
        h=None):

    if h is None:
        h = 1e-4
    if not callable(f):
        raise Exception("f must callable or array of f and h")

    grads = []
    new_variables = []
    for x in variables:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        new_variables.append(x)
    variables = new_variables

    for x in variables:
        g = np.zeros_like(x)
        grads.append(g)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        h2 = h*2
        with it:
            while not it.finished:
                idx = it.multi_index
                value = xx[idx]
                x[idx] = value + h
                y1 = f(*variables)
                x[idx] = value - h
                y2 = f(*variables)
                d = (y1 - y2)
                g[idx] = np.sum(d)/h2
                x[idx] = value
                it.iternext()
    return grads


x = np.array([
    [0,1,2,9],
    #[1],
])
#t = np.array(
#    [1,0]
#)
x = keras.utils.to_categorical(x.reshape(4,), num_classes=10).reshape(1,4,10)
#x = keras.utils.to_categorical(x.reshape(1,), num_classes=10).reshape(1,1,10)
def my_init(shape, dtype=None):
    return tf.fill(shape,2.)
func = keras.layers.GRU(
        10,# units
        input_shape=(4,10),
        #input_shape=(1,10),
        activation=None,
        #recurrent_activation=None,
        return_sequences=True,
        kernel_initializer='ones',
        recurrent_initializer='ones',
        bias_initializer='zeros',
        #reset_after=False,
    )
#kernel = np.full([10,3],0.01);
#recurrent = np.full([3,3],0.01);
#bias = np.zeros([3]);
#func.build((4,10))
#func.weights[0][...] = kernel[...]

xx = tf.Variable(x)
#tt = tf.Variable(t)
with tf.GradientTape() as g:
    #yy = func(tt,xx)
    yy = func(xx)
dy_dx = g.gradient(yy,xx)
print(yy)
print(dy_dx)

grads = numerical_gradient(func,x,h=1e-3)
print(grads[0])
