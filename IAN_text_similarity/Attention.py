from keras import backend as K
from keras.layers import  Layer, initializers, regularizers, constraints
import tensorflow as tf


class mm_IAN(Layer):
    def __init__(self, step_dim,get_alpha=False,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.get_alpha=get_alpha
        super(mm_IAN, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3

        self.W = self.add_weight((input_shape[0][-1] * input_shape[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[0][-1]
        if self.bias:
            self.b = self.add_weight((input_shape[0][1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        '''
        xw=K.dot(x[0], K.reshape(self.W, (features_dim, features_dim)))
        yavg=K.mean(x[1], axis=1, keepdims=True)
        yavg=K.permute_dimensions(yavg,[0,2,1])
        eij = K.reshape(
            K.batch_dot(xw,yavg), (-1, step_dim))
        '''
        eij = K.reshape(K.dot(K.dot(K.reshape(x[0], (-1, features_dim)),
                              K.reshape(self.W, (features_dim, features_dim))),K.reshape(K.transpose(K.mean(K.mean(x[1], axis=1),axis=0)),(features_dim,-1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x[0] * a
        print(weighted_input.get_shape())
        if self.get_alpha:
            return a
        else:
            return K.sum(weighted_input, axis=1)

    def get_config(self):
        config = {
            'step_dim': self.step_dim
        }
        base_config = super(mm_IAN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.get_alpha:
            return input_shape[0][0],input_shape[0][1],1
        else:
            return input_shape[0][0], self.features_dim

