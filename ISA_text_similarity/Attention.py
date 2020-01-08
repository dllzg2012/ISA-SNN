from keras import backend as K
from keras.layers import  Layer, initializers, regularizers, constraints
import tensorflow as tf


class ISA(Layer):
    def __init__(self, step_dim,get_alpha=False,return_sequence=False,
                 W_regularizer=None, b_regularizer=None,L_regularizer=None,
                 W_constraint=None, b_constraint=None,L_constraint=None,
                 bias=False,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.l_init = initializers.constant(value=0.5)
        self.return_sequence=return_sequence
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.L_regularizer = regularizers.get(L_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.L_constraint = constraints.get(L_constraint)
        self.bias=bias
        self.step_dim = step_dim
        self.get_alpha=get_alpha
        self.features_dim = 0
        super(ISA, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3

        self.W1 = self.add_weight((2,input_shape[0][-1] * input_shape[0][-1],),
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
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        xavg=K.mean(x[0], axis=1, keepdims=True)
        yavg = K.mean(x[1], axis=1, keepdims=True)
        xcon=K.concatenate([x[0],yavg],axis=1)
        ycon = K.concatenate([x[1], xavg], axis=1)
        xw=K.dot(xcon, K.reshape(self.W1[0], (features_dim, features_dim)))
        yw = K.dot(ycon, K.reshape(self.W1[1], (features_dim, features_dim)))
        xwt=K.permute_dimensions(xw,[0,2,1])
        ywt = K.permute_dimensions(yw, [0, 2, 1])
        xws=K.batch_dot(xw,xwt)/ (step_dim ** 0.5)
        yws = K.batch_dot(yw, ywt)/ (step_dim ** 0.5)
        print(xws.shape)
        xws=K.softmax(xws)
        yws=K.softmax(yws)
        xws=xws[:,:-1,:-1]
        yws = yws[:, :-1, :-1]
        print(xws.shape)
        VX=K.dot(x[0],K.reshape(self.W1[0], (features_dim, features_dim)))
        VY=K.dot(x[1],K.reshape(self.W1[1], (features_dim, features_dim)))
        Vx=VX*K.mean(xws,axis=2,keepdims=True)
        if self.get_alpha:
            return xws
        elif self.return_sequence:
            return VX
        else:
            return K.sum(Vx, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0][0], input_shape[0][1],input_shape[0][2]
        if self.get_alpha:
            return input_shape[0][0],input_shape[0][1],input_shape[0][2]
        elif self.return_sequence:
            return input_shape[0][0],input_shape[0][1],input_shape[0][2]
        else:
            return input_shape[0][0], self.features_dim


