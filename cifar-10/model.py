import keras
import keras.layers as layers
import keras.models as models
from keras import backend as K


# custom layer for gauss membership function
class GaussMembership(layers.Layer):
    
    def __init__(self, num_rules, epsilon=1e-8, **kwargs):
        self.epsilon = epsilon
        self.num_rules = num_rules
        super(GaussMembership, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        self.mu = self.add_weight(
            name="mu",
            shape=(self.num_rules, input_shape[1]),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        self.sigma = self.add_weight(
            name="sigma",
            shape=(self.num_rules, input_shape[1]),
            initializer=keras.initializers.Ones(),
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        super(GaussMembership, self).build(input_shape)
    
    
    def call(self, x):
        x = K.expand_dims(x, axis=1)
        x = K.square((x - self.mu) / (self.sigma + self.epsilon))
        return K.exp(-0.5 * x)
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_rules, input_shape[1])
    
    
class LogGaussMF(layers.Layer):
    
    def __init__(
        self, 
        rules,
        mu_initializer="zeros",
        beta_initializer="ones",
        mu_regularizer=None,
        beta_regularizer=None,
        activity_regularizer=None,
        mu_constraint=None,
        beta_constraint="nonneg",
        **kwargs
    ):
        super(LogGaussMF, self).__init__(**kwargs)
        self.rules = int(rules)
        self.mu_initializer = keras.initializers.get(mu_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.mu_regularizer = keras.regularizers.get(mu_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.mu_constraint = keras.constraints.get(mu_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        
        
    def build(self, input_shape):
        self.mu = self.add_weight(
            name="mu",
            shape=(self.rules, input_shape[1]),
            initializer=self.mu_initializer,
            regularizer=self.mu_regularizer,
            trainable=True)
        self.beta = self.add_weight(
            name="beta",
            shape=(self.rules, input_shape[1]),
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            trainable=True)
        super(LogGaussMF, self).build(input_shape)
    
    
    def call(self, x):
        x = K.expand_dims(x, axis=1)
        return -0.5 * K.square(x - self.mu) * self.beta
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.rules, input_shape[1])
    
    
# multiplication layer lambda function
def normalized_product_fn(x):
    x = K.sum(K.log(x + 1e-8), axis=2)
    return K.exp(x - K.max(x, axis=1, keepdims=True))