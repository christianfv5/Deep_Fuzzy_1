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
    
    
# multiplication layer lambda function
def normalized_product_fn(x):
    x = K.sum(K.log(x + 1e-8), axis=2)
    return K.exp(x - K.max(x, axis=1, keepdims=True))