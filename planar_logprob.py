import keras
import keras.backend as K

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras_normalizing_flows.layers.flows import PlanarFlow,RadialFlow
from keras_normalizing_flows.utils.plt_utils import draw_heatmap
tfd = tfp.distributions
sess = K.get_session()

'''This script shows what kind of transformation the planar flow does
    Uses tensorflow_probability to explicitly define an output distribution
    that we can use to compute the logprobability. So don't need the ELBO
    
    converts unimodal normal to multimodal
'''

    
n_samples = 10000
n_dim = 2

#Generate the prior distribution
prior_dist = tfd.MultivariateNormalDiag(loc = [0., 0.],
                                        scale_diag = [1., 1.])
data = prior_dist.sample(n_samples).eval(session=sess)
y_data = np.zeros_like(data) #not actually used for fitting

#Generate the target distribution
mix = 0.5
output_dist = tfd.Mixture(
  cat=tfd.Categorical(probs=[mix, 1.-mix]),
  components=[
    tfd.MultivariateNormalDiag(loc = [2, 2],
                               scale_diag = [1., 1.]),
    tfd.MultivariateNormalDiag(loc = [-2, -2],
                           scale_diag = [1., 1.]),
])
    
   
#simple model with 1 flow
input_layer = keras.layers.Input(shape=(n_dim,))
out,logdetjac = PlanarFlow().apply(input_layer)



def loss(not_used,model_output):
    return -K.mean(output_dist.log_prob(model_output) )

model = keras.models.Model(input_layer, out)
optimizer = keras.optimizers.Adamax()

model.compile(optimizer=optimizer,loss=loss)


early_stop = keras.callbacks.EarlyStopping(patience=10)
model.fit(x=data,y= y_data,validation_split=0.8,epochs=200,callbacks=[early_stop])

valid_data = prior_dist.sample(n_samples).eval(session=sess)
valid_out = model.predict(valid_data)

draw_heatmap(output_dist.sample(n_samples).eval(session=sess),name='Target Distribution')
draw_heatmap(valid_data, name='Base Distribution')
draw_heatmap(valid_out,name='Transformed Distribution')