
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input,Dense,Activation
from keras_normalizing_flows.layers.flows import Gaussian,PlanarFlow,RadialFlow
from keras_normalizing_flows.utils.plt_utils import draw_heatmap
import sys
import numpy as np
import matplotlib.pyplot as plt


'''Uses a stack of basic flows to transform the VAE prior distribution
    into a more complex distribution before sending it into a NN decoder

'''
    
def draw_scattered_samples(all_points,labels):
    for idx in range(10):
        points = all_points[labels==idx]
        plt.scatter(points[:,0],points[:,1])
    plt.show()

def process_data(data):
    data = data/255.
    data = np.reshape(data,(data.shape[0],-1))
    return data
train_set,test_set = mnist.load_data()
X_train,y_train = train_set
X_test,y_test = test_set


X_train,X_test = (process_data(X_train),process_data(X_test))


#############
### Model ###
#############
hidden_size = 128
latent_size = 2
input_shape = X_train.shape[1:]
ndim = input_shape[0]
input_layer = Input(shape=input_shape)

#Encoder
enc_hidden = Dense(hidden_size)(input_layer)
enc_hidden = Activation('relu')(enc_hidden)
z_mean = Dense(latent_size,name='z_mean')(enc_hidden)
z_log_var = Dense(latent_size,name='z_log_var',
                  kernel_initializer=keras.initializers.constant(-0.1)
                  )(enc_hidden)

kl_loss = 0
#Sampler
sampler = Gaussian()
sample = sampler([z_mean,z_log_var])
kl_loss += sampler.kl_divergence([z_mean,z_log_var])

num_flows = 10
flow_out = sample
for idx in range(num_flows):
    if idx % 2 == 0:
        flower = PlanarFlow()
    else:
        flower = RadialFlow()
    flow_out,logdetjac = flower.apply(flow_out)
    kl_loss -= logdetjac
    
#Decoder
output = Dense(hidden_size)(flow_out)
output = Activation('relu')(output)
output = Dense(ndim)(output)
output = Activation('sigmoid')(output)
model = keras.models.Model(input_layer, output)

################
### Training ###
################

def vae_loss(y_true,y_pred):
    recon_loss =  ndim*keras.losses.binary_crossentropy(
                                    K.flatten(y_true),
                                    K.flatten(y_pred))
    return K.mean(recon_loss + kl_loss)



optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer,loss=vae_loss)
early_stop = keras.callbacks.EarlyStopping(patience=5)
model.fit(x=X_train,y=X_train,validation_split=0.8,epochs=100,callbacks=[early_stop])

encoder = keras.backend.Function([model.input],[z_mean])
transformed_dist = keras.backend.Function([sample],[flow_out])

encoded = encoder([X_test])[0]
encoded_out = transformed_dist([encoded])[0]
draw_heatmap(encoded,name='Base Distribution')
draw_heatmap(encoded_out,name='Transformed Distribution')
draw_scattered_samples(encoded,y_test)
draw_scattered_samples(encoded_out,y_test)