from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, BatchNormalization, Flatten, Dropout, Concatenate, AveragePooling2D, Layer, Permute, Conv2DTranspose, UpSampling2D, Add
from tensorflow.keras.models import model_from_json, Model
from keras.activations import softmax
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from tensorflow import keras

def scheduler(epoch, lr):
	lr = 0.0005 / (1 + epoch/2)
	print("New learning rate:", lr)
	return lr

def lossFun(y_true, y_pred):
	loss = 0

	loss_obj =  K.sum(y_true[...,0] * K.square(y_pred[...,0] - y_true[...,0]), axis=[1,2]) / (K.sum(y_true[...,0], axis=[1,2])+1)
	loss_nobj = K.sum(K.abs(y_true[...,0]-1) * K.square(y_pred[...,0] - y_true[...,0]), axis=[1,2]) / (K.sum(K.abs(y_true[...,0]-1), axis=[1,2])+1)
	loss_mid = K.sum(y_true[...,0] * K.mean(K.square(y_pred[...,1:3] - y_true[...,1:3]), axis = 3), axis=[1,2]) / (K.sum(y_true[...,0], axis=[1,2])+1)
	loss_box = K.sum(y_true[...,0] * K.mean(K.square(K.sqrt(y_pred[...,3:5]) - K.sqrt(y_true[...,3:5])), axis = 3), axis=[1,2]) / (K.sum(y_true[...,0], axis=[1,2])+1)
	loss_class = K.sum(y_true[...,0] * K.mean(K.square(y_pred[...,5:] - y_true[...,5:]), axis = 3), axis=[1,2]) / (K.sum(y_true[...,0], axis=[1,2])+1)

	# FOCAL LOSS
	# pt = y_true[...,0] * K.square(y_pred[...,0]) + (1-y_true[...,0]) * 1-y_pred[...,0])
	# CE = -1 * K.log(pt) * K.pow(1-pt, 3)
	# loss_obj = K.mean(y_true[...,0] * CE, axis=[1,2])
	# loss_nobj = K.mean(K.abs(y_true[...,0]-1) * CE, axis=[1,2])

	loss += loss_obj
	loss += loss_nobj 
	loss += loss_mid 
	loss += loss_box 
	loss += loss_class

	return loss


def build_model(input_h, input_w, out_classes = 3):
	inp = keras.Input(shape=(input_h,input_w,3))
	
	# ############################### STEM #################################
	n_channels = [32, 32]
	x = Conv2D(n_channels[0], kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(inp)
	xout = Conv2D(n_channels[1], kernel_size=(3,3), strides=(2,2), padding="same", activation='mish')(x)

	# ############################### TRUNK ################################

	trunk_architecture = [[32, 32, 64, 4], [32, 64, 128, 8], [64, 128, 256, 4]]

	for architecture in trunk_architecture:
		ss1, ss2, ss3, count = architecture
		xx = Conv2D(ss1, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xout)
		for _ in range(count):
			x = Conv2D(ss1, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xx)
			xx = Add()([xx, x])
		x = Conv2D(ss2, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xx)
		xout = Add()([x, xout])
		xout = Conv2D(ss3, kernel_size=(3,3), strides=(2,2), padding="same", activation='mish')(xout)

	# ############################### LEAFS ################################
	n_channels_leafs = 64
	xobj1 = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xout)
	xmid1 = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xout)
	xbox1 = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xout)
	xclass1 = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xout)

	xobj = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xobj1)
	xmid = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xmid1)
	xbox = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xbox1)
	xclass = Conv2D(n_channels_leafs, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(xclass1)

	xobj = Conv2D(1, kernel_size=(1,1), strides=(1,1), activation='linear', padding="same")(xobj)
	xmid = Conv2D(2, kernel_size=(1,1), strides=(1,1), activation='linear', padding="same")(xmid)
	xbox = Conv2D(2, kernel_size=(1,1), strides=(1,1), activation='linear', padding="same")(xbox)
	xclass = Conv2D(out_classes, kernel_size=(1,1), strides=(1,1), activation='sigmoid', padding="same")(xclass)

	x = Concatenate()([xobj, xmid, xbox, xclass])

	model = Model(inputs=inp, outputs=x)

	return model