import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
start = time.time()

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers
from dir_fil import get_directional_filters_initializer_np
from dir_fil import get_matched_filters_initializer_np

# Replica for Shape
# Batch Size = 32, image size = (192,128), 
# total dataset = 335, total # of classes = 5, 
# training = 268, validation = 67.

# Batch Size = 16, image size = (192,128), 
# total dataset = 256, total # of classes = 4, 
# training = 192, validation = 64.

# MathEdged
# Batch Size = 8, image size = (192,128), 
# total dataset = 150, total # of classes = 6, 
# training = 120, validation = 30.

# Replica for Shape + Mathews and Oregon
# Batch Size = 8, image size = (192,128), 
# total dataset = 150, total # of classes = 6, 
# training = 120, validation = 30.

# Constants
path 		= "Replica_for_augm"
tr_len 		= 1344
val_len 	= 448
batch_size 	= 32
ep_num 		= 28
step_tr 	= 50
step_val 	= 30

# Shuffle and Repeat Values
tr_shuf 	= tr_len//batch_size
tr_rep 		= (ep_num*step_tr)//tr_shuf
val_shuf 	= val_len//batch_size
val_rep 	= (ep_num*step_val)//val_shuf

print(tr_shuf, tr_rep, val_shuf, val_rep)

######################## Read the data using preprocessing ###################################
train_ds = tf.keras.preprocessing.image_dataset_from_directory(path,validation_split=0.25,
	subset="training",seed=123,image_size=(192, 128),color_mode='grayscale',batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(path,validation_split=0.25,
	subset="validation",seed=123,image_size=(192, 128),color_mode='grayscale',batch_size=batch_size)

################################### Plotting the data ########################################
'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(images[i,:,:,0].numpy().astype("uint8"),cmap='gray')
		plt.title(class_names[labels[i]])
		plt.axis("off")
plt.show()
'''

################################# Data Augmentation #########################################
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
	#layers.experimental.preprocessing.RandomFlip("horizontal"),
	layers.experimental.preprocessing.RandomTranslation((-0.1,0.1),(-0.1,0.1)),
	layers.experimental.preprocessing.RandomRotation((-0.01,0.05)),
	layers.experimental.preprocessing.RandomZoom(0.1),
])

######################## Representation of the data augmentation ############################
'''
plt.figure(figsize=(10, 10))
i = 0
for images, _ in train_ds.take(9):
	augmented_images = data_augmentation(images)
	ax = plt.subplot(3, 3, i + 1)
	plt.imshow(augmented_images[0,:,:,0].numpy().astype("uint8"),cmap='gray')
	plt.axis("off")
	i += 1
plt.show()
'''
############################# Configuring the dataset performance ###########################
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

############################# Standartize and Augment the dataset ###########################
normalized_ds = train_ds.map(lambda x, y: (rescale(x), y))
augmented_ds = normalized_ds.map(lambda x, y: (data_augmentation(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (rescale(x), y))
augmented_val_ds = normalized_val_ds.map(lambda x, y: (data_augmentation(x), y))

normalized_ds = normalized_ds.shuffle(tr_shuf, reshuffle_each_iteration=True)
normalized_ds = normalized_ds.repeat(tr_rep)

normalized_val_ds = normalized_val_ds.shuffle(val_shuf, reshuffle_each_iteration=True)
normalized_val_ds = normalized_val_ds.repeat(val_rep)

augmented_ds = augmented_ds.shuffle(tr_shuf, reshuffle_each_iteration=True)
augmented_ds = augmented_ds.repeat(tr_rep)

augmented_val_ds = augmented_val_ds.shuffle(val_shuf, reshuffle_each_iteration=True)
augmented_val_ds = augmented_val_ds.repeat(val_rep)


############################# The model with directional filter #############################
input_images = tf.keras.Input(shape=(192,128,1), name="input_images")

# A regular layer
out_regular = tf.keras.layers.Conv2D(8, (3,3), padding='same', kernel_regularizer='l2')(input_images)
out_norm 	= tf.keras.layers.BatchNormalization()(out_regular)
out_rel0 	= tf.keras.activations.relu(out_norm)
out_drop 	= tf.keras.layers.Dropout(0.5)(out_rel0)

# A directional layer
directional_filters_initializer_np = get_directional_filters_initializer_np()

directional_filters_initializer_tf = tf.constant_initializer(directional_filters_initializer_np)
# The special convolution. set use bias to false, set activation to none (linear)
out_special = tf.keras.layers.Conv2D(8, (7, 7), use_bias=False,padding='same',
									kernel_initializer=directional_filters_initializer_tf,
									trainable=False)(input_images)

# A matched layer
# Process the special features using the matched filters
# Each directional filter has a corresponding matched filter
matched_filter_init_np = get_matched_filters_initializer_np()

matched_filter_init_tf = tf.constant_initializer(matched_filter_init_np)
#Here is the depthwise convolution with convolves each feature channel with a filter
out_speical_matched = tf.keras.layers.DepthwiseConv2D(kernel_size=(7,7),
													padding='same', depth_multiplier=1, 
													depthwise_initializer=matched_filter_init_tf,
													activation=None, use_bias=False, trainable=False)(out_special)
out_rel1 	= tf.keras.activations.relu(out_speical_matched)
out_drop1 	= tf.keras.layers.Dropout(0.5)(out_rel1)

out_1 = tf.keras.layers.Concatenate(axis=-1)([out_drop, out_drop1]) 
# From here, ordinary layers
out_2 		= tf.keras.layers.Conv2D(8, (3,3), kernel_regularizer='l2')(out_drop1)
out_3 		= tf.keras.layers.BatchNormalization()(out_2)
out_rel2 	= tf.keras.activations.relu(out_3)
out_4 		= tf.keras.layers.MaxPooling2D(pool_size=(2,2))(out_rel2)
out_5 		= tf.keras.layers.Dropout(0.5)(out_4)
'''
out_6 		= tf.keras.layers.Conv2D(32, (3,3), kernel_regularizer='l2')(out_5)
out_7 		= tf.keras.layers.BatchNormalization()(out_6)
out_rel3 	= tf.keras.activations.relu(out_7)
out_8 		= tf.keras.layers.MaxPooling2D(pool_size=(2,2))(out_rel3)
out_9 		= tf.keras.layers.Dropout(0.5)(out_8)
'''
# End of convolution, flatten and NN
out_10 = tf.keras.layers.Flatten()(out_5)

out_11 		= tf.keras.layers.Dense(64, kernel_regularizer='l2')(out_10)
out_12 		= tf.keras.layers.BatchNormalization()(out_11)
out_rel4 	= tf.keras.activations.relu(out_12)
out_13 		= tf.keras.layers.Dropout(0.5)(out_rel4)

out_last 	= tf.keras.layers.Dense(4,activation='softmax')(out_13)

model = tf.keras.Model(inputs=input_images, outputs=out_last)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# To store the weights of the best performing epoch, val_acc is the parameter
# checkpoint_filepath = '/tmp/checkpoint'
# checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,monitor='val_sparse_categorical_accuracy',mode='max',save_best_only=True)

#################################### Summary and Plot of the Model ########################
'''
model.summary()

# Try to avoid this in my PC.
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
plot_model(model, to_file='model.png')

'''
############################################ Model Train ###################################
history = model.fit(normalized_ds,steps_per_epoch = step_tr,epochs = ep_num,
	validation_data = normalized_val_ds,validation_steps = step_val)

# model.load_weights('best_weights.hdf5')
# model.save('shapes_cnn.h5')

######################################## Visualization of the Model #######################
im_path = 'Replica_for_shape/CS4/112.png'
from keras.preprocessing import image
img = image.load_img(im_path,target_size=(192,128),color_mode='grayscale')
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.

plt.figure()
plt.title('Original Image')
plt.imshow(img_tensor[0,:,:,0], cmap='gray')

layer_outputs = [layer.output for layer in model.layers[:10]] # Extracts the outputs of the top 6 layers
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

activations = activation_model.predict(img_tensor)

layer_names = []
for layer in model.layers[:10]:
	layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

images_per_row = 4

for layer_name, layer_activation in zip(layer_names, activations):# Displays the feature maps
	n_features = layer_activation.shape[-1] # Number of features in the feature map
	rsize = layer_activation.shape[1] #The feature map has shape (1, rsize, csize, n_features).
	csize = layer_activation.shape[2]
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((rsize * n_cols, images_per_row * csize))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = layer_activation[0,:,:,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * rsize : (col + 1) * rsize, # Displays the grid
						row * csize : (row + 1) * csize] = channel_image
	rscale = 1. / rsize
	cscale = 1. / csize
	plt.figure(figsize=(cscale * display_grid.shape[1],rscale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='gray')
plt.show()
''''''
##################################### Elapsed Time #########################################
end = time.time()
print('Elapsed time is %d seconds' %(end-start))