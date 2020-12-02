from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
import numpy as np
from keras import Input
from keras import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, DepthwiseConv2D, Add, Concatenate
from keras.layers.experimental.preprocessing import Rescaling
from keras.activations import relu
from keras.initializers import Constant
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image_dataset_from_directory
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.0/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so"

def get_directional_filters_initializer_np():
	#return the directional filters tensor
	#it should have shape height*width*1*num_filters
	dir_filter_minus63 = np.array([[0,-0.0313, -0.0313,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0.2813,0.2813,0,0,0],
		[0,0,0,1,0,0,0],[0,0,0,0.2813,0.2813,0,0],[0,0,0,0,0,0,0],[0,0,0,0,-0.0313, -0.0313,0]])
	dir_filter_minus45 = np.array(np.mat('-0.0625,0,0,0,0,0,0;0,0,0,0,0,0,0;0,0,0.5625,0,0,0,0;0,0,0,1,0,0,0;\
		0,0,0,0,0.5625,0,0;0,0,0,0,0,0,0;0,0,0,0,0,0,-0.0625'))
	dir_filter_minus26 = np.transpose(dir_filter_minus63)
	dir_filter0 = np.vstack((np.zeros((3,7)),np.array([-0.0625,0,0.5625,1,0.5625,0,-0.0625]),np.zeros((3,7))))
	dir_filter26 = np.fliplr(dir_filter_minus26)
	dir_filter45 = np.fliplr(dir_filter_minus45)
	dir_filter63 = np.fliplr(dir_filter_minus63)
	dir_filter90 = np.transpose(dir_filter0)
	dir_filters = np.dstack((dir_filter_minus63,dir_filter_minus45,dir_filter_minus26,dir_filter0,dir_filter26,
		dir_filter45,dir_filter63,dir_filter90))
	dir_filters = np.reshape(dir_filters,(7,7,1,8))
	return dir_filters

def get_matched_filters_initializer_np():
	#return the matched filters tensor
	#it should have shape height*width*1*num_filters
	matched_filter_minus63 = np.transpose(np.array(np.mat('0,0.25,0,0,0,0,0;0,0.5,0.5,0,0,0,0;0,0,1,1,0,0,0;0,0,0,2,0,0,0;\
		0,0,0,-2,0,0,0;0,0,0,-1,-1,0,0;0,0,0,0,-0.5, -0.5,0')))
	matched_filter_minus45 = np.array(np.mat('0.25,0,0,0,0,0,0;0,0.5,0,0,0,0,0;0,0,1,0,0,0,0;0,0,0,2,0,0,0;0,0,0,0,-2,0,0;\
		0,0,0,0,0,-1,0;0,0,0,0,0,0,-0.5'))
	matched_filter_minus26 = np.transpose(matched_filter_minus63)
	matched_filter0 = np.transpose(np.vstack((np.zeros((3,7)),np.array([0.25,0.5,1,2,-2,-1,-0.5]),np.zeros((3,7)))))
	matched_filter26 = np.fliplr(matched_filter_minus26)
	matched_filter45 = np.fliplr(matched_filter_minus45)
	matched_filter63 = np.fliplr(matched_filter_minus63)
	matched_filter90 = np.transpose(matched_filter0)
	matched_filters = np.dstack((matched_filter_minus63,matched_filter_minus45,matched_filter_minus26,matched_filter0,
		matched_filter26,matched_filter45,matched_filter63,matched_filter90))
	matched_filters = np.reshape(matched_filters,(7,7,1,8))
	return matched_filters

with tf.device('/device:GPU:0'):

  test_path = '/content/drive/MyDrive/Colab Notebooks/Replica_name_plain2'
  class_names = ['CS1','CS2','CS3','CS4','CS5','CS6']

  test_ds = image_dataset_from_directory(test_path,image_size=(192, 128),color_mode='grayscale',batch_size=5544)

  test_data = np.zeros((5544,192,128))
  test_label = np.zeros((5544),'uint8')

  i = 0
  for images, labels in test_ds.take(1):
    test_data = images.numpy().astype('uint8')
    test_label = labels.numpy()
    i += 1
  
  kfold = KFold(n_splits=5, shuffle=True)

  # Define per-fold score containers
  acc_per_fold = []
  loss_per_fold = []

  normalized_data = np.zeros((5544,192,128,1))
  for i in range(5544):
    m = np.max(test_data[i,:,:])
    rescale = Rescaling(1./m)
    normalized_data[i,:,:,:] = rescale(test_data[i,:,:])

  normalized_data = normalized_data.squeeze()
  fold_no = 1

  for train,test in kfold.split(normalized_data,test_label):
    ############################# The model with directional filter #############################
    input_images = Input(shape=(192,128,1), name="input_images")

    # A regular layer
    out_regular = Conv2D(8, (3,3), padding='same', kernel_regularizer='l2')(input_images)
    out_norm 	= BatchNormalization()(out_regular)
    out_rel0 	= relu(out_norm)
    out_drop 	= Dropout(0.5)(out_rel0)

    # A directional layer
    directional_filters_initializer_np = get_directional_filters_initializer_np()

    directional_filters_initializer_tf = Constant(directional_filters_initializer_np)
    # The special convolution. set use bias to false, set activation to none (linear)
    out_special = Conv2D(8, (7, 7), use_bias=False,padding='same',
                      kernel_initializer=directional_filters_initializer_tf,
                      trainable=False)(input_images)

    # A matched layer
    # Process the special features using the matched filters
    # Each directional filter has a corresponding matched filter
    matched_filter_init_np = get_matched_filters_initializer_np()

    matched_filter_init_tf = Constant(matched_filter_init_np)
    #Here is the depthwise convolution with convolves each feature channel with a filter
    out_speical_matched = DepthwiseConv2D(kernel_size=(7,7),
                              padding='same', depth_multiplier=1, 
                              depthwise_initializer=matched_filter_init_tf,
                              activation=None, use_bias=False, trainable=False)(out_special)
    out_rel1 	= relu(out_speical_matched)
    out_rel1_m 	= relu(-out_speical_matched)
    out_add 	= Add()([out_rel1, out_rel1_m])
    out_norm2 	= BatchNormalization()(out_add)
    out_drop1 	= Dropout(0.5)(out_norm2)

    out_1 = Concatenate(axis=-1)([out_drop, out_drop1]) 
    # From here, ordinary layers
    out_2 		= Conv2D(16, (3,3), kernel_regularizer='l2')(out_1)
    out_3 		= BatchNormalization()(out_2)
    out_rel2 	= relu(out_3)
    out_4 		= MaxPooling2D(pool_size=(2,2))(out_rel2)
    out_5 		= Dropout(0.5)(out_4)

    out_6 		= Conv2D(32, (3,3), kernel_regularizer='l2')(out_5)
    out_7 		= BatchNormalization()(out_6)
    out_conv 	= Conv2D(64, (3,3), kernel_regularizer='l2')(out_7)
    out_batch 	= BatchNormalization()(out_conv)
    out_rel3 	= relu(out_batch)
    out_8 		= MaxPooling2D(pool_size=(2,2))(out_rel3)
    out_9 		= Dropout(0.5)(out_8)
    ''''''
    # End of convolution, flatten and NN
    out_10 = Flatten()(out_9)

    out_11 		= Dense(256, kernel_regularizer='l2')(out_10)
    out_12 		= BatchNormalization()(out_11)
    out_rel4 	= relu(out_12)
    out_13 		= Dropout(0.5)(out_rel4)

    out_last 	= Dense(6,activation='softmax')(out_13)

    model = Model(inputs=input_images, outputs=out_last)

    opt = Adam(learning_rate = 0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.5, patience=4, min_lr=0.00001)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')


    # Train
    history = model.fit(
      normalized_data[train], test_label[train],
      steps_per_epoch=120,
      epochs=80,
      validation_split = 0.25,
      validation_steps = 80,
      callbacks=[reduce_lr])
    
    # Generate generalization metrics
    scores = model.evaluate(normalized_data[test], test_label[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Confusion Matrix
    predicted = model.predict(normalized_data[test])
    predicted = np.argmax(predicted,axis=1)
    cm = confusion_matrix(test_label[test],predicted,[0,1,2,3,4,5])
    print("Confusion matrix")
    print(cm)

    # Increase fold number
    fold_no = fold_no + 1

  # == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')