#!/usr/bin/env python3
	# -*- coding: utf-8 -*-
	"""
	Created on Sat Nov 12 01:09:17 2016
	
	@author: stephen
	"""
	#SBATCH --mem=88437
	#SBATCH -c 14
	

	from tensorflow import keras
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from keras.callbacks import EarlyStopping
	from keras.callbacks import ModelCheckpoint
	from keras.layers import LeakyReLU
	

	np.random.seed(813306)
	 
	def build_resnet(input_shape, n_feature_maps, nb_classes):
	    print ('build conv_x')
	    x = keras.layers.Input(shape=(input_shape))
	    conv_x = keras.layers.BatchNormalization()(x)
	    conv_x = keras.layers.Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
	    conv_x = keras.layers.BatchNormalization()(conv_x)
	    conv_x = keras.layers.LeakyReLU(alpha=.01)(conv_x)
	     
	    print ('build conv_y')
	    conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
	    conv_y = keras.layers.BatchNormalization()(conv_y)
	    conv_y = keras.layers.LeakyReLU(alpha=.01)(conv_y)
	     
	    print ('build conv_z')
	    conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
	    conv_z = keras.layers.BatchNormalization()(conv_z)
	     
	    is_expand_channels = not (input_shape[-1] == n_feature_maps)
	    if is_expand_channels:
	        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1,padding='same')(x)
	        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
	    else:
	        shortcut_y = keras.layers.BatchNormalization()(x)
	    print ('Merging skip connection')
	    y = keras.layers.Add()([shortcut_y, conv_z])
	    y = keras.layers.LeakyReLU(alpha=.01)(y)
	     
	    print ('build conv_x')
	    x1 = y
	    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
	    conv_x = keras.layers.BatchNormalization()(conv_x)
	    conv_x = keras.layers.LeakyReLU(alpha=.01)(conv_x)
	     
	    print ('build conv_y')
	    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
	    conv_y = keras.layers.BatchNormalization()(conv_y)
	    conv_y = keras.layers.LeakyReLU(alpha=.01)(conv_y)
	     
	    print ('build conv_z')
	    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
	    conv_z = keras.layers.BatchNormalization()(conv_z)
	     
	    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
	    if is_expand_channels:
	        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
	        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
	    else:
	        shortcut_y = keras.layers.BatchNormalization()(x1)
	    print ('Merging skip connection')
	    y = keras.layers.Add()([shortcut_y, conv_z])
	    y = keras.layers.LeakyReLU(alpha=.01)(y)
	     
	    print ('build conv_x')
	    x1 = y
	    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
	    conv_x = keras.layers.BatchNormalization()(conv_x)
	    conv_x = keras.layers.LeakyReLU(alpha=.01)(conv_x)
	     
	    print ('build conv_y')
	    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
	    conv_y = keras.layers.BatchNormalization()(conv_y)
	    conv_y = keras.layers.LeakyReLU(alpha=.01)(conv_y)
	     
	    print ('build conv_z')
	    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
	    conv_z = keras.layers.BatchNormalization()(conv_z)
	

	    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
	    if is_expand_channels:
	        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
	        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
	    else:
	        shortcut_y = keras.layers.BatchNormalization()(x1)
	    print ('Merging skip connection')
	    y = keras.layers.Add()([shortcut_y, conv_z])
	    y = keras.layers.LeakyReLU(alpha=.01)(y)
	     
	    full = keras.layers.GlobalAveragePooling2D()(y)
	    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
	    print ('        -- model was built.')
	    return x, out
	 
	       
	def readucr(filename):
	    with open(filename, 'r', encoding='utf-8-sig') as f: 
	        data = np.genfromtxt(f,dtype = float, delimiter = ';')
	        Y = data[:,15]
	        print(Y)
	        X = data[:,:14]
	        return X, Y
	   
	nb_epochs = 10000
	 
	 
	#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
	#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
	#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
	#'NonInvasiveFatalECG_Thorax0', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
	#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']
	

	flist  = [['corn'],['djia'],['ftse'],['gold'],['hyield'],['nikkei'],['oil'],['pequity'],['realestate'],['tbond']]
	for each in flist:
	    fname = each
	    x_train, y_train = readucr('/work/al324/TDA/run2/UCR_Time_Series_Classification_Deep_Learning_Baseline/newData'+'/'+fname+'_train.csv')
	    
	        
	    x_test, y_test = readucr('/work/al324/TDA/run2/UCR_Time_Series_Classification_Deep_Learning_Baseline/newData'+'/'+fname+'_test.csv')
	    
	    
	    nb_classes = len(np.unique(y_test))
	    print(nb_classes)
	    batch_size = min(x_train.shape[0]/10, 16)
	    print(batch_size)
	    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
	    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
	    Y_train = keras.utils.to_categorical(y_train, nb_classes)
	    Y_test = keras.utils.to_categorical(y_test, nb_classes)
	    x_train_mean = x_train.mean()
	    x_train_std = x_train.std()
	    x_train = (x_train - x_train_mean)/(x_train_std)
	    x_test = (x_test - x_train_mean)/(x_train_std)
	    x_train = x_train.reshape(x_train.shape + (1,1,))
	    x_test = x_test.reshape(x_test.shape + (1,1,))
	     
	    print(x_train)
	    print(Y_train)
	    print(x_train)
	    print(y_test)
	    x , y = build_resnet(x_train.shape[1:], 64, nb_classes)
	    model = keras.models.Model(inputs=x, outputs=y)
	    optimizer = keras.optimizers.Adam(learning_rate=0.00000001)
	    model.compile(loss='categorical_crossentropy',
	                  optimizer=optimizer,
	                  metrics=['accuracy'])
	      
	    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.00000001,
	                      patience=0, min_lr=0.000000005)
	    mc = keras.callbacks.ModelCheckpoint(fname+'weights2{epoch:08d}.h5', 
	                                     save_weights_only=True,period=500)
	    mcp_save = ModelCheckpoint('besttbond.h5', save_best_only=True, monitor='val_loss', mode='min') 
	    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
	              verbose=1, validation_data=(x_test, Y_test), callbacks = [mc] )
	    log = pd.DataFrame(hist.history)
	    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])
	   
	    # Plot training & validation accuracy values
	    plt.plot(hist.history['acc'])
	    plt.plot(hist.history['val_acc'])
	    plt.title('Model accuracy')
	    plt.ylabel('Accuracy')
	    plt.xlabel('Epoch')
	    plt.legend(['Train', 'Test'], loc='upper left')
	    plt.savefig('realestate.pdf')
	

	    # Plot training & validation loss values
	    plt.plot(hist.history['loss'])
	    plt.plot(hist.history['val_loss'])
	    plt.title('Model loss')
	    plt.ylabel('Loss')
	    plt.xlabel('Epoch')
	    plt.legend(['Train', 'Test'], loc='upper left')
	    plt.savefig('realestate.pdf')

