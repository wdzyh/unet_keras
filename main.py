# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:08:15 2019

@author: zhangyonghui
"""

'''
系统包
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
'''
私人包
'''
import read_data 
import model_unet
'''
文件路径
'''
image_dir = os.path.join(os.getcwd(), 'ADEChallengeData2016')
mat_dir = os.path.join(os.getcwd(), 'dataset.mat')
save_dir = os.path.join(os.getcwd(), 'Saved_models')
save_model_path = os.path.join(save_dir, 'unet_membrane.{epoch:02d}-{val_loss:.4f}.hdf5')
weights_save_path = os.path.join(save_dir, 'Unet_weights')
'''
超参数
'''
IMAGE_SIZE = 128
batch_size = 2
epochs = 10

# 读取图像数据
train_images, train_labels, valid_images, valid_labels = read_data.read_image_creat_mat(image_dir, 
                                                                                        IMAGE_SIZE, 
                                                                                        mat_dir)
train_labels = train_labels[:, :, :, np.newaxis]
valid_labels = valid_labels[:, :, :, np.newaxis]

#类型转换
train_images = train_images.astype('float32') 
train_labels = train_labels.astype('float32')
valid_images = valid_images.astype('float32')
valid_labels = valid_labels.astype('float32')

train_images /= 255
train_labels /= 255
valid_images /= 255
valid_labels /= 255

# 导入模型
model = model_unet.unet(IMAGE_SIZE=IMAGE_SIZE)

#模型设置
model_checkpoint = ModelCheckpoint(save_model_path, monitor='loss',verbose=1, save_best_only=True, mode='auto', period=1)
EarlyStopping = EarlyStopping(monitor='loss', patience=50, verbose=1)
tensorboard = TensorBoard(log_dir=weights_save_path, histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)

callback_lists = [model_checkpoint, EarlyStopping, tensorboard, reduce_lr]

# 训练
model.fit(train_images,
          train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(valid_images, valid_labels),
          shuffle=True,
          callbacks=callback_lists)





