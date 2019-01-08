# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 08:45:26 2019

@author: Administrator
"""

import numpy as np
import os
from keras.models import load_model
import cv2
from datetime import datetime

import model_unet

'''
文件路径
'''
model_dir = os.path.join(os.getcwd(), 'Saved_models')
model_path = os.path.join(model_dir, 'unet_membrane.08-0.1536.hdf5')  # 模型

pre_image_dir = os.path.join(os.getcwd(), 'pre_image')
pre_image_path = os.path.join(pre_image_dir, 'data_test3.png')  # 原图
pre_image_result = os.path.join(pre_image_dir, 'data_test3_result_08.png')  # 结果图
'''
预测图像的大小
'''
IMAGE_SIZE = 1024

def load_image(image_path, IMAGE_SIZE):
    '''
    加载需要预测的图像
    然后做一些必要的转换
    '''
    # 加载预测数据并重置大小
    x_pre = cv2.imread(image_path, -1)
    x_pre = cv2.resize(x_pre, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 保存重置后的图像
    cv2.imwrite(r'.\pre_image\data_test3_resize.png', x_pre)
    
    # 类型转换
    x_pre = x_pre.astype('float32')
    x_pre = x_pre / 255
    
    return x_pre

def model_prediction(image_path, IMAGE_SIZE, model_path, image_result):
    '''
    1、加载模型
    2、读取图像
    3、预测
    4、保存
    ''' 
    # 加载模型参数
    print('{}: Load model weights...'.format(datetime.now().strftime('%c')))
    #model = load_model(save_model_path)
    model = model_unet.unet(IMAGE_SIZE = IMAGE_SIZE)
    model.load_weights(model_path)
    #model.summary()
    
    # 加载图像并添加一维以适应模型的输入
    print('{}: Load predict image...'.format(datetime.now().strftime('%c')))
    x_pre = load_image(image_path, IMAGE_SIZE)
    x_pre = x_pre[np.newaxis, :, :, :]
    
    #模型预测并调整越策结果
    print('{}: Start model predict...'.format(datetime.now().strftime('%c')))
    y_pre = model.predict(x_pre)
    y_pre = np.squeeze(y_pre, axis = 0)
    y_pre = np.round(y_pre)
    y_pre *= 255
   
    # 保存预测结果
    cv2.imwrite(pre_image_result, y_pre)
    print('{}: Save prefict result success!'.format(datetime.now().strftime('%c')))
    

if __name__ == '__main__':
    '''
    主函数
    '''
    model_prediction(image_path=pre_image_path, 
                     IMAGE_SIZE=IMAGE_SIZE, 
                     model_path=model_path, 
                     image_result=pre_image_result)
    








