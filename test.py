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
import glob
from sklearn.metrics import precision_score, recall_score, f1_score
import json

import model_unet

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"
    
    
def calculation(y_label, y_pre, IMAGE_SIZE):
    '''
    本函数主要计算以下评估标准的值：
    1、精准率
    2、召回率
    3、F1分数
    '''
    
    # 去除=1的维度
    y_label = np.squeeze(y_label)
    y_pre = np.squeeze(y_pre) 
    
    # 转成列向量
    y_label = np.reshape(y_label, (IMAGE_SIZE*IMAGE_SIZE, 1))
    y_pre = np.reshape(y_pre, (IMAGE_SIZE*IMAGE_SIZE, 1))
    
    # 精准率
    precision = precision_score(y_label, y_pre)
    print('{}: precision: {:.5f}'.format(datetime.now().strftime('%c'), precision))

    # 召回率
    recall = recall_score(y_label, y_pre)
    print('{}: recall: {:.5f}'.format(datetime.now().strftime('%c'), recall))
        
    # F1
    f1 = f1_score(y_label, y_pre)
    print('{}: f1: {:.5f}'.format(datetime.now().strftime('%c'), f1))
    
    return precision, recall, f1

    
def load_image(image_path, label_path, IMAGE_SIZE):
    '''
    加载需要预测的图像
    然后做一些必要的转换
    '''
    # 加载预测数据并重置大小
    x_pre = cv2.imread(image_path, -1)
    x_pre = x_pre[0:IMAGE_SIZE, 0:IMAGE_SIZE, :]
    
    y_label = cv2.imread(label_path, -1)
    y_label = y_label[0:IMAGE_SIZE, 0:IMAGE_SIZE]

    # 保存重置后的图像
    cv2.imwrite(r'.\pre_image\data_test3_resize.png', x_pre)
    cv2.imwrite(r'.\pre_image\data_test_gt3_resize.png', y_label)
    
    # 类型转换
    x_pre = x_pre.astype('float32')
    x_pre = x_pre / 255
    y_label = y_label.astype('float32')
    y_label = y_label / 255
    
    # 变成3维
    y_label = y_label[:, :, np.newaxis]
    
    return x_pre, y_label


def model_prediction(image_path, label_path, IMAGE_SIZE, model_list, pre_image_dir):
    '''
    1、加载模型
    2、读取图像
    3、预测
    4、保存
    ''' 
    # 存储评估结果
    result = {}
    result['data'] = []
    result['image'] = image_path
    result['IMAGE_SIZE'] = str(IMAGE_SIZE)
    
    # 加载图像并添加一维以适应模型的输入
    print('{}: Load predict image...'.format(datetime.now().strftime('%c')))
    x_pre, y_label = load_image(image_path, label_path, IMAGE_SIZE)
    
    # 变成4维, 以适应计算需求
    x_pre_4d = x_pre[np.newaxis, :, :, :]
    y_label_4d = y_label[np.newaxis, :, :, :]
    
    for i in range(len(model_list)):

        # 切割出模型的名字
        model_name = os.path.splitext(model_list[i].split("\\")[-1])[0] # for Window: '\\', Linux: '/'
        print('{}: =============================== {}: {} ==============================='.format(datetime.now().strftime('%c'), i, model_name))
    
        # 加载模型参数
        #model = load_model(save_model_path)
        model = model_unet.unet(IMAGE_SIZE = IMAGE_SIZE)
        model.load_weights(model_list[i])
        
        # 模型预测并调整越策结果
        y_pre = model.predict(x_pre_4d)
        y_pre = np.squeeze(y_pre, axis = 0)
        y_pre = np.round(y_pre)
               
        # 评估模型                
        scores = model.evaluate(x_pre_4d, y_label_4d, verbose=1)        
        print('{}: loss: {:.5f}'.format(datetime.now().strftime('%c'), scores[0]))
        print('{}: acc: {:.5f}'.format(datetime.now().strftime('%c'), scores[1]))        
        precision, recall, f1 = calculation(y_label, y_pre, IMAGE_SIZE)
        # 保存评估结果
        tmp = {}
        tmp = {"model_name": model_name,
               "Loss": str(round(scores[0], 5)),
               "Accuracy": str(round(scores[1], 5)),
               "Precision": str(round(precision, 5)),
               "Recall": str(round(recall, 5)),
               "F1": str(round(f1, 5))}
        
        result['data'].append(tmp)

        # 保存预测结果图
        y_pre *= 255
        pre_image_name = model_name + '.png' 
        pre_image_result = os.path.join(pre_image_dir, pre_image_name)  # 结果图
        cv2.imwrite(pre_image_result, y_pre)
        result['pre_image'] = pre_image_result
        print('{}: Save model:{} prefict result success!'.format(datetime.now().strftime('%c'), model_name))
        
    # 存储成json格式
    with open(os.path.join(pre_image_dir, 'evaluated_results.txt'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))
    

if __name__ == '__main__':
    '''
    主函数
    '''
    # 预测图像的大小
    IMAGE_SIZE = 1024
    
    # 获取路径下的所有模型，放在列表中
    model_list = []
    model_dir = os.path.join(os.getcwd(), 'Saved_models')
    model_glob = os.path.join(model_dir, '*.' + 'hdf5')  # 模型
    model_list.extend(glob.glob(model_glob))   
    
    print('{}: There will be {} models for predicting!'.format(datetime.now().strftime('%c'), len(model_list)))

    # 预测图像的原图和标签图的路径 
    pre_image_dir = os.path.join(os.getcwd(), 'pre_image')
    pre_image_path = os.path.join(pre_image_dir, 'data_test3.png')  # 原图
    pre_label_path = os.path.join(pre_image_dir, 'data_test_gt3.png')  # 标签图
    
    # 模型预测
    model_prediction(image_path=pre_image_path,
                     label_path=pre_label_path,
                     IMAGE_SIZE=IMAGE_SIZE, 
                     model_list=model_list,
                     pre_image_dir=pre_image_dir)
    








