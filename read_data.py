# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:56:30 2019

@author: zhangyonghui
"""

import numpy as np
import os
import scipy.io as scio
import glob
import cv2
from datetime import datetime
#import scipy.misc as misc


def create_image_lists(image_dir):
    '''
    image_list:
        training:
            filename, image, annocation
        validation:    
            filename, image, annocation
    '''
    
    directories = ['training', 'validation']
    image_list = {}
    
    num_of_images = []
    
    for directory in directories:
        
        file_list = []
        image_list[directory] = []
        
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))
        
        if not file_list:
            print('{}: No files found'.format(datetime.now().strftime('%c')))
        else:
            for f in file_list:
                
                filename = os.path.splitext(f.split("\\")[-1])[0] #for Windows: '\\', Linux: '/'
    
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                
                if os.path.exists(annotation_file):
                    record = {'filename': filename, 'image': f, 'annotation': annotation_file}
                    image_list[directory].append(record)
                else:
                    print('{}: Annotation file not found for {} - Skipping'.format(datetime.now().strftime('%c'), filename))
    
        no_of_image = len(image_list[directory])
        num_of_images.append(no_of_image)
        print('{}: No. of {} files: {:d}'.format(datetime.now().strftime('%c'), directory, no_of_image))
    
    return image_list, num_of_images


def read_image_creat_mat(image_dir, IMAGE_SIZE, mat_dir):

    # 将字典中的‘training’和‘validation’取出
    if (os.path.isfile(mat_dir)):
        print('{}: Loading the mat file of dataset...'.format(datetime.now().strftime('%c')))
        train_images = scio.loadmat(mat_dir)['train_images']
        train_labels = scio.loadmat(mat_dir)['train_labels']
        valid_images = scio.loadmat(mat_dir)['valid_images']
        valid_labels = scio.loadmat(mat_dir)['valid_labels']
        
    else:            
        print('{}: Reading the images of training and validation...'.format(datetime.now().strftime('%c')))
        image_lists, num_of_images = create_image_lists(image_dir)   
        train_records = image_lists['training']
        valid_records = image_lists['validation']
        
        train_images = []
        train_labels = []
        valid_images = []
        valid_labels = []
        
        for i in range(num_of_images[0]):
            '''
            training
            '''
            
            train_image = cv2.imread(train_records[i]['image'], -1)
#            train_image = misc.imresize(train_image, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
            train_images.append(np.array(train_image))
                              
            train_label = cv2.imread(train_records[i]['annotation'], -1)
#            train_label = misc.imresize(train_label, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
            train_labels.append(np.array(train_label))
            
          
        for i in range(num_of_images[1]):
            '''
            validation
            '''
            
            valid_image = cv2.imread(valid_records[i]['image'], -1)
#            valid_image = misc.imresize(valid_image, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
            valid_images.append(np.array(valid_image))
            
            valid_label = cv2.imread(valid_records[i]['annotation'], -1)
#            valid_label = misc.imresize(valid_label, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
            valid_labels.append(np.array(valid_label))
        
        # 将列表转换成数组    
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        valid_images = np.array(valid_images)
        valid_labels = np.array(valid_labels)
        
        print('{}: Reading success!'.format(datetime.now().strftime('%c')))

            
        # 存成mat文件
        print('{}: Creating the mat file of dataset...'.format(datetime.now().strftime('%c')))      
        scio.savemat(mat_dir, {'train_images': train_images, 'train_labels': train_labels,
                               'valid_images': valid_images, 'valid_labels': valid_labels})
    
        print('{}: Creating success!'.format(datetime.now().strftime('%c')))
    
    return train_images, train_labels, valid_images, valid_labels



#c = cv2.imread(r'G:\Mariculture_code\inceptionV3-keras\ADEChallengeData2016\annotations\validation\row42col15.png', -1)
#d = np.array(c)














