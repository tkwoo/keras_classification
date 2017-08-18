from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse

import models

class predictor:
    def __init__(self, flag):
        self.flag = flag
    
    def predict(self, image_path=None):
        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print "[*] model load : %s"%weight_list[-1]
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print "[*] model loading Time: %.3f ms"%t_total

        # image path ex) './dataset/sagital/odd/ori/AD_154_103.png'
        if image_path == None:
            imgInput = cv2.imread(self.flag.test_image_path, 0)
        else:
            imgInput = cv2.imread(image_path, 0)
        input_data = imgInput.reshape((1,self.flag.image_size,self.flag.image_size,1))
        input_data /= 1./255

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print "Predict Time: %.3f ms"%t_total

        print result

        window_name = "show"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, imgInput)
    
    def evaluate(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        # model = model_from_json(loaded_model_json)
        model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print "[*] model load : %s"%weight_list[-1]
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print "[*] model loading Time: %.3f ms"%t_total

        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'validation'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle=False,
                color_mode='grayscale',
                class_mode='categorical')        
        t_start = cv2.getTickCount()
        loss, acc = model.evaluate_generator(test_generator, test_generator.n // self.flag.batch_size)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print '[*] test loss : %.4f'%loss
        print '[*] test acc  : %.4f'%acc
        print "[*] evaluation Time: %.3f ms"%t_total