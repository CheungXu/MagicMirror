#-*- coding:utf-8 -*-

import tensorflow as tf
import DataCooker as DC
from Network import Vgg
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    print('------------------------Prepareing Data...-----------------------------------')
    train_cooker = DC.DataCooker(batch_size=16)
    print('------------------------Prepareing Data Completed----------------------------')
    print('------------------------Constructing Network...------------------------------')
    model = Vgg(16,256,256)
    net = model.build_network()
    print('------------------------Network Done !---------------------------------------')
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True

    with tf.Session(config=sess_config) as sess:
        print('----------------------------Start Train ...----------------------------------')
        model.train(sess, train_cooker, epoch_num=1000)

#First Edition