#-*- coding:utf-8 -*-

import tensorflow as tf
from DataCooker2 import DataCooker
from Network import Vgg
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    print('------------------------Prepareing Data...-----------------------------------')
    dc = DataCooker(os.path.join('/','data','Images'), os.path.join('/','data','labels.txt'), epoch = 20)
    print('------------------------Prepareing Data Completed----------------------------')
    print('------------------------Constructing Network...------------------------------')
    model = Vgg(16,256,256)
    net = model.build_network(dc)
    print('------------------------Network Done !---------------------------------------')

    print('----------------------------Start Train ...----------------------------------')
    model.train()