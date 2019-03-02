import os,cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import tensorflow as tf
from Network import Vgg

def read_image(path):
    #print(path)
    img = cv2.imread(path,1)
    img = cv2.resize(img,(256,256), interpolation=cv2.INTER_CUBIC)
    img_net = img[:,:, (2, 1, 0)]
    img_net = img_net.astype(np.float)/127.5 - 1.
    img_net = img_net[np.newaxis, :]
    return img, img_net

def inference_once(path):
    img, img_for_net = read_image(path)
    with Vgg(image_height=256,image_width=256,is_train=False, use_gpu=True) as model:
        model.build_network()
        model.load()
        output = model.inference(img_for_net)
        score = 5 * (output[0][0] + 1.) / 2.
        print(score)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def inference_batch(path):
    img_list = os.listdir(path)
    with Vgg(batch_size=1,image_height=256,image_width=256,is_train=False, use_gpu=True) as model:
        model.build_network()
        model.load()
        for filename in img_list:
            img, img_for_net = read_image(os.path.join(path,filename))
            output = model.inference(img_for_net)
            score = 5 * (output[0][0] + 1.) / 2.
            print(score)
            cv2.imshow('img',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                                          
def inference_batch2(path):
    batch_size = 50
    img_list = os.listdir(path)
    imgs = []
    for i in range(batch_size):
        img, img_for_net = read_image(os.path.join(path,img_list[i]))
        imgs.append(img_for_net)
    imgs_for_net = np.vstack(imgs)
    print(imgs_for_net.shape)
    with Vgg(batch_size=batch_size,image_height=256,image_width=256,is_train=False, use_gpu=True) as model: 
        model.build_network()
        model.load()
        output = model.inference(imgs_for_net)
        print(output.shape)
        for i in range(len(output)):
            s = 5 * (output[i][0] + 1.) / 2.
            print(img_list[i],output[i][0],s)
            img, _ = read_image(os.path.join(path,img_list[i]))
            cv2.imshow('img',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    #Inference Once
    #img_path = os.path.join('/','data','Images','AM1384.jpg')
    #inference_once(img_path)
    
    #Inference Batch
    #img_path = os.path.join('/','data','Images')
    #inference_batch(img_path)
    
    #Inference Batch
    img_path = os.path.join('/','data','test')
    inference_batch2(img_path)    