import numpy as np
import cv2,random
import os

class DataCooker(object):
    def __init__(self, image_size=256, batch_size = 1):
        #Image&Label Path
        self.image_path = os.path.join('.','data','images') 
        self.label_path = os.path.join('.','data','labels.txt')
        #Read Labels
        with open(self.label_path,'r') as f:
            lines = f.readlines()
        #Create label dictronary
        self.label_dict = {}
        if len(lines) <= 0:
            print('Error: Do not have any data！')
        else:
            for line in lines:
                strs = line.strip().split()
                self.label_dict[strs[0]] = (float(strs[1]) - 1.)/2 - 1.
        #Get Data Image List
        self.image_list = list(self.label_dict.keys())
        #Set Batch Size Options
        if len(self.image_list)%batch_size != 0:
            print('Warning: Data Length(', len(self.image_list), ') is not divisible by Batch Size',batch_size, '!')
        self.batch_size = batch_size
        self.batch_num = int(len(self.image_list)/self.batch_size)
        self.get_index = 0
        self.image_data = {}
        #Read Image Data
        num = 0
        for image_name in self.image_list:
            if num % 1000 == 0:
                print('%d/%d' % (num, len(self.image_list)))
            num += 1
            img = cv2.imread(os.path.join(self.image_path,image_name), 1)
            img = cv2.resize(img,(image_size,image_size), interpolation=cv2.INTER_CUBIC)
            img = img[:,:, (2, 1, 0)]
            self.image_data[image_name] = img.astype(np.float)/127.5 - 1.

    def __len__(self):
        return self.batch_num

    def shuffle(self):
        #Shuffle Dataset
        if self.get_index == 0:
            random.shuffle(self.image_list)
        else:
            print('Cannot Shuffle Dataset during Get Batch !')
    
    def get_batch(self, batch_idx = '-1'):
        #Get Batch Image Data By Index or Sequence
        if batch_idx < 0 or batch_idx >= self.batch_num:
            batch_list = self.image_list[self.get_index * self.batch_size:(self.get_index+1) * self.batch_size]
            self.get_index += 1
            if self.get_index >= self.batch_num:
                self.get_index = 0
        else:
            batch_list = self.image_list[batch_idx * self.batch_size:(batch_idx+1) * self.batch_size]

        image_batch = []
        label_batch = []
        for image_name in batch_list:
            image_batch.append(self.image_data[image_name][np.newaxis,:])
            label_batch.append(self.label_dict[image_name])
        return np.concatenate(image_batch,axis=0), np.array(label_batch)[:,np.newaxis]

    def show_label_dict(self, key='all'):
        #Show Label Dictronary
        if key == 'all':
            return self.label_dict
        else:
            for k in self.label_dict.keys():
                if key in k:
                    return self.label_dict[k]
        return False
        
    def show_image_names(self, num=-1):
        #Show Image List
        if num < 0 or num >= len(self.image_list):
            return self.image_list
        else:
            return self.image_list[num]
