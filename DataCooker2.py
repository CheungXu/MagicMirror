import numpy as np
import os,cv2
import tensorflow as tf

#归一化数据范围到[-1, 1]
def normalized(data):
    mx = max(data)
    mn = min(data)
    return [(2 *(float(i) - mn) / (mx - mn)) - 1 for i in data]

class DataCooker(object):
    def __init__(self, image_path = os.path.join('.','data'), label_path = os.path.join('.','data','label.txt'), image_size = 256, epoch = 1):
        #图像、标签路径
        self.image_path = image_path
        self.label_path = label_path
        self.epoch = epoch
        self.image_size = image_size
        #读取标签
        with open(self.label_path,'r') as f:
            lines = f.readlines()
        #记录标签对
        image_name_list = []
        label_list = []
        if len(lines) <= 0:
            print('Error: Do not have label data！')
        else:
            for line in lines:
                strs = line.strip().split()
                image_name_list.append(os.path.join(self.image_path,strs[0]))
                label_list.append(float(strs[1]))
        self.image_lists = np.array(image_name_list)
        self.labels = np.array(normalized(label_list))
        #读取函数
        def __read_image_dataset(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
            image_normalized = tf.to_float(image_resized, name='ToFloat')/127.5 - 1.0
            image_normalized = tf.reshape(image_normalized, [image_size, image_size, 3])
            return image_normalized, tf.to_float(label, name='ToFloat')
        #读取图片    
        filenames = tf.constant(self.image_lists)
        labels= tf.constant(self.labels)
        self.dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        self.dataset = self.dataset.map(__read_image_dataset)
        #设置epoch
        if self.epoch > 1:
            self.dataset = self.dataset.repeat(self.epoch)
        elif self.epoch == -1:
            self.dataset = self.dataset.repeat()
            
    def get_batch_data(self, batch_size = 1):
        self.batch_size = batch_size
        #batch未整除警告
        if len(self.image_lists)%batch_size != 0:
            print('Warning: Data Length(', len(self.image_lists), ') is not divisible by Batch Size',batch_size, '!')
        #设置batch
        self.batch_dataset = self.dataset.batch(batch_size, drop_remainder=True)
        #创建迭代器
        self.iterator = self.batch_dataset.make_one_shot_iterator()
        sample, label = self.iterator.get_next()
        return tf.reshape(sample, [batch_size, self.image_size, self.image_size, 3]), tf.reshape(label, [batch_size, 1])
    
    def get_dataset():
        return self.dataset
    
    def shuffle(buffer_size=10000):
        self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
        return True

if __name__ == '__main__':
    dc = DataCooker(os.path.join('/','data','Images'), os.path.join('/','data','labels.txt'))
    #print(dc.labels[0])
    img_iter, label_iter = dc.get_batch_data(batch_size = 16)
    print(img_iter.shape)
    print(label_iter.shape)

    with tf.Session() as sess:
        res = sess.run(label_iter)
    for label in res:
        print(label*1000)