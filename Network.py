import tensorflow as tf
import numpy as numpy
import DataCooker as DC
import os,time

def batch_normal(input_ , scope="scope" , reuse=False):
    return tf.contrib.layers.batch_norm(input_ , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse , updates_collections=None)

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape.as_list()
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor,1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor,s[3]])
    return x

def conv2d(input_, kernel_num, kernel_size=3, stride=1,pad='same', conv_name='conv2d'):
    return tf.layers.conv2d(input_,filters=kernel_num,kernel_size=[kernel_size, kernel_size],strides=stride, padding=pad,activation=tf.nn.relu,name=conv_name)

def deconv2d(input_, kernel_num, kernel_size=3, stride=1,pad='same', conv_name='deconv2d'):
    return tf.layers.conv2d(upscale2d(input_),filters=kernel_num,kernel_size=[kernel_size, kernel_size],strides=stride, padding=pad,activation=tf.nn.relu,name=conv_name)

def max_pool2d(input_, kernel_size=3, pool_name='pool2d'):
    return tf.layers.max_pooling2d(input_, kernel_size, strides=2 ,padding='same',data_format='channels_last',name=pool_name)

def sigmoid_cross_entropy_with_logits(x, y):
    #try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    #except:
     #   return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

class resVae(object):
    def __init__(self,batch_size,image_height,image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.learning_rate = 0.05
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 3] , name='input_images')
        self.label = tf.placeholder(tf.float32, [self.batch_size,1], name='input_labels')

    def build_network(self,reuse=False):
        with tf.device('/gpu:0'):
            with tf.variable_scope('resVae') as scope:
                if reuse:
                    scope.reuse_variables()  
                h0 = batch_normal(conv2d(self.input, 16, 5, conv_name='resVae_conv_0'), scope='resVae_bn_0') #256->256
                h1 = batch_normal(conv2d(h0, 32, 3, conv_name='resVae_conv_1'), scope='resVae_bn_1') #256->256
                h2 = max_pool2d(h1, pool_name='resVae_maxpool_2')

                h3 = batch_normal(conv2d(h2, 16, 3, 2, conv_name='resVae_conv_3'), scope='resVae_bn_3') #128->64
                h4 = batch_normal(conv2d(h3, 16, 3, 2, conv_name='resVae_conv_3'), scope='resVae_bn_3') #64->32
                h5 = batch_normal(deconv2d(h4, 16, 3, 2, conv_name='resVae_conv_3'), scope='resVae_bn_3') #32->64


class ResNet(object):
    def __init__(self,batch_size,image_height,image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.learning_rate = 0.05
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 3] , name='input_images')
        self.label = tf.placeholder(tf.float32, [self.batch_size,1], name='input_labels')
    
    def network(self,reuse=False):
        with tf.device('/gpu:0'):
            with tf.variable_scope('Res') as scope:
                if reuse:
                    scope.reuse_variables()
                h0 = batch_normal(conv2d(self.input, 16, 5, conv_name='Res_conv_0'), scope='Res_bn_0') #256


    def train(self, sess, datacooker,epoch_num=1):
        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=global_step, decay_steps=1000,decay_rate=0.98)

        #self.loss = tf.reduce_sum(tf.square(self.label - self.output))
        self.loss = tf.reduce_sum(sigmoid_cross_entropy_with_logits(self.output, self.label))

        self.image_sum = tf.summary.image('image',self.input)
        self.loss_sum = tf.summary.scalar('loss',self.loss)
        self.vars = tf.trainable_variables()

        #self.trainer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        self.trainer = tf.train.AdamOptimizer(learning_rate = new_learning_rate)
        self.gradients = self.trainer.compute_gradients(self.loss, var_list=self.vars)
        self.optionor = self.trainer.apply_gradients(self.gradients)

        tf.global_variables_initializer().run()
        self.all_sum = tf.summary.merge([self.image_sum, self.loss_sum])
        self.writer = tf.summary.FileWriter(".\\logs", sess.graph)

        self.saver = tf.train.Saver(max_to_keep=0)

        index_num = len(datacooker)
        start_time = time.time()
        for epoch in range(epoch_num):
            for idx in range(index_num):
                images, labels = datacooker.get_batch(idx)
                with tf.device('/gpu:0'):
                    _, summary_str = sess.run([self.optionor, self.all_sum], feed_dict={self.input : images, self.label : labels})
                self.writer.add_summary(summary_str,epoch*index_num+idx)
                learning_rate = sess.run(new_learning_rate)
                if learning_rate > 0.001:
                    sess.run(add_global)
                if (epoch*index_num+idx) % 10 == 0:   
                    loss = 0.0
                    loss = sess.run(self.loss, feed_dict= {self.input : images, self.label : labels})
                    end_time = time.time()
                    print('Epoch: %d/%d Batch: %d/%d Loss: %.7f Time: %.2fs LR: %.7f' % (epoch, epoch_num, idx, index_num, loss, (end_time-start_time), learning_rate))
                    start_time = end_time
            if epoch % 10 == 0:
                if not os.path.exists('.\\model\\Vgg'):
                    os.makedirs('.\\model\\Vgg')
                checkpoint_path = os.path.join('.\\model\\Vgg', 'vgg_%d' % (epoch))
                self.saver.save(sess, checkpoint_path, global_step=epoch)
                print('Save checkpoint in : %s' % (checkpoint_path))   



class Vgg(object):
    def __init__(self,batch_size,image_height,image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.learning_rate = 0.05
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 3] , name='input_images')
        self.label = tf.placeholder(tf.float32, [self.batch_size,1], name='input_labels')

    def build_network(self,reuse=False):
        with tf.device('/gpu:0'):
            with tf.variable_scope('Vgg') as scope:
                if reuse:
                    scope.reuse_variables() 
                h0 = batch_normal(conv2d(self.input, 16, 5, conv_name='Vgg_conv_0'), scope='Vgg_bn_0') #256
                h1 = batch_normal(conv2d(h0, 16, 3, conv_name='Vgg_conv_1'), scope='Vgg_bn_1') #256
                h2 = max_pool2d(h1, pool_name='Vgg_maxpool_2')
                h3 = batch_normal(conv2d(h2, 32, 3, conv_name='Vgg_conv_3'), scope='Vgg_bn_3') #128
                h4 = batch_normal(conv2d(h3, 32, 3, conv_name='Vgg_conv_4'), scope='Vgg_bn_4') #128
                h5 = max_pool2d(h4, pool_name='Vgg_maxpool_4')
                h6 = batch_normal(conv2d(h5, 64, 3, conv_name='Vgg_conv_6'), scope='Vgg_bn_6') #64
                h7 = batch_normal(conv2d(h6, 64, 3, conv_name='Vgg_conv_7'), scope='Vgg_bn_7') #64
                h8 = max_pool2d(h7, pool_name='Vgg_maxpool_8')
                h9 = batch_normal(conv2d(h8, 64, 3, conv_name='Vgg_conv_9'), scope='Vgg_bn_9') #32
                h10 = max_pool2d(h9, pool_name='Vgg_maxpool_10')
                h11 = batch_normal(conv2d(h10, 128, 3, conv_name='Vgg_conv_11'), scope='Vgg_bn_11') #16
                h12 = max_pool2d(h11, pool_name='Vgg_maxpool_12')
                h13 = batch_normal(conv2d(h12, 128, 3, conv_name='Vgg_conv_13'), scope='Vgg_bn_13') #8
                h14 = max_pool2d(h13, pool_name='Vgg_maxpool_14') 
                h15 = batch_normal(conv2d(h14, 128, 4, pad='valid', conv_name='Vgg_conv_15'), scope='Vgg_bn_15') #4
                h16 = tf.layers.dense(tf.reshape(h15,[self.batch_size,-1]),1,name='Vgg_liner_16')
                self.output = tf.nn.tanh(h16)
        return self.output

    def train(self, sess, datacooker,epoch_num=1):
        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=global_step, decay_steps=1000,decay_rate=0.98)

        #self.loss = tf.reduce_sum(tf.square(self.label - self.output))
        self.loss = tf.reduce_sum(sigmoid_cross_entropy_with_logits(self.output, self.label))

        self.image_sum = tf.summary.image('image',self.input)
        self.loss_sum = tf.summary.scalar('loss',self.loss)
        self.vars = tf.trainable_variables()

        #self.trainer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        self.trainer = tf.train.AdamOptimizer(learning_rate = new_learning_rate)
        self.gradients = self.trainer.compute_gradients(self.loss, var_list=self.vars)
        self.optionor = self.trainer.apply_gradients(self.gradients)

        tf.global_variables_initializer().run()
        self.all_sum = tf.summary.merge([self.image_sum, self.loss_sum])
        self.writer = tf.summary.FileWriter(".\\logs", sess.graph)

        self.saver = tf.train.Saver(max_to_keep=0)

        index_num = len(datacooker)
        start_time = time.time()
        for epoch in range(epoch_num):
            for idx in range(index_num):
                images, labels = datacooker.get_batch(idx)
                with tf.device('/gpu:0'):
                    _, summary_str = sess.run([self.optionor, self.all_sum], feed_dict={self.input : images, self.label : labels})
                self.writer.add_summary(summary_str,epoch*index_num+idx)
                learning_rate = sess.run(new_learning_rate)
                if learning_rate > 0.001:
                    sess.run(add_global)
                if (epoch*index_num+idx) % 10 == 0:   
                    loss = 0.0
                    loss = sess.run(self.loss, feed_dict= {self.input : images, self.label : labels})
                    end_time = time.time()
                    print('Epoch: %d/%d Batch: %d/%d Loss: %.7f Time: %.2fs LR: %.7f' % (epoch, epoch_num, idx, index_num, loss, (end_time-start_time), learning_rate))
                    start_time = end_time
            if epoch % 10 == 0:
                if not os.path.exists('.\\model\\Vgg'):
                    os.makedirs('.\\model\\Vgg')
                checkpoint_path = os.path.join('.\\model\\Vgg', 'vgg_%d' % (epoch))
                self.saver.save(sess, checkpoint_path, global_step=epoch)
                print('Save checkpoint in : %s' % (checkpoint_path))

if __name__=='__main__':
    net = Vgg(16,256,256)
    res = net.build_network()
    sess = 0
    data = 0
    #net.train(sess,data,1)     