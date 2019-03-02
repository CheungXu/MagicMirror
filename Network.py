import tensorflow as tf
import numpy as numpy
import DataCooker2 as DC
import os,time

#批归一化
def batch_normal(input_ , scope="scope" , reuse=False, is_train = True):
    return tf.contrib.layers.batch_norm(input_ , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse , updates_collections=None, is_training = is_train)

#上采样
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape.as_list()
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor,1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor,s[3]])
    return x

#2维卷积层
def conv2d(input_, kernel_num, kernel_size=3, stride=1,pad='same', conv_name='conv2d'):
    return tf.layers.conv2d(input_,filters=kernel_num,kernel_size=[kernel_size, kernel_size],strides=stride, padding=pad,activation=tf.nn.relu,name=conv_name)

#2维反卷积层
def deconv2d(input_, kernel_num, kernel_size=3, stride=1,pad='same', conv_name='deconv2d'):
    return tf.layers.conv2d(upscale2d(input_),filters=kernel_num,kernel_size=[kernel_size, kernel_size],strides=stride, padding=pad,activation=tf.nn.relu,name=conv_name)

#最大池化
def max_pool2d(input_, kernel_size=3, pool_name='pool2d'):
    return tf.layers.max_pooling2d(input_, kernel_size, strides=2 ,padding='same',data_format='channels_last',name=pool_name)

#交叉熵
def sigmoid_cross_entropy_with_logits(x, y):
    #try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    #except:
     #   return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

#参差自编码网（完成中...）
"""
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
"""
#参差网（已完成，待更新...）
"""
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
"""

#VGG网络（已完成）
class Vgg(object):
    def __init__(self, batch_size = 1, image_height = 256, image_width = 256, datacooker=None, is_train=True, use_gpu=True):
        #基本参数
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.learning_rate = 0.01
        self.model_loaded = False
        self.graph_build = False
        self.ckpt = None
        self.sess = None
        #设备参数
        self.device_info = ''
        if use_gpu:
            self.device_info = '/gpu:0'
        else:
            self.device_info = '/cpu:0'
        #运行模式（训练/推断）
        self.is_train = is_train
        self.input = None
        self.label = None
        if self.is_train:
            self.input, self.label = datacooker.get_batch_data(batch_size = self.batch_size)
        else:
            self.input = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 3], name='input_images')
    
    def __enter__(self):
        return self
    
    def __exit__(self, batch_size = 1, image_height = 256, image_width = 256, datacooker=None, is_train=True, use_gpu=True):
        if self.sess is not None:
            self.sess.close()
            
    def build_network(self, reuse=False):
        with tf.device(self.device_info):
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
        self.graph_build = True
        return self.output

    def train(self, use_gpu = True):
        #判断是否可训练
        if not self.is_train:
            print('Not a Trainable Model !')
            return False
        
        #判读计算图是否建立
        if not self.graph_build:
            _ = self.build_network()
        
        #构建损失函数
        self.loss = tf.reduce_sum(tf.square(self.label - self.output))
        #self.loss = tf.reduce_sum(sigmoid_cross_entropy_with_logits(self.output, self.label))
        
        #获取所有训练参数
        self.train_vars = tf.trainable_variables()
        
        #设置Tensorboard记录
        self.image_sum = tf.summary.image('image',self.input)
        self.loss_sum = tf.summary.scalar('loss',self.loss)
        self.all_sum = tf.summary.merge([self.image_sum, self.loss_sum])
        
        #训练轮数
        self.global_step = tf.train.get_or_create_global_step()
        
        #学习率衰减
        new_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step = self.global_step, decay_steps=1000,decay_rate = 0.98)

        #设置依赖BN参数的优化器
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.trainer = tf.train.AdamOptimizer(learning_rate = new_learning_rate)
            #self.trainer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)  
        self.gradients = self.trainer.compute_gradients(self.loss, var_list=self.train_vars)
        self.training_op = self.trainer.apply_gradients(self.gradients, global_step = self.global_step)

        #模型保存器
        self.all_vars = tf.global_variables()
        bn_moving_vars = [g for g in self.all_vars if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in self.all_vars if 'moving_variance' in g.name]
        self.save_list = self.train_vars + bn_moving_vars
        self.saver = tf.train.Saver(var_list = self.save_list, max_to_keep=5)
        
        #设置会话配置
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        
        #模型保存挂件,定轮保存模型
        model_save_path = os.path.join('.','model','Vgg')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir = model_save_path, save_steps = 500, saver = self.saver)
        
        #日志记录挂件，记录日志
        summary_path = os.path.join('.','log')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_hook = tf.train.SummarySaverHook(save_steps = 5, output_dir = summary_path, summary_op = self.all_sum)

        loss_sum = 0.0
        loss_sum_num = 1
        print_num = 10
        
        #启动会话，开始训练
        self.sess = tf.train.MonitoredTrainingSession(hooks = [checkpoint_hook, summary_hook], config = sess_config)
        #参数初始化
        #tf.global_variables_initializer().run()
        #冻结计算图
        self.sess.graph.finalize()
        #开始训练
        while not self.sess.should_stop():
            #更新参数,记录loss
            with tf.device(self.device_info):
                _,summary_str, loss, global_step = self.sess.run([self.training_op, self.all_sum, self.loss, self.global_step])
                loss_sum += loss
            #输出信息
            if global_step % print_num == 0:
                learning_rate = self.sess.run(new_learning_rate)
                print('Batch %d Loss: %.7f LR:%.7f' % (global_step, loss_sum/loss_sum_num, learning_rate))
                loss_sum = 0
                loss_sum_num = 0
            else:
                loss_sum_num += 1
        return True
    
    def load(self, model_path = os.path.join('.','model','Vgg')):
        #载入模型文件状态
        self.ckpt = tf.train.get_checkpoint_state(model_path)
        
        #判断模型文件是否载入成功
        if self.ckpt is None:
            print('Model File Not Found !')
            return False
        
        if self.graph_build:
            #已建立图，则使用当前图，只载入参数
            saver = tf.train.Saver()
        else:
            #未建立图，则同时载入图和参数
            saver = tf.train.import_meta_graph(self.ckpt.model_checkpoint_path +'.meta')
            
        #建立会话
        if self.sess is None:
            self.sess = tf.Session()
        #载入模型
        saver.restore(self.sess,self.ckpt.model_checkpoint_path)
        self.model_loaded = True
             
    def inference(self, input_):
        #判读是否可推断
        if self.is_train:
            print('Not a Model for Inference !')
            return False
        
        #判读是否已载入模型
        if not self.model_loaded:
            print('No Model Loaded !')
            return False
        
        #推断
        with tf.device(self.device_info):
            output = self.sess.run(self.output, feed_dict={self.input: input_})
            
        return output
        

if __name__=='__main__':
    net = Vgg(16,256,256)
    res = net.build_network()
    sess = 0
    data = 0
    #net.train(sess,data,1)     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    