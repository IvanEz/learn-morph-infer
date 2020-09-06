from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config

        self.batch_manager = batch_manager
        self.x, self.y, self.geom, self.x_val, self.y_val, self.geom_val = batch_manager.batch() # normalized input
        #self.x_val, self.y_val, self.geom_val = batch_manager.batch_val()  # normalized input

        self.is_3d = config.is_3d
        self.dataset = config.dataset
        self.data_type = config.data_type
        self.arch = config.arch
        
        if 'nn' in self.arch:
            self.xt, self.yt = batch_manager.test_batch()
            self.xtw, self.ytw = batch_manager.test_batch(is_window=True)
            self.xw, self.yw = batch_manager.batch(is_window=True)            
        else:
            if self.is_3d:
                #self.x_jaco, self.x_vort = jacobian3(self.x)
                self.x_jaco = jacobian3tumor(self.x)
                self.x_jaco_val = jacobian3tumor(self.x_val)

            else:
                self.x_jaco, self.x_vort = jacobian(self.x)

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.c_num = batch_manager.c_num
        self.b_num = config.batch_size
        self.test_b_num = config.test_batch_size
        self.val_set_size = batch_manager.num_samples_validation

        self.repeat = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.w1 = config.w1
        self.w2 = config.w2
        if 'dg' in self.arch: self.w3 = config.w3

        self.use_c = config.use_curl
        if self.use_c:
            if self.is_3d:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [3]
            else:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [1]
        else:
            self.output_shape = get_conv_shape(self.x)[1:]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
                
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = config.start_step
        self.step = tf.Variable(self.start_step, name='step', trainable=False)
        # self.max_step = config.max_step
        self.max_step = int(config.max_epoch // batch_manager.epochs_per_step)

        self.lr_update = config.lr_update
        if self.lr_update == 'decay':
            lr_min = config.lr_min
            lr_max = config.lr_max
            self.g_lr = tf.Variable(lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, 
               lr_min+0.5*(lr_max-lr_min)*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_step)+1), name='g_lr_update')
        elif self.lr_update == 'step':
            self.g_lr = tf.Variable(config.lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_min), name='g_lr_update')    
        else:
            raise Exception("[!] Invalid lr update method")

        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.save_sec = config.save_sec

        self.is_train = config.is_train
        if 'ae' in self.arch:
            self.z_num = config.z_num
            self.p_num = self.batch_manager.dof
            self.use_sparse = config.use_sparse
            self.sparsity = config.sparsity
            self.w4 = config.w4
            self.w5 = config.w5
            self.code_path = config.code_path
            self.build_model_ae()

        elif 'nn' in self.arch:
            self.z_num = config.z_num
            self.w_num = config.w_size
            self.p_num = self.batch_manager.dof
            self.build_model_nn()

        else:
            self.build_model()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.save_sec,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if 'nn' in self.arch:
            self.batch_manager.init_it(self.sess)
            self.log_step = batch_manager.train_steps
        
        elif self.is_train:
            self.batch_manager.start_thread(self.sess)
            print('debug.Threads started')

        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False


    def get_vort_image(self, x):
        x = vort_np(x[:,:,:,:2])
        if not 'ae' in self.arch: x /= np.abs(x).max() # [-1,1]
        x_img = (x+1)*127.5
        x_img = np.uint8(plt.cm.RdBu(x_img[...,0]/255)*255)[...,:3]
        return x_img

