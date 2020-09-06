from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *
from trainer import Trainer

class Trainer3(Trainer):
    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape,
                                               num_conv=self.num_conv, repeat=self.repeat)
            _, self.G_ = jacobian3(self.G_s)
        else:

            self.G_, self.G_var = TumorGenerator(self.geom, self.y, self.filters, self.output_shape,
                                                 num_conv=self.num_conv, repeat=self.repeat, arch=self.arch,
                                                 name='tumor', reuse=False)

        self.G = denorm_img3(self.G_) # for debug
        

        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
            g_optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer
            g_optimizer = optimizer(self.g_lr)
        else:
            raise Exception("[!] Invalid opimizer")

        # losses

        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.G_[self.x>=0.001] - self.x[self.x>=0.001]), name='g_loss_l1')
        self.g_csf_loss_l1 = tf.reduce_mean(tf.abs(self.G_[self.geom[...,2]>=0.001] - self.x[self.geom[...,2]>=0.001]), name='g_csf_loss_l1')

        self.g_loss = self.g_loss_l1 + self.g_csf_loss_l1

        if 'dg' in self.arch:
            self.g_loss_real = tf.reduce_mean(tf.square(self.D_G-1))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.D_G))
            self.d_loss_real = tf.reduce_mean(tf.square(self.D_x-1))

            self.g_loss += self.g_loss_real*self.w3

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.d_optim = g_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)
        #self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if 'dg' in self.arch:
            summary += [
                tf.summary.scalar("loss/g_loss_real", tf.sqrt(self.g_loss_real)),
                tf.summary.scalar("loss/d_loss_real", tf.sqrt(self.d_loss_real)),
                tf.summary.scalar("loss/d_loss_fake", tf.sqrt(self.d_loss_fake)),
            ]

        self.writer_val = tf.summary.FileWriter(self.model_dir + "/val")
        self.summary_op = tf.summary.merge_all()

        # summary once
        #x = denorm_img3(self.x)
        #x_vort = denorm_img3(self.x_vort)
        
        #summary = [
        #    tf.summary.image("xym/x", x['xym'][:,::-1]),
        #    tf.summary.image("zym/x", x['zym'][:,::-1]),
            #tf.summary.image("xym/vort", x_vort['xym'][:,::-1]),
            #tf.summary.image("zym/vort", x_vort['zym'][:,::-1]),
        #]
        #self.summary_once = tf.summary.merge(summary) # call just once

    def train(self):
        if 'ae' in self.arch:
            self.train_ae()
        elif 'nn' in self.arch:
            self.train_nn()
        else:
            self.train_()

    def train_(self):
        # test1: varying on each axis
        z_range = [0, 1]
        z_shape = (self.b_num, self.c_num)
        z_samples = []
        z_varying = np.linspace(z_range[0], z_range[1], num=self.b_num)

        # exit
        time_space = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,
                       0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0]

        #time_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        zi = np.zeros(shape=z_shape)
        zi[:, 2] = time_space[-self.b_num:]
        z_samples.append(zi)

        # test2: compare to gt

        gen_list = self.batch_manager.random_list(self.b_num, z_samples)

        x_xy = gen_list['xym']
        x_zy = gen_list['zym']
        save_image(x_xy, '{}/x_fixed_xym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        save_image(x_zy, '{}/x_fixed_zym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        save_image(gen_list['xym_val'], '{}/x_val_xym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        save_image(gen_list['zym_val'], '{}/x_val_zym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)

        #for item in len(self.b_num):
        path_to_gt = self.batch_manager.list_from_p(gen_list['p'])
        print('debug,here')
        print(path_to_gt)

        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(gen_list['p']) + '\n')
            f.write(str(gen_list['z']))
            f.write('\n')
            f.write(str(gen_list['p_val']) + '\n')
            f.write(str(gen_list['z_val']))
        print('debug,here2')
        #print("here1",np.asarray(z_samples).shape)
        zi = np.zeros(shape=z_shape)
        for i, z_gt in enumerate(gen_list['z']):
            zi[i,:] = z_gt
        z_samples.append(zi)
        #print("here2")
        z_samples.append(np.asarray(gen_list['z_val']))
        #print("here3", z_samples[1:])

        # call once
        print('debughere2.5')
        summary_once = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_once, 0)
        self.summary_writer.flush()
        print('debug,here3')

        summary_once = self.sess.run(self.summary_op, {
           self.geom: self.geom_val.eval(session=self.sess),
           })
        self.writer_val.add_summary(summary_once, 0)
        self.writer_val.flush()
        # train
        for step in trange(self.start_step, self.max_step):
            if 'dg' in self.arch:
                self.sess.run([self.g_optim, self.d_optim])
            else:
                self.sess.run(self.g_optim)

            if step % self.log_step == 0 or step == self.max_step-1:
                ep = step*self.batch_manager.epochs_per_step
                loss, summary = self.sess.run([self.g_loss,self.summary_op]) \
                     #, feed_dict={self.epoch: ep})

                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                print("\n[{}/{}/ep{:.2f}] Loss: {:.6f}".format(step, self.max_step, ep, loss))

                self.summary_writer.add_summary(summary, global_step=step)
                self.summary_writer.flush()

                summary_once = self.sess.run(self.summary_op, {
                   self.geom: self.geom_val.eval(session=self.sess)})
                self.writer_val.add_summary(summary_once, global_step=step)
                self.writer_val.flush()

            if step % self.test_step == 0 or step == self.max_step-1:

                self.generate(z_samples[1:], gen_list, self.model_dir, idx=step)

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run(self.g_lr_update)
            else:
                self.sess.run(self.g_lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

#    def build_test_model(self):
#        # build a model for testing
#        self.z = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.c_num])
#        if self.use_c:
#            self.G_s, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
#                                      num_conv=self.num_conv, repeat=self.repeat, reuse=True)
#            self.G_ = curl(self.G_s)
#        else:
#            self.G_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
#                                     num_conv=self.num_conv, repeat=self.repeat, reuse=True)        
    
    def generate(self, inputs, gt_list, root_path=None, idx=None):


        xym, zym = self.sess.run([self.G['xym'], self.G['zym']], {self.geom:gt_list["geom"],
                                                                  self.y: inputs[0]})
        xym_val, zym_val = self.sess.run([self.G['xym'], self.G['zym']], {self.geom: gt_list["geom_val"],
                                                                  self.y: inputs[1]})


        gen_random = np.asarray(tuple(xym))
        #print(np.unique(gen_random))
        #maxim = (gen_random-gen_random.min())/(gen_random.max()-gen_random.min())
        x_xy_path = os.path.join(root_path, 'x_fixed_xym_{}.png'.format(idx))
        save_image(gen_random, x_xy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_xy_path))

        #gen_random = np.concatenate(tuple(zym_list[-2:]), axis=0)
        gen_random = np.asarray(tuple(zym))
        #maxim = (gen_random-gen_random.min())/(gen_random.max()-gen_random.min())
        x_zy_path = os.path.join(root_path, 'x_fixed_zym_{}.png'.format(idx))
        save_image(gen_random, x_zy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_zy_path))

        gen_random = np.asarray(tuple(xym_val))
        # print(np.unique(gen_random))
        #maxim = (gen_random - gen_random.min()) / (gen_random.max() - gen_random.min())
        x_xy_path = os.path.join(root_path, 'x_val_xym_{}.png'.format(idx))
        save_image(gen_random, x_xy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_xy_path))

        # gen_random = np.concatenate(tuple(zym_list[-2:]), axis=0)
        gen_random = np.asarray(tuple(zym_val))
        #maxim = (gen_random - gen_random.min()) / (gen_random.max() - gen_random.min())
        x_zy_path = os.path.join(root_path, 'x_val_zym_{}.png'.format(idx))
        save_image(gen_random, x_zy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_zy_path))
        

    def build_test_model(self):
        self.z = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.c_num])
        self.geom_z = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.res_z, self.res_y, self.res_x, 3])

        self.Gz_, _ = TumorGenerator(self.geom_z, self.z, self.filters, self.output_shape,
                                                 num_conv=self.num_conv, repeat=self.repeat, arch=self.arch,
                                                 name='tumor', reuse=tf.AUTO_REUSE)

    def test(self):

        self.f = 1.0
        self.x_range = self.f
        self.y_range = [[0.0003, 0.0009], [0.0051, 0.0299], [0.0, 20.0]]

        self.build_test_model()

        # for placeholder based implementation
        paths = self.config.data_dir + "v/"
        # start = time.time()
        y_list = []
        geom_list = []
        filenames = []
        for k, path in enumerate(os.listdir(paths)):
            x_, y_, geom_ = self._preprocess(paths + path, "v", self.x_range, self.y_range)
            Gz_ = self.sess.run(self.Gz_, {self.z: np.expand_dims(y_, 0), self.geom_z: np.expand_dims(geom_, 0)})
            Gz_, _ = self.batch_manager.denorm(x=Gz_)

            np.savez_compressed(self.config.inf_save + path, x=Gz_)

    def _preprocess(self, file_path, data_type, x_range, y_range):
        with np.load(file_path) as data:
            y = data['y']
            x = np.expand_dims(data['x'][..., 0][int(round(y[3] * 128)) - 32:int(round(y[3] * 128)) + 32,
                               int(round(y[4] * 128)) - 32:int(round(y[4] * 128)) + 32,
                               int(round(y[5] * 128)) - 32:int(round(y[5] * 128)) + 32], axis=3)
            geom = data['x'][..., 1:][int(round(y[3] * 128)) - 32:int(round(y[3] * 128)) + 32,
                   int(round(y[4] * 128)) - 32:int(round(y[4] * 128)) + 32,
                   int(round(y[5] * 128)) - 32:int(round(y[5] * 128)) + 32]
            y = y[:3]

            # binarize segmentation
            # geom[..., 0][geom[..., 0] >= 0.5] = 1
            # geom[..., 0][geom[..., 0] < 0.5] = 0
            # geom[..., 1][geom[..., 1] > 0.5] = 1
            # geom[..., 1][geom[..., 1] <= 0.5] = 0

        # normalize
        if data_type[0] == 'd':
            x = x * 2 - 1
        else:
            x /= x_range
        # print("preprocess_range", y_range, y)
        for i, ri in enumerate(y_range):
            y[i] = (y[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
            # y[i] = y[i]/ri[1]
        # print("processed", y)
        return x, [round(elem, 2) for elem in y], geom