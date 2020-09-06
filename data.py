import os
from glob import glob

import threading
import multiprocessing
import signal
import sys
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ops import *

class BatchManager(object):
    def __init__(self, config):
        self.rng = np.random.RandomState(config.random_seed)
        self.root = config.data_path        
        self.root_val = config.valid_dataset_dir
        # read data generation arguments
        self.args = {}
        with open(os.path.join(self.root, 'args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        self.is_3d = config.is_3d
        if 'ae' in config.arch:
            def sortf(x):
                nf = int(self.args['num_frames'])
                n = os.path.basename(x)[:-4].split('_')
                return int(n[0])*nf + int(n[1])

            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])),
                                key=sortf)
            # num_path = len(self.paths)          
            # num_train = int(num_path*0.95)
            # self.test_paths = self.paths[num_train:]
            # self.paths = self.paths[:num_train]
        else:
            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])))
            self.valid_paths = glob("{}/*".format(self.root_val))

        self.num_samples = len(self.paths)
        self.num_samples_validation = len(self.valid_paths)

        assert(self.num_samples > 0)
        self.batch_size = config.batch_size
        self.epochs_per_step = self.batch_size / float(self.num_samples) # per epoch

        self.data_type = config.data_type
        #if self.data_type == 'velocity':
        #    if self.is_3d: depth = 3
        #    else: depth = 2
        #else:
        depth = 1
        
        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.depth = depth
        self.c_num = int(self.args['num_param'])

        if self.is_3d:
            feature_dim = [self.res_z, self.res_y, self.res_x, self.depth]
            geom_dim = [self.res_z, self.res_y, self.res_x, 3]
        else:
            feature_dim = [self.res_y, self.res_x, self.depth]
        
        if 'ae' in config.arch:
            self.dof = int(self.args['num_dof'])
            label_dim = [self.dof, int(self.args['num_frames'])]
        else:
            label_dim = [self.c_num]

        if self.is_3d:
            min_after_dequeue = 500 #################
        else:
            min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * self.batch_size
        #self.q = tf.FIFOQueue(capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.q = tf.FIFOQueue(capacity, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32], [feature_dim, label_dim, geom_dim, feature_dim, label_dim, geom_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        #print("here               ",feature_dim, label_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.geom = tf.placeholder(dtype=tf.float32, shape=geom_dim)

        # self.q_val = tf.FIFOQueue(capacity, [tf.float32, tf.float32, tf.float32], [feature_dim, label_dim, geom_dim],
        #                         name='fifo_queue_val')
        self.x_val = tf.placeholder(dtype=tf.float32, shape=feature_dim, name='x_val_placeholder')
        # print("here               ",feature_dim, label_dim)
        self.y_val = tf.placeholder(dtype=tf.float32, shape=label_dim, name='y_val_placeholder')
        self.geom_val = tf.placeholder(dtype=tf.float32, shape=geom_dim, name='geom_val_placeholder')
        # self.enqueue_val = self.q_val.enqueue([self.x_val, self.y_val, self.geom_val], name='enqueue_val_operation')
        # self.num_threads_val = 1  # TODO: this is hardcoded for the time being

        self.enqueue = self.q.enqueue([self.x, self.y, self.geom, self.x_val, self.y_val, self.geom_val])
        self.num_threads = np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])



        r = np.loadtxt(os.path.join(self.root, self.data_type[0]+'_range.txt'))
        self.x_range = max(abs(r[0]), abs(r[1]))
        self.y_range = []
        self.y_num = []

        if 'ae' in config.arch:
            for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.y_num.append(p_num)
            for i in range(label_dim[0]):
                self.y_range.append([-1, 1])
        else:
            print(self.c_num)
            for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.y_range.append([p_min, p_max])
                self.y_num.append(p_num)
            print("initial_range", self.y_range)
    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, geom, data_type, x_range, y_range,
                           valid_paths, x_val, y_val, geom_val):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    val_id = rng.randint(len(valid_paths))
                    x_, y_, geom_, x_val_, y_val_, geom_val_ = preprocess(paths[id], data_type, x_range, y_range, valid_paths[val_id])

                    #geom_ = x_[...,1:]
                    #x_ = np.expand_dims(x_[..., 0], axis=3)
                    #print(x_.shape, y_.shape)
                    sess.run(enqueue, feed_dict={x: x_, y: y_, geom: geom_, x_val: x_val_, y_val: y_val_, geom_val: geom_val_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self.sess, 
                                                self.enqueue,
                                                self.coord,
                                                self.paths,
                                                self.rng,
                                                self.x,
                                                self.y,
                                                self.geom,
                                                self.data_type,
                                                self.x_range,
                                                self.y_range,
                                                self.valid_paths,
                                                self.x_val,
                                                self.y_val,
                                                self.geom_val,
                                                )
                                          ) for i in range(self.num_threads)]


        # define signal handler
        def signal_handler(signum, frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            #self.sess.run(self.q_val.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        #self.sess.run(self.q_val.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    #def batch_val(self):
    #    return self.q_val.dequeue_many(self.batch_size, name='val_dequeue_operation')

    def batch_(self, b_num):
        assert(len(self.paths) % b_num == 0)
        x_batch = []
        y_batch = []
        for i, filepath in enumerate(self.paths):
            x, _ = preprocess(filepath, self.data_type, self.x_range, self.y_range)
            x = np.expand_dims(x[..., 0], axis=3)
            x_batch.append(x)

            if (i+1) % b_num == 0:
                yield np.array(x_batch), y_batch
                x_batch.clear()
                y_batch.clear()

    def denorm(self, x=None, y=None):
        # input range [-1, 1] -> original range
        if x is not None:
            x *= self.x_range

        if y is not None:
            r = self.y_range
            for i, ri in enumerate(self.y_range):
                y[:,i] = (y[:,i]+1) * 0.5 * (ri[1]-ri[0]) + ri[0]
        return x, y

    def list_from_p(self, p_list):
        path_format = os.path.join(self.root, self.data_type[0], self.args['path_format'])
        #print(path_format)
        filelist = []
        for p in p_list:
            filelist.append(path_format % tuple(p[:3]))
        return filelist

    def list_from_p_val(self, p_list):
        path_format = os.path.join(self.root_val, self.args['path_format'])
        #print(path_format)
        filelist = []
        for p in p_list:
            filelist.append(path_format % tuple(p[:3]))
        return filelist

    def random_list2d(self, num):
        xs = []
        pis = []
        zis = []
        for _ in range(num):
            pi = []
            for y_max in self.y_num:
                pi.append(self.rng.randint(y_max))

            filepath = self.list_from_p([pi])[0]
            x, y = preprocess(filepath, self.data_type, self.x_range, self.y_range)
            if self.data_type[0] == 'v':
                b_ch = np.zeros((self.res_y, self.res_x, 1))
                x = np.concatenate((x, b_ch), axis=-1)
            elif self.data_type[0] == 'l':
                offset = 0.5
                eps = 1e-3
                x[x<(offset+eps)] = -1
                x[x>-1] = 1
            x = np.clip((x+1)*127.5, 0, 255)
            zi = [(p/float(self.y_num[i]-1))*2-1 for i, p in enumerate(pi)] # [-1,1]

            xs.append(x)
            pis.append(pi)
            zis.append(zi)
        return np.array(xs), pis, zis

    def random_list3d(self, num, z_samples):
        sample = {
            'x': [],
            'y': [],
            'geom': [],
            'xy': [],
            'zy': [],
            'xym': [],
            'zym': [],
            
            'p': [],
            'z': [],
            'z_val': [],
            'x_val': [],
            'y_val': [],
            'geom_val': [],
            'p_val': [],
            'xym_val': [],
            'zym_val': [],
        }
        
        for ll in range(np.asarray(z_samples).shape[1]):
            p = []
            print(z_samples)
            p.append(int((z_samples[0][ll][2]+1)*10))

            p.insert(0,0)
            p.insert(0,0)
            p.append(0)
            p.append(0)
            p.append(0)
            sample['p'].append(p)

            file_path = self.list_from_p([p])[0]
            print('debug. File_path = %s'%(file_path))
            p_val = []

            p_val.append(int((z_samples[0][ll][2] + 1) * 10))

            p_val.insert(0, 1)
            p_val.insert(0, 10)
            p_val.append(0)
            p_val.append(0)
            p_val.append(0)
            sample['p_val'].append(p_val)

            file_path_val = self.list_from_p_val([p_val])[0]
            print(file_path_val)
            x, y, geom, x_val, y_val, geom_val = preprocess(file_path, self.data_type, self.x_range, self.y_range, file_path_val)
            #print(np.unique(x))
            #print(np.unique(x_val))
            sample['x'].append(x)
            sample['y'].append(y)
            sample['geom'].append(geom)
            sample['x_val'].append(x_val)
            sample['y_val'].append(y_val)
            sample['geom_val'].append(geom_val)
            sample['z'].append([y[0], y[1], z_samples[0][ll][2]]) #, y[3], y[4], y[5]])
            sample['z_val'].append([y_val[0], y_val[1], z_samples[0][ll][2]])
            #print(sample['z'])
            xy = plane_view_np(x, xy_plane=True, project=True)
            zy = plane_view_np(x, xy_plane=False, project=True)
            xym = plane_view_np(x, xy_plane=True, project=False)
            zym = plane_view_np(x, xy_plane=False, project=False)
            xym_val = plane_view_np(x_val, xy_plane=True, project=False)
            zym_val = plane_view_np(x_val, xy_plane=False, project=False)

            #sample['xy'].append(xy)
            #sample['zy'].append(zy)
            sample['xym'].append(xym)
            sample['zym'].append(zym)
            sample['xym_val'].append(xym_val)
            sample['zym_val'].append(zym_val)

                        
        sample['x'] = np.array(sample['x'])
        sample['y'] = np.array(sample['y'])
        sample['geom'] = np.array(sample['geom'])

        sample['x_val'] = np.array(sample['x_val'])
        sample['y_val'] = np.array(sample['y_val'])
        sample['geom_val'] = np.array(sample['geom_val'])

        #sample['xy'] = np.array(sample['xy'])
        #sample['zy'] = np.array(sample['zy'])
        sample['xym'] = np.array(sample['xym'])
        sample['zym'] = np.array(sample['zym'])
        sample['xym_val'] = np.array(sample['xym_val'])
        sample['zym_val'] = np.array(sample['zym_val'])

        

        return sample

    def random_list(self, num, z_samples):
        if self.is_3d:
            return self.random_list3d(num, z_samples)
        else:
            return self.random_list2d(num)
    

def preprocess(file_path, data_type, x_range, y_range, val_path):
    with np.load(file_path) as data:
        y = data['y']
        x = np.expand_dims(data['x'][..., 0][int(round(y[3] * 128)) - 32:int(round(y[3] * 128)) + 32,
                           int(round(y[4] * 128)) - 32:int(round(y[4] * 128)) + 32,
                           int(round(y[5] * 128)) - 32:int(round(y[5] * 128)) + 32], axis=3)
        geom = data['x'][..., 1:][int(round(y[3] * 128)) - 32:int(round(y[3] * 128)) + 32,
               int(round(y[4] * 128)) - 32:int(round(y[4] * 128)) + 32,
               int(round(y[5] * 128)) - 32:int(round(y[5] * 128)) + 32]
        y = y[:3]

    with np.load(val_path) as val_data:
        val_y = val_data['y']
        # print(val_y)
        # print(".............AAAAAAAAAA", [round(y[3]*128)-32,round(y[3]*128)+32,round(y[4]*128)-32,round(y[4]*128)+32,round(y[5]*128)-32,round(y[5]*128)+32])
        val_x = np.expand_dims(val_data['x'][..., 0][int(round(val_y[3] * 128)) - 32:int(round(val_y[3] * 128)) + 32,
                               int(round(val_y[4] * 128)) - 32:int(round(val_y[4] * 128)) + 32,
                               int(round(val_y[5] * 128)) - 32:int(round(val_y[5] * 128)) + 32], axis=3)
        val_geom = val_data['x'][..., 1:][int(round(val_y[3] * 128)) - 32:int(round(val_y[3] * 128)) + 32,
                   int(round(val_y[4] * 128)) - 32:int(round(val_y[4] * 128)) + 32,
                   int(round(val_y[5] * 128)) - 32:int(round(val_y[5] * 128)) + 32]
        val_y = val_y[:3]


    # normalize
    if data_type[0] == 'd':
        x = x * 2 - 1
    else:
        x /= x_range
        val_x /= x_range
    # print("preprocess_range", y_range, y)
    for i, ri in enumerate(y_range):
        y[i] = (y[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
        val_y[i] = (val_y[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
        # y[i] = y[i]/ri[1]
    # print("processed", y)
    return x, [round(elem, 2) for elem in y], geom, val_x, [round(elem, 2) for elem in val_y], val_geom

