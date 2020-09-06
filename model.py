import numpy as np
import tensorflow as tf
from ops import *


def TumorGenerator(geom,y,filters,output_shape, num_conv , repeat,arch, name = 'tumor', reuse=tf.AUTO_REUSE ):

    with tf.variable_scope(name, reuse=reuse) as vs:
        if arch == 'alternative':
            # perform a concatenation betwe a latent anatomy of shape bx8x8x8xself.encode_ch, and bx8x8x8.input_ch .
            # encode_ch and input_ch are hyperparameters, to be adjusted in config.py or command line

            # TODO: for the time being we are just declaring them here
            # since filters=128 by default, thought these are common sense values

            encode_ch = 64
            input_ch = 64
            test_choice = 3

            if test_choice == 0:
                zz, _ = EncoderBE3(geom, filters, encode_ch, 'enc',
                                   num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                                   act=relu, reuse=reuse, alternative_output_shape=True)

                y_expanded_shape = [8, 8, 8, input_ch]

                y_expanded = linear(y, int(np.prod(y_expanded_shape)), name='expand_params_fc')
                y_expanded = lrelu(y_expanded)
                y_expanded = tf.reshape(y_expanded, [-1] + y_expanded_shape)

                param_geom = tf.concat([y_expanded, zz], axis=-1)
                G_, _ = GeneratorBE3(param_geom, filters, output_shape, reuse = reuse,
                                                   num_conv=num_conv, repeat=repeat, alternative_input_shape=True)
            elif test_choice == 1:
                zz, _ = FullyConnectedEncoder(geom, filters, encode_ch, 'enc',
                                   num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                                   act=relu, reuse=reuse, alternative_output_shape=True,test_number=1)

                y_expanded_shape = [8, 8, 8, input_ch]

                y_expanded = linear(y, int(np.prod(y_expanded_shape)), name='expand_params_fc')
                y_expanded = relu(y_expanded)
                y_expanded = tf.reshape(y_expanded, [-1] + y_expanded_shape)

                param_geom = tf.concat([y_expanded, zz], axis=-1)
                G_, _ = GeneratorBE3(param_geom, filters, output_shape, reuse=reuse,
                                     num_conv=num_conv, repeat=repeat, alternative_input_shape=True)

            elif test_choice == 2:
                zz, _ = FullyConnectedEncoder(geom, filters, encode_ch, 'enc',
                                              num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                                              act=relu, reuse=reuse, alternative_output_shape=True)

                y_expanded_shape = [8, 8, 8, 128]

                y_expanded = linear(y, int(np.prod(y_expanded_shape)), name='expand_params_fc')
                y_expanded = relu(y_expanded)
                y_expanded = tf.reshape(y_expanded, [-1] + y_expanded_shape)

                param_geom = y_expanded + zz
                G_, _ = GeneratorBE3(param_geom, filters, output_shape, reuse=reuse,
                                     num_conv=num_conv, repeat=repeat, alternative_input_shape=True)

            elif test_choice == 3:
                encode_ch = 125
                input_ch = 3
                zz, _ = EncoderBE3(geom, filters, encode_ch, 'enc',
                                   num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                                   act=relu, reuse=reuse, alternative_output_shape=True)

                y_expanded_shape = [8, 8, 8, input_ch]

                y_expanded = linear(y, int(np.prod(y_expanded_shape)), name='expand_params_fc')
                y_expanded = tf.reshape(y_expanded, [-1] + y_expanded_shape)

                param_geom = tf.concat([y_expanded, zz], axis=-1)
                G_, _ = GeneratorBE3(param_geom, filters, output_shape, reuse=reuse,
                                     num_conv=num_conv,act=relu, repeat=repeat, alternative_input_shape=True)

            elif test_choice == 4 :
                zz, _ = FullyConnectedEncoder(geom, filters, encode_ch, 'enc',
                                              num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                                              act=relu, reuse=reuse, alternative_output_shape=True, test_number=4)



                param_geom = tf.concat([y, zz], axis=1)
                param_geom_skip = param_geom

                param_geom = relu(param_geom)
                param_geom = linear(param_geom, 2003, name=str(1) + '_fc')
                param_geom = relu(param_geom)
                param_geom = linear(param_geom, 2003, name=str(2) + '_fc')
                param_geom = relu(param_geom)
                param_geom = linear(param_geom, 2003, name=str(3) + '_fc')

                param_geom +=param_geom_skip

                G_, _ = GeneratorBE3(param_geom, filters, output_shape, reuse=reuse,
                                     num_conv=num_conv, repeat=repeat, alternative_input_shape=False, act= relu)


        else:
            # ivan's original architecture, latent anatomy represented as 1d array, concatenated with input parameters as a 1d array
            zz, _ = EncoderBE3(geom, filters, 1024, 'enc',
                               num_conv=num_conv - 1, conv_k=3, repeat=repeat,
                               act=lrelu, reuse=reuse)
            param_geom = tf.concat([y, zz], axis=1)
            print(param_geom)
            print(y)
            print(zz)
            G_,_ = GeneratorBE3(param_geom, filters, output_shape,
                                               num_conv=num_conv, repeat=repeat,reuse = reuse)

    variables = tf.contrib.framework.get_variables(vs)
    return G_,variables

def FullyConnectedEncoder(x, filters, z_num, name='enc', num_conv=3, conv_k=3, repeat=0, act=lrelu, reuse=False,
               alternative_output_shape=False, skip_connect=False, test_number =4):
    with tf.variable_scope(name, reuse=reuse) as vs:
        print('debug.shape of input:', get_conv_shape(x))
        #proposed architecture: 3 fully connected layers, 100 neurons each:
        #flatten input:
        b = get_conv_shape(x)[0]
        layer_num = 0
        x0 = x
        repeat_num = 4
        ch = 0
        num_conv = 2
        filters = 16
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
                layer_num += 1

            # skip connection
            # if skip_connect:
            #     x = tf.concat([x, x0], axis=-1)
            #     ch += filters
            # else:
            #     x += x0

            if idx < repeat_num - 1:
                filters *=2
                x = conv3d(x, filters, k=conv_k, s=2, act=act, name=str(layer_num) + '_conv')
                print('debug.Shape of x: ', get_conv_shape(x))
                layer_num += 1
                x0 = x

        flat = tf.reshape(x, [b, -1])
        out = linear(flat, 2000, name=str(0) + '_fc')


        if test_number == 1:
            out = linear(out, 8*8*8*64, name=str(3) + '_fc')
            out = tf.reshape(out, [b,8,8,8,64])
        elif test_number !=4:
            out = linear(out, 8 * 8 * 8 * 128, name=str(3) + '_fc')
            out = tf.reshape(out, [b, 8, 8, 8, 128])

        # x_shape = get_conv_shape(x)[1:]
        # if repeat == 0:
        #     repeat_num = int(np.log2(np.max(x_shape[:-1]))) - 2
        # else:
        #     repeat_num = repeat
        # assert (repeat_num > 0 and np.sum([i % np.power(2, repeat_num - 1) for i in x_shape[:-1]]) == 0)
        #
        # ch = filters
        # layer_num = 0
        # x = conv3d(x, ch, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
        # x0 = x
        # layer_num += 1
        # for idx in range(repeat_num):
        #     for _ in range(num_conv):
        #         x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
        #         layer_num += 1
        #
        #     # skip connection
        #     if skip_connect:
        #         x = tf.concat([x, x0], axis=-1)
        #         ch += filters
        #     else:
        #         x += x0
        #
        #     if idx < repeat_num - 1:
        #         x = conv3d(x, ch, k=conv_k, s=2, act=act, name=str(layer_num) + '_conv')
        #         layer_num += 1
        #         x0 = x
        #         # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        #
        # if alternative_output_shape:
        #     shape = get_conv_shape(x)
        #     for dim in shape[1:-1]:
        #         assert int(dim) is 8, 'Problem'
        #     out = conv3d(x, z_num, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
        # else:
        #     b = get_conv_shape(x)[0]
        #     flat = tf.reshape(x, [b, -1])
        #     out = linear(flat, z_num, name=str(layer_num) + '_fc')

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def GeneratorBE3(z, filters, output_shape, name='G',
                 num_conv=4, conv_k=3, last_k=3, repeat=0, skip_concat=False, act=relu, reuse=False,
                 alternative_input_shape=False):
    with tf.variable_scope(name, reuse=reuse) as vs:

        if repeat == 0:
            repeat_num = int(np.log2(np.max(output_shape[:-1]))) - 2
        else:
            repeat_num = repeat
        assert (repeat_num > 0 and np.sum([i % np.power(2, repeat_num - 1) for i in output_shape[:-1]]) == 0)

        if alternative_input_shape:
            shape = get_conv_shape(z)
            for dim in shape[1:-1]:
                assert int(dim) is 8, 'Problem'

            x = z
            layer_num = 0

        else:

            x0_shape = [int(i / np.power(2, repeat_num - 1)) for i in output_shape[:-1]] + [filters]
            print('first layer:', x0_shape, 'to', output_shape)

            num_output = int(np.prod(x0_shape))
            layer_num = 0
            x = linear(z, num_output, name=str(layer_num) + '_fc')
            layer_num += 1
            x = tf.reshape(x, [-1] + x0_shape)

        x0 = x

        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
                layer_num += 1

            if idx < repeat_num - 1:
                if skip_concat:
                    x = upscale3(x, 2)
                    x0 = upscale3(x0, 2)
                    x = tf.concat([x, x0], axis=-1)
                else:
                    x += x0
                    x = upscale3(x, 2)
                    x0 = x
                #filters /=2
                x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
                layer_num += 1
                print('shape of x in generator:',get_conv_shape(x))
                #x0 = x

            elif not skip_concat:
                x += x0

        out = conv3d(x, output_shape[-1], k=last_k, s=1, name=str(layer_num) + '_conv')

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def DiscriminatorPatch(x, filters, name='D', train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        repeat_num = 3 # if c4k3s2, rfs 95, w/16=8, if c3k3s2, rfs 47, w/8=16
        d = int(filters/2)
        for _ in range(repeat_num): 
            x = conv2d(x, d, k=3, act=lrelu) # 64/32/16-64/128/256
            d *= 2
        x = conv2d(x, d, k=3, s=1, act=lrelu) # 16x16x512
        out = conv2d(x, 1, k=3, s=1) # 16x16x1

        # x = conv2d(x, int(d/2), k=3, s=2, act=lrelu) # 8x8x256
        # b = get_conv_shape(x)[0]
        # flat = tf.reshape(x, [b, -1])
    variables = tf.contrib.framework.get_variables(vs)    
    return out, variables

def DiscriminatorPatch3(x, filters, name='D', train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        repeat_num = 3 # if c4k3s2, rfs 95, w/16=8, if c3k3s2, rfs 47, w/8=16
        d = int(filters/2)
        for _ in range(repeat_num): 
            x = conv3d(x, d, k=3, act=lrelu) # 64/32/16-64/128/256
            d *= 2
        x = conv3d(x, d, k=3, s=1, act=lrelu) # 16x16x512
        out = conv3d(x, 1, k=3, s=1) # 16x16x1

    variables = tf.contrib.framework.get_variables(vs)    
    return out, variables

def EncoderBE(x, filters, z_num, name='enc', num_conv=4, conv_k=3, repeat=0, act=lrelu, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        x_shape = get_conv_shape(x)[1:]
        if repeat == 0:
            repeat_num = int(np.log2(np.max(x_shape[:-1]))) - 2
        else:
            repeat_num = repeat
        assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in x_shape[:-1]]) == 0)
        
        ch = filters
        layer_num = 0
        x = conv2d(x, ch, k=conv_k, s=1, act=act, name=str(layer_num)+'_conv')
        x0 = x
        layer_num += 1
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv2d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num)+'_conv')
                layer_num += 1

            # skip connection
            x = tf.concat([x, x0], axis=-1)
            ch += filters

            if idx < repeat_num - 1:
                x = conv2d(x, ch, k=conv_k, s=2, act=act, name=str(layer_num)+'_conv')
                layer_num += 1
                x0 = x
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        b = get_conv_shape(x)[0]
        flat = tf.reshape(x, [b, -1])
        out = linear(flat, z_num, name=str(layer_num)+'_fc')

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def EncoderBE3(x, filters, z_num, name='enc', num_conv=3, conv_k=3, repeat=0, act=lrelu, reuse=False,
               alternative_output_shape=False, skip_connect=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        x_shape = get_conv_shape(x)[1:]
        if repeat == 0:
            repeat_num = int(np.log2(np.max(x_shape[:-1]))) - 2
        else:
            repeat_num = repeat
        assert (repeat_num > 0 and np.sum([i % np.power(2, repeat_num - 1) for i in x_shape[:-1]]) == 0)

        ch = filters
        layer_num = 0
        x = conv3d(x, ch, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
        x0 = x
        layer_num += 1
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
                layer_num += 1

            # skip connection
            if skip_connect:
                x = tf.concat([x, x0], axis=-1)
                ch += filters
            else:
                x += x0
            print('debug.Shape of x: ', get_conv_shape(x))
            if idx < repeat_num - 1:
                x = conv3d(x, ch, k=conv_k, s=2, act=act, name=str(layer_num) + '_conv')
                layer_num += 1
                x0 = x
                # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        if alternative_output_shape:
            shape = get_conv_shape(x)
            for dim in shape[1:-1]:
                assert int(dim) is 8, 'Problem'
            out = conv3d(x, z_num, k=conv_k, s=1, act=act, name=str(layer_num) + '_conv')
        else:
            b = get_conv_shape(x)[0]
            flat = tf.reshape(x, [b, -1])
            out = linear(flat, z_num, name=str(layer_num) + '_fc')

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def AE(x, filters, z_num, name='AE', num_conv=4, conv_k=3, last_k=3, repeat=0,
                    act=lrelu, skip_concat=False, use_sparse=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        z, _ = EncoderBE(x, filters, z_num, 'enc',
                         num_conv=num_conv-1, conv_k=conv_k, repeat=repeat,
                         act=act, reuse=reuse)
        if use_sparse: z = tf.sigmoid(z)
        out, _ = GeneratorBE(z, filters, get_conv_shape(x)[1:], 'dec',
                             num_conv=num_conv, conv_k=conv_k, last_k=last_k, repeat=repeat,
                             skip_concat=skip_concat, act=act, reuse=reuse)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def AE3(x, filters, z_num, name='AE', num_conv=4, conv_k=3, last_k=3, repeat=0,
                    act=lrelu, skip_concat=False, use_sparse=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        z, _ = EncoderBE3(x, filters, z_num, 'enc',
                         num_conv=num_conv-1, conv_k=conv_k, repeat=repeat,
                         act=act, reuse=reuse)
        if use_sparse: z = tf.sigmoid(z)
        out, _ = GeneratorBE3(z, filters, get_conv_shape(x)[1:], 'dec',
                             num_conv=num_conv, conv_k=conv_k, last_k=last_k, repeat=repeat,
                             skip_concat=skip_concat, act=act, reuse=reuse)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables
    
def NN(x, filters, onum, name='NN', act=tf.nn.elu, dropout=0.1, train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        x = slim.dropout(batch_norm(linear(x, filters*2), train, act=act), dropout, is_training=train)
        x = slim.dropout(batch_norm(linear(x, filters), train, act=act), dropout, is_training=train)
        out = linear(x, onum)
    variables = tf.contrib.framework.get_variables(vs)    
    return out, variables

def main(_):
    # #########
    # # 2d
    # b_num = 8
    # c_num = 3

    # res_y = 128
    # res_x = 96
    # ch_num = 2

    # filters = 128

    # z = tf.placeholder(dtype=tf.float32, shape=[b_num, c_num])
    # x = tf.placeholder(dtype=tf.float32, shape=[b_num, res_y, res_x, ch_num])
    # output_shape = get_conv_shape(x)[1:]

    # # dec, d_var = GeneratorBE(z, filters, output_shape, name='dec')
    # # dis, g_var = DiscriminatorPatch(dec, filters)

    # z_num = 16
    # ae, z_ae, a_var = AE(x, filters, z_num, name='AE')
    ##############
   
    #########
    # # 3d
    # b_num = 4
    # c_num = 3

    # res_x = 112
    # res_y = 64
    # res_z = 32
    # ch_num = 3

    # filters = 128

    # z = tf.placeholder(dtype=tf.float32, shape=[b_num, c_num])
    # x = tf.placeholder(dtype=tf.float32, shape=[b_num, res_z, res_y, res_x, ch_num])
    # output_shape = get_conv_shape(x)[1:]

    # # dec, d_var = GeneratorBE3(z, filters, output_shape, name='dec')
    # # dis, g_var = DiscriminatorPatch3(dec, filters)

    # z_num = 16
    # ae, z_ae, a_var = AE3(x, filters, z_num, name='AE')
    #############

    ########
    # NN
    b_num = 1024
    c_num = 16
    p_num = 2
    x_num = c_num + p_num
    z_num = c_num - p_num
    x = tf.placeholder(dtype=tf.float32, shape=[b_num, x_num])
    filters = 512
    y = NN(x, filters, z_num)

    show_all_variables()

if __name__ == '__main__':
    tf.app.run()
