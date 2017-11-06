from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v2(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.8,min_depth=16,depth_multiplier=1.0,prediction_fn=slim.softmax,spatial_squeeze=True,reuse=None,scope='InceptionV2'):

  with tf.variable_scope(scope, 'InceptionV2', [inputs, num_classes],reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1,padding='SAME',data_format='NHWC'):
          
          depth = lambda d: max(int(d * depth_multiplier), min_depth)
          depthwise_multiplier = min(int(depth(64) / 3), 8)

          # 224 x 224 x 3
          net = slim.separable_conv2d(inputs, depth(64), [7, 7],depth_multiplier=depthwise_multiplier,stride=2,padding='SAME',weights_initializer=trunc_normal(1.0),scope='Conv2d_1a_7x7')
          # 112 x 112 x 64
          net = slim.max_pool2d(net, [3, 3], scope='MaxPool_2a_3x3', stride=2)
          # 56 x 56 x 64
          net = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_2b_1x1',weights_initializer=trunc_normal(0.1))
          # 56 x 56 x 64
          net = slim.conv2d(net, depth(192), [3, 3], scope='Conv2d_2c_3x3')
          # 56 x 56 x 192
          net = slim.max_pool2d(net, [3, 3], scope='MaxPool_3a_3x3', stride=2)
          # 28 x 28 x 192

          # Inception module.
          with tf.variable_scope('Mixed_3b'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 28 x 28 x 256
          with tf.variable_scope('Mixed_3c'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 28 x 28 x 320
          with tf.variable_scope('Mixed_4a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(128), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
              branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

          # 14 x 14 x 576
          with tf.variable_scope('Mixed_4b'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(64), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(96), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 14 x 14 x 576
          with tf.variable_scope('Mixed_4c'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(96), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(96), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 14 x 14 x 576
          with tf.variable_scope('Mixed_4d'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(128), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(128), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(96), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 14 x 14 x 576
          with tf.variable_scope('Mixed_4e'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(128), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(160), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(96), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 14 x 14 x 576
          with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(128), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],scope='Conv2d_0b_3x3')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.max_pool2d(net, [3, 3], stride=2,scope='MaxPool_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

          # 7 x 7 x 1024
          with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(160), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          # 7 x 7 x 1024
          with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(net, depth(192), [1, 1],weights_initializer=trunc_normal(0.09),scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
              branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],weights_initializer=trunc_normal(0.1),scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

          with tf.variable_scope('Logits'):
            net = slim.avg_pool2d(net,[7,7], padding='VALID',scope='AvgPool_1a_7x7')
            # 1 x 1 x 1024
            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,normalizer_fn=None, scope='Conv2d_1c_1x1')
            if spatial_squeeze:
              logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            return logits

