# -*- coding: utf-8 -*-

# Hyper parameters

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('TFrecords_dir', default_value=['data_reader_write/train5000_-45to45.tfrecords'], docstring='Training data tfrecords')

# tf.app.flags.DEFINE_string('pretrained_model',default_value='cpm_hand.pkl',docstring='Pretrained mode')

tf.app.flags.DEFINE_integer('input_height', default_value=128, docstring='Input image length')
tf.app.flags.DEFINE_integer('input_width', default_value=128, docstring='Input image width')
tf.app.flags.DEFINE_integer('input_dim', default_value=3, docstring='Input image channel')

tf.app.flags.DEFINE_integer('output_height', default_value=32, docstring='Output heatmap height')
tf.app.flags.DEFINE_integer('output_width', default_value=32, docstring='Output heatmap width')
tf.app.flags.DEFINE_integer('output_dim', default_value=15, docstring='Output heatmap dimension')

tf.app.flags.DEFINE_integer('stages', default_value=5, docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('center_radius', default_value=21, docstring='Center map gaussian variance')

tf.app.flags.DEFINE_integer('num_of_features', default_value=15, docstring='Number of joints')
tf.app.flags.DEFINE_integer('batch_size', default_value=7, docstring='Training mini-batch size')
tf.app.flags.DEFINE_integer('training_iterations', default_value=100000, docstring='Training iterations')

tf.app.flags.DEFINE_integer('num_of_scales', default_value=3, docstring='num of scales')

tf.app.flags.DEFINE_integer('lr', default_value=0.001, docstring='Learning rate')
tf.app.flags.DEFINE_integer('lr_decay_rate', default_value=0.96, docstring='Learning rate decay rate')
tf.app.flags.DEFINE_integer('lr_decay_step',default_value=5000, docstring='Learning rate decay steps')

tf.app.flags.DEFINE_string('saver_dir', default_value='/saver/_cpm_quad' , docstring='Saved model name')
tf.app.flags.DEFINE_string('log_dir', default_value='/logs/_cpm_quad', docstring='Log directory name')
tf.app.flags.DEFINE_string('color_channel', default_value='RGB', docstring='dim')

tf.app.flags.DEFINE_integer('istraining', default_value=True, docstring='Learning rate decay steps')
tf.app.flags.DEFINE_integer('isbc', default_value=False, docstring='training on blue crystal')

tf.app.flags.DEFINE_string('model_name', default_value='cmu_VerySlim_3stages_pc/VerySlim_CPM_pc', docstring='name of the model')