import tensorflow as tf
import numpy as np

from config import config

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width

def TESnet(x, scope, drop_rate, reuse, getter=None):
	with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
		x = tf.layers.conv2d(x, 128, 3, 1, 'same', reuse=reuse, name='conv1a')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv1a/LReLU')
		x = tf.layers.conv2d(x, 128, 3, 1, 'same', reuse=reuse, name='conv1b')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv1b/LReLU')
		x = tf.layers.conv2d(x, 128, 3, 1, 'same', reuse=reuse, name='conv1c')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv1c/LReLU')

		x = tf.nn.max_pool(x, [1,2,2,1], [1,1,1,1], 'SAME', name='max_pooling_1')
		x = tf.layers.dropout(x, rate=drop_rate, name='dropout_1')

		x = tf.layers.conv2d(x, 256, 3, 1, 'same', reuse=reuse, name='conv2a')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv2a/LReLU')
		x = tf.layers.conv2d(x, 256, 3, 1, 'same', reuse=reuse, name='conv2b')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv2b/LReLU')
		x = tf.layers.conv2d(x, 256, 3, 1, 'same', reuse=reuse, name='conv2c')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv2c/LReLU')

		x = tf.nn.max_pool(x, [1,2,2,1], [1,1,1,1], 'SAME', name='max_pooling_2')
		x = tf.layers.dropout(x, rate=drop_rate, name='dropout_2')

		x = tf.layers.conv2d(x, 512, 3, 1, 'valid', reuse=reuse, name='conv3a')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv3a/LReLU')
		x = tf.layers.conv2d(x, 256, 1, 1, 'same', reuse=reuse, name='conv3b')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv3b/LReLU')
		x = tf.layers.conv2d(x, 128, 1, 1, 'same', reuse=reuse, name='conv3c')
		x = tf.nn.leaky_relu(x, alpha=0.1, name='conv3c/LReLU')

		x = tf.layers.average_pooling2d(x, 6, 1, 'same', name='avg_pool')
		x = tf.reduce_mean(x, axis=[1, 2])

		x = tf.layers.dense(x, 10)

	return x

