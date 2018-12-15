import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class CPM_Model(object):
	def __init__(self, stages, joints, cfg):
		self.stages = stages
		self.stage_heatmap = []
		self.stage_loss = [0] * stages
		self.total_loss = 0
		self.input_image = None
		self.center_map = None
		self.gt_heatmap = None
		self.learning_rate = 0
		self.merged_summary = None
		self.joints = joints
#        self.batch_size = cfg.FLAGS.batch_size

	def build_model(self, input_image, batch_size):
		self.batch_size = batch_size
		self.input_image = input_image
		with tf.variable_scope('sub_stages'):

			sub_conv1 = tf.layers.conv2d(
				inputs=input_image,
				filters=64,
				kernel_size=[5, 5],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='sub_conv1')

			sub_pool1 = tf.layers.max_pooling2d(
				inputs=sub_conv1,
				pool_size=[3, 3],
				strides=2,
				padding='same',
				name='sub_pool1')

			sub_conv2 = tf.layers.conv2d(
				inputs=sub_pool1,
				filters=64,
				kernel_size=[5, 5],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='sub_conv2')

			sub_pool2 = tf.layers.max_pooling2d(
				inputs=sub_conv2,
				pool_size=[3, 3],
				strides=2,
				padding='same',
				name='sub_pool2')

			sub_conv3 = tf.layers.conv2d(
				inputs=sub_pool2,
				filters=128,
				kernel_size=[5, 5],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='sub_conv3')

			# sub_pool3 = tf.layers.max_pooling2d(
			# 	inputs=sub_conv3,
			# 	pool_size=[3, 3],
			# 	strides=1,                                                      ##CHANGED TO 1 SO HEATMAP IS 32X32 NOT 16X16
			# 	padding='same',
			# 	name='sub_pool3')                                               # NO NEED, GET RID

			sub_conv4 = tf.layers.conv2d(
				inputs=sub_conv3,
				filters=32,
				kernel_size=[3, 3],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='sub_conv4')

			self.sub_stage_img_feature = sub_conv4

		with tf.variable_scope('stage_1'):

			conv1 = tf.layers.conv2d(
				inputs=self.sub_stage_img_feature,
				filters=256,
				kernel_size=[5, 5],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='conv1')

			conv2 = tf.layers.conv2d(
				inputs=conv1,
				filters=256,
				kernel_size=[1, 1],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='conv2')

			self.stage_heatmap.append(tf.layers.conv2d(
				inputs=conv2,
				filters=self.joints,
				kernel_size=[1, 1],
				strides=[1, 1],
				padding='same',
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='stage_heatmap'))

		for stage in range(2, self.stages + 1):
			self._middle_conv(stage)

	def _middle_conv(self, stage):
		with tf.variable_scope('stage_' + str(stage)):
			self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
												 self.sub_stage_img_feature,
												 # self.center_map],
												 ],
												axis=3)
			mid_conv1 = tf.layers.conv2d(
				inputs=self.current_featuremap,
				filters=128,
				kernel_size=[7, 7],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='mid_conv1')

			mid_conv2 = tf.layers.conv2d(
				inputs=mid_conv1,
				filters=128,
				kernel_size=[7, 7],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='mid_conv2')

			mid_conv3 = tf.layers.conv2d(
				inputs=mid_conv2,
				filters=128,
				kernel_size=[7, 7],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='mid_conv3')

			mid_conv4 = tf.layers.conv2d(
				inputs=mid_conv3,
				filters=128,
				kernel_size=[1, 1],
				strides=[1, 1],
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='mid_conv4')

			self.current_heatmap = tf.layers.conv2d(
				inputs=mid_conv4,
				filters=self.joints,
				kernel_size=[1, 1],
				strides=[1, 1],
				padding='same',
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='mid_conv5')

			self.stage_heatmap.append(self.current_heatmap)

	def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
		self.gt_heatmap = gt_heatmap
		self.total_loss = 0
		self.learning_rate = lr
		self.lr_decay_rate = lr_decay_rate
		self.lr_decay_step = lr_decay_step

		for stage in range(self.stages):
			with tf.variable_scope('stage' + str(stage + 1) + '_loss'):

				self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
													   name='l2_loss') / self.batch_size
			tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])
		with tf.variable_scope('total_loss'):
			self.total_loss = self.stage_loss[self.stages-1]                       # CHANGE TOTAL LOSS TO ARRAY WITH INDIVIDUAL LOSSES BY STAGE AND JOINT
			tf.summary.scalar('total loss', self.total_loss)

		with tf.variable_scope('train'):
			self.global_step = tf.train.get_or_create_global_step()

			self.lr = tf.train.exponential_decay(self.learning_rate,
												 global_step=self.global_step,
												 decay_rate=self.lr_decay_rate,
												 decay_steps=self.lr_decay_step)
			tf.summary.scalar('learning rate', self.lr)

			self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
															global_step=self.global_step,
															learning_rate=self.lr,
															optimizer='Adam')

		with tf.variable_scope('joint_losses'):
			for joint_id in range(self.joints):
				joint_loss = tf.nn.l2_loss(self.stage_heatmap[self.stages-1][:,:,:,joint_id] - self.gt_heatmap[:,:,:,joint_id],
							  name='l2_loss_joint' + str(joint_id+1)) / self.batch_size
				tf.summary.scalar('joint' + str(joint_id + 1) + '_loss', joint_loss)

		self.merged_summary = tf.summary.merge_all()


	def model_train(self, batch_x_np, batch_y_np, sess):

		total_loss_np, stage_loss, _, summary, current_lr, stage_heatmap, global_step = sess.run([
			self.total_loss,
			self.stage_loss,
			self.train_op,
			self.merged_summary,
			self.lr,
			self.stage_heatmap,
			self.global_step
		],
			feed_dict={self.input_image: batch_x_np,
					   self.gt_heatmap: batch_y_np})

		return total_loss_np, stage_loss, _, summary, current_lr, stage_heatmap, global_step