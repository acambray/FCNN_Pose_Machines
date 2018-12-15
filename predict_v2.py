import tensorflow as tf
from pylab import *
import matplotlib.pyplot as plt
import scipy.io as sio
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import the config file according to the tasks
import configs.config as cfg
# import the model file according to the tasks
import models.cpm_cmu_slim as cpm                                                                                                   # CHANGE
import os
# training on the server
if cfg.FLAGS.isbc is False:
	import cv2


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def show_individual_heatmaps(new_size, demo_img, stage_heatmap, colormap, model_name, id=1):
	new_size = 128
	image_blend = cv2.resize(demo_img, (new_size, new_size))
	# image_blend = image_blend.astype('uint8')

	for stage_id in list(range(len(stage_heatmap))):
		hm = stage_heatmap[stage_id]
		hm = hm[0]
		hm_list = []
		hmb_list = []
		hm_stack = []
		joint_coord = []
		joint_name = ['FL Paw', 'FL wrist', 'FL Elbow',
					  'FR Paw', 'FR wrist', 'FR Elbow',
					  'BL Paw', 'BL wrist', 'BL Elbow',
					  'BR Paw', 'BR wrist', 'BR Elbow',
					  'Hip', 'Shoulder', 'Face']

		for joint_id in list(range(0, 15)):
			hm_ind = hm[:, :, joint_id]
			hm_ind = cv2.resize(hm_ind, (new_size, new_size))
			joint_coord.append(list(np.unravel_index(np.argmax(hm_ind), hm_ind.shape)))
			hm_ind = cv2.normalize(hm_ind, None, 0, 255, cv2.NORM_MINMAX)
			hm_ind = hm_ind.astype('uint8')
			hmb_ind = cv2.applyColorMap(hm_ind, colormap)
			hm_blend = cv2.addWeighted(image_blend, 0.65, hmb_ind, 0.35, 0, None)

			# Add joint name and joint circle
			cv2.putText(hm_blend, joint_name[joint_id], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
			if 1 or stage_id+1 is len(stage_heatmap):
				cv2.circle(hm_blend, (joint_coord[joint_id][1], joint_coord[joint_id][0]), 1, (255, 255, 255), 1)

			hm_list.append(hm_ind)
			hmb_list.append(hm_blend)

		# Stack blended image+heatmaps
		hmb_stack = np.hstack(hmb_list)

		# Split into 3 rows of (6,6,3) heatmaps
		row1 = hmb_stack[:, 0:new_size * 6, :]
		row2 = hmb_stack[:, new_size * 6:new_size * 12, :]
		row3 = hmb_stack[:, new_size * 12::, :]
		black_square = np.zeros((new_size, new_size * 3, 3), np.int8)
		row3 = np.hstack([row3, black_square])

		hmb_fullstack = np.vstack((row1, row2, row3))
		hmb_fullstack = hmb_fullstack.astype('uint8')

		newpath = 'results/Angular/' + model_name +  '/heatmaps/' + str(id)                                                             # CHANGE
		if not os.path.exists(newpath):
			os.makedirs(newpath)
		cv2.imwrite(newpath + '/(' + str(id) + ', ' + str(stage_id + 1) + ') Heatmap.png', hmb_fullstack)
		# cv2.imshow('(' + str(id) + ') FULLSTACK - blend' + 'STAGE = ' + str(stage_id+1), hmb_fullstack)
		# cv2.moveWindow('(' + str(id) + ') FULLSTACK - blend' + 'STAGE = ' + str(stage_id+1), 20, 20)
		# cv2.waitKey(8000)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_label_coords(text_path, new_size):
	input_file = open(text_path, 'r')
	print('READING GROUND-TRUTH COORDINATE DATA')

	joint_coord_gt = []
	viewpoints = []
	for line in input_file:
		line = line.strip()
		line = line.split(',')

		joints_line = list(map(int, line[0:-1]))
		if not new_size == 128:
			joints_line = [x * new_size/128 for x in joints_line]
		joints_line = np.reshape(joints_line, (-1, 2))

		joint_coord_gt.append(joints_line)
		viewpoints.append(float(line[-1]))

	return joint_coord_gt, viewpoints


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_joint_coordinates(demo_img, hm, new_size):
	joint_coord = []
	hm = hm[-1]     # Last Stage
	hm = hm[0]      # First Image
	for joint_id in list(range(0, 15)):
		hm_ind = hm[:, :, joint_id]
		hm_ind = cv2.resize(hm_ind, (new_size, new_size))
		list_index = list(np.unravel_index(np.argmax(hm_ind), hm_ind.shape))
		list_index.reverse()
		joint_coord.append(list_index)

		# Draw circles on original image
		# cv2.circle(demo_img, (joint_coord[joint_id][1], joint_coord[joint_id][0]), 2, (0, 0, 255), -1)

	# cv2.imshow('Image + coords', demo_img)
	return joint_coord


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def plot_limbs(new_size, demo_img, joint_coord, model_name, id=1):
	demo_img = cv2.resize(demo_img, (new_size, new_size))
	demo_img = cv2.cvtColor(demo_img, cv2.COLOR_RGB2BGR)
	limbs = [[0, 1],
			[1, 2],
			[2, 13],
			[3, 4],
			[4, 5],
			[5, 13],
			[6, 7],
			[7, 8],
			[8, 12],
			[9, 10],
			[10, 11],
			[11, 12],
			[12, 13],
			[13, 14]]

	for limb_id in range(len(limbs)):
		x0 = joint_coord[limbs[limb_id][0]][0]
		y0 = joint_coord[limbs[limb_id][0]][1]
		x1 = joint_coord[limbs[limb_id][1]][0]
		y1 = joint_coord[limbs[limb_id][1]][1]

		cv2.line(demo_img,(x0, y0),(x1, y1), (0, 255, 0), thickness=1)

	for joint_id in list(range(0, 15)):
		cv2.circle(demo_img, (joint_coord[joint_id][0], joint_coord[joint_id][1]), 2, (0, 0, 255), -1)

	if (id >= 0) & (id <= 1000):
		cv2.imwrite('results/Angular/' + model_name +  '/(' + str(id) + ') Links.png', demo_img)                                              # CHANGE
		# cv2.imshow('Links', demo_img)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def getPCK(joint_coord_gt, joint_coord_pred, threshold = 5, PCKn = False):

	n_joints = 0
	joints_correct = 0
	joints_ind_correct = [0] * 15
	n_joints_ind = [0] * 15
	PCK_img = []
	neck_length = 0.001
	for img_id in list(range(0, len(joint_coord_gt))):
		joints_correct_per_image = 0
		if PCKn:
			x1 = joint_coord_gt[img_id][1][0]

			y1 = joint_coord_gt[img_id][1][1]
			x2 = joint_coord_gt[img_id][13][0]
			y2 = joint_coord_gt[img_id][13][1]
			neck_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)

		for joint_id in list(range(0,len(joint_coord_gt[0]))):
			dx = joint_coord_gt[img_id][joint_id][0] - joint_coord_pred[img_id][joint_id][0]
			dy = joint_coord_gt[img_id][joint_id][1] - joint_coord_pred[img_id][joint_id][1]
			dist = math.sqrt(dx**2 + dy**2)
			if (dist/neck_length*100 < threshold) and (PCKn is True):
				joints_correct += 1
				joints_ind_correct[joint_id] += 1
				joints_correct_per_image += 1
			if (dist <= threshold) and PCKn is False:
				joints_correct += 1
				joints_ind_correct[joint_id] += 1
				joints_correct_per_image += 1
			n_joints += 1
			n_joints_ind[joint_id] += 1

		# print('n_joints = ' + str(n_joints))
		# print('n_joints_ind = ' + str(n_joints_ind[joint_id]))
		PCK_img.append((joints_correct_per_image/15)*100)

	return joints_correct/n_joints*100, np.divide(joints_ind_correct, n_joints_ind)*100, PCK_img


# - MAIN --------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Config
	ground_truth_available = False
	res_show = 128

	# -- Building the model -------------------------------------------------------------------------
	input_placeholder = tf.placeholder(
		dtype=tf.float32,
		shape=(1, cfg.FLAGS.input_height, cfg.FLAGS.input_width, cfg.FLAGS.input_dim),
		name='input_placeholder')
	model = cpm.CPM_Model(cfg.FLAGS.stages, cfg.FLAGS.num_of_features, cfg)
	model.build_model(input_placeholder, 1)
	print('MODEL BUILT \n')

	# -- Start Session and Predict ------------------------------------------------------------------
	with tf.Session() as sess:
		# tf_w = tf.summary.FileWriter('./log', sess.graph)

		# Create model saver
		saver = tf.train.Saver(max_to_keep=2)
		# sess.run(tf.global_variables_initializer())
		iterations = 1000
		stageritas = cfg.FLAGS.stages
		model_name = 'SingleFE_' + str(stageritas) + 'stage'

		# saver.restore(sess, 'saver/cmu_og_slim/cmu_og_slim_50000iter.ckpt-50000')                                       # OG CPM (3 STAGES)
		# saver.restore(sess, 'saver/cmu_slim_3stages/cmu_slim_3stages_50000iter.ckpt-50000')                             # SINGLE-FE (3 STAGES)
		# saver.restore(sess, 'saver/cmu_slim_3stages_nolocal/cmu_slim_3stages_nolocal_50000iter.ckpt-50000')           # SINGLE-FE NOLOCAL (3 STAGES)
		# saver.restore(sess, 'saver/cmu_timctho_3stages/cmu_timctho_3stages_50000iter.ckpt-50000')                       # VGG16 (3 STAGES)
		# saver.restore(sess, 'saver/cmu_timctho_7stages/cmu_timctho_7stages_50000iter.ckpt-50000')                     # VGG16 (7 STAGES)
		# saver.restore(sess, 'saver/cmu_VerySlim_3stages/VerySlim_CPM.ckpt-10000')                                       # VerySLIM
		# saver.restore(sess, 'saver/cmu_VerySlim_-45to45_3stages/VerySlim_CPM_-45to45.ckpt-1')                         # VerySLIM (RESTR)
		# saver.restore(sess, 'saver/cmu_slim_-45to45_3stages/SingleFE_315to45.ckpt-' + str(iterations))                # SINGLE-FE (RESTR)
		# saver.restore(sess, 'saver/cmu_slim_8stages/cmu_slim_8stages_50000iter.ckpt-' + str(iterations))              # SINGLE-FE (8 STAGES)
		saver.restore(sess, 'saver/Stages/SingleFE_' + str(stageritas) + 'stage.ckpt-1000')                              # SINGLE-FE (8 STAGES)

		if ground_truth_available is True:
			# Obtain LABEL coordinate list
			joint_coord_gt, viewpoints = get_label_coords('data_reader_write/coordinates2D_cr_uniformAngular.csv', res_show)
			list_forloop = list(range(1, len(joint_coord_gt) + 1))
		else:
			list_forloop = list(range(1, 1))

		joint_coord_pred = []
		print('PROCESSING IMAGES')
		start = time.time()
		times = []
		for img_id in list_forloop:
			tt=time.time()
			# - Read and pre-process image [ X ]
			img = cv2.imread('data_reader_write/test - 1000 - UniformAngular/' + str(img_id) + '_sameAR.png')  # dog & dog_7 work well
			img = cv2.imread('ata_reader_write/')
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, (cfg.FLAGS.input_height, cfg.FLAGS.input_height))
			img_m = img * (1 / 255) - 0.5
			img_m = np.expand_dims(img_m, axis=0)

			t0 = time.time()
			# - Predict heatmaps using CNN [ Y = f(X) ]
			stage_heatmap_np = sess.run(model.stage_heatmap[0::], feed_dict={model.input_image: img_m})
			t1 = time.time()

			# - Obtain joint coordinates from heatmaps and print them on image
			joint_coord = get_joint_coordinates(img, stage_heatmap_np, res_show)
			joint_coord_pred.append(joint_coord)
			t2 = time.time()

			# - Show / Print all heatmaps for each stage
			if img_id < 100:
				show_individual_heatmaps(res_show, img, stage_heatmap_np, cv2.COLORMAP_JET, model_name, img_id)

			# - Print links on image
			plot_limbs(res_show, img, joint_coord, model_name, img_id)

			t3 = time.time()

			times.append(t3-t0)
			print('Image ' + str(img_id) + ': ' + str(t3-tt)
			+ '\n\t Time to load and resize image:     ' + str(t0 - tt)
			+ '\n\t Time to sess.run:                  ' + str(t1 - t0)
			+ '\n\t Time to extract joint predictions: ' + str(t2 - t1)
			+ '\n\t Time to save images:               ' + str(t3 - t2))
			# time.sleep(0.00)

		endito = time.time()
		print('Average Inference Time: ' + str(np.mean(times)*1000))
		print('Average Inference Time v2: ' + str((endito-start)))


	# -- Calculate PCK over a range of thresholds ------------------------------------------------------------------
	if ground_truth_available is True:
		x = np.arange(0, 100.01, 1)

		# # PCK-l (normalised leg)
		# pck = []
		# pck_jt = []*15
		# for i in x:
		# 	pck_i, pck_jt_i, pck_img = getPCK(joint_coord_gt, joint_coord_pred, threshold=i, PCKn=True)
		# 	pck.append(pck_i)
		# 	pck_jt.append(pck_jt_i)
		# sio.savemat('PCKl_' + model_name + '.mat', {'PCKl_' + model_name: pck})
		# sio.savemat('PCKl_jt_' + model_name + '.mat', {'PCKl_jt_' + model_name: pck_jt})

		# PCK (pixels)
		pck = []
		pck_jt = [] * 15
		pck_img = [] * 1000
		for i in x:
			pck_i, pck_jt_i, pck_img_i = getPCK(joint_coord_gt, joint_coord_pred, threshold=i, PCKn=False)
			pck.append(pck_i)
			pck_jt.append(pck_jt_i)
			pck_img.append(pck_img_i)
		sio.savemat('matlab_data/' + model_name + '.mat', {'PCK_' + model_name: pck, 'PCK_jt_' + model_name: pck_jt, 'viewpoints': viewpoints, 'PCK_img_' + model_name: pck_img})

	print('Finished.')
