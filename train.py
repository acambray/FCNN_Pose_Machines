# IMPORTS --------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import configs.config as cfg
import models.cpm_cmu_superslim as cpm
import Reading_pipeline as rp
import time
# training on the server
if cfg.FLAGS.isbc is False:
    import cv2


# ----------------------------------------------------------------------------------------------------------------------
def main(argv):
    # Pre-processing ---------------------------------------------------------------------------------------------------
    if cfg.FLAGS.isbc is False:

        batch_x, batch_y, batch_x_orig = rp.read_batch_cpm(cfg.FLAGS.TFrecords_dir,
                                                           cfg.FLAGS.input_height,
                                                           cfg.FLAGS.input_width,
                                                           cfg.FLAGS.output_height,
                                                           cfg.FLAGS.output_width,
                                                           cfg.FLAGS.input_dim,
                                                           cfg.FLAGS.output_dim,
                                                           cfg.FLAGS.batch_size)
    if cfg.FLAGS.isbc is True:
        tf_dir='train.tfrecords'
        batch_x, batch_y, batch_x_orig = rp.read_batch_cpm(tf_dir,
                                                           cfg.FLAGS.input_height,
                                                           cfg.FLAGS.input_width,
                                                           cfg.FLAGS.output_height,
                                                           cfg.FLAGS.output_width,
                                                           cfg.FLAGS.input_dim,
                                                           cfg.FLAGS.output_dim,
                                                           cfg.FLAGS.batch_size)
    # Make Input Placeholder -------------------------------------------------------------------------------------------
    input_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(cfg.FLAGS.batch_size, cfg.FLAGS.input_height, cfg.FLAGS.input_width,
                                              cfg.FLAGS.input_dim), name='input_placeholer')

    hm_placeholder = tf.placeholder(dtype=tf.float32,
                                    shape=(cfg.FLAGS.batch_size, cfg.FLAGS.output_height, cfg.FLAGS.output_width,
                                           cfg.FLAGS.output_dim), name='hm_placeholder')
    # Building the model -----------------------------------------------------------------------------------------------
    print('# BUILDING MODEL --------------------------------------------------------')
    model = cpm.CPM_Model(cfg.FLAGS.stages, cfg.FLAGS.num_of_features, cfg)
    model.build_model(input_placeholder, cfg.FLAGS.batch_size)
    model.build_loss(hm_placeholder, cfg.FLAGS.lr, cfg.FLAGS.lr_decay_rate, cfg.FLAGS.lr_decay_step)

    # TRAINING ---------------------------------------------------------------------------------------------------------
    with tf.Session() as sess:

        coord = tf.train.Coordinator()                                                              # good
        threads = tf.train.start_queue_runners(coord=coord)                                         # good
        tf_w = tf.summary.FileWriter('./log', sess.graph)                                           # good

        # Create model saver
        print('# CREATING SAVER AND INITIALISING --------------------------------------------')
        saver = tf.train.Saver(max_to_keep=5)                                                       # good

        sess.run(tf.global_variables_initializer())                                                 # good
        # saver.restore(sess, 'saver/cmu_superslim_3stages/cmu_superslim_3stages.ckpt-1001')                                # good

        print('# BEGINNING TRAINING --------------------------------------------------------')
        while True:
            # Read in batch data
            t0 = time.time()
            batch_x_np, batch_y_np, batch_x_orig_np = sess.run([batch_x, batch_y, batch_x_orig])
            t1 = time.time()
            print('\n\nTime to sess.run: \t' + str(t1 - t0))
            total_loss_np, stage_loss, _, summary, current_lr, stage_heatmap, global_step = model.model_train(batch_x_np,
                                                                                                  batch_y_np,
                                                                                                  sess)
            t2 = time.time()

            print('Time to model.train\t: ' + str(t2 - t1))
            # Write logs
            tf_w.add_summary(summary, global_step)

            # Draw intermediate results
            visualizing_training(global_step, stage_heatmap, batch_x_orig_np, batch_y_np)

            print('------- Iteration {:>6d} ---------------------'.format(global_step))
            print('Current learning rate: {:.8f}'.format(current_lr))
            print('Total loss: {:>.3f}'.format(total_loss_np))
            for stage in range(cfg.FLAGS.stages):
                print('\t- Stage {} loss: {:>.3f}'.format(stage + 1, stage_loss[stage]))

            if global_step % 500 == 1:
                save_path_str = 'saver/' + cfg.FLAGS.model_name + '.ckpt'
                saver.save(sess=sess, save_path=save_path_str, global_step=global_step)
                print('\nModel checkpoint saved...\n')

            # Finish training
            if global_step == cfg.FLAGS.training_iterations:
                break

        coord.request_stop()
        coord.join(threads)
        print('Training done.')


# ----------------------------------------------------------------------------------------------------------------------
def visualizing_training(global_step, stage_heatmap, batch_x_orig_np, batch_y_np):
    if global_step % 1 == 0:
        colormap = cv2.COLORMAP_JET

        # - INPUT IMAGE ---------------------------------------------------------------------------------------------
        demo_img = (batch_x_orig_np[0])                         # Gets last image from Batch
        demo_img = cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB)
        demo_img_256 = cv2.resize(demo_img, (256, 256))
        # -----------------------------------------------------------------------------------------------------------

        # - LABEL HEATMAPS ------------------------------------------------------------------------------------------
        demo_gt_heatmap = batch_y_np[0].reshape(
            cfg.FLAGS.output_height, cfg.FLAGS.output_width, cfg.FLAGS.output_dim)
        demo_gt_heatmap = np.sum(demo_gt_heatmap, axis=-1)                  # COLLATES ALL JOINTS TOGETHER

        batch_y_np = np.sum(batch_y_np[0], axis=-1)                         # GETS FIRST IMAGE

        batch_y_np_256 = cv2.resize(batch_y_np, (256, 256))                                     # Resize GT HP
        batch_y_np_256 = cv2.normalize(batch_y_np_256, None, 0, 255, cv2.NORM_MINMAX)           # Re-scale to [0-255]
        batch_y_np_256 = batch_y_np_256.astype('uint8')                                         # Convert to UNIT8
        batch_y_np_256 = cv2.applyColorMap(batch_y_np_256, colormap)                    # Apply JET Colormap
        # -----------------------------------------------------------------------------------------------------------

        # - OUTPUT INDIVIDUAL HEATMAPS ------------------------------------------------------------------------------
        new_size = 256
        image_blend = cv2.resize(demo_img, (new_size, new_size))
        # image_blend = image_blend.astype('uint8')
        hm = stage_heatmap
        hm = hm[-1]             # SELECTS LAST STAGE
        hm = hm[0]              # SELECTS FIRST IMAGE
        hm_list = []
        hmb_list = []
        hm_stack = []
        joint_name = ['FL Paw', 'FL wrist', 'FL Elbow',
                      'FR Paw', 'FR wrist', 'FR Elbow',
                      'BL Paw', 'BL wrist', 'BL Elbow',
                      'BR Paw', 'BR wrist', 'BR Elbow',
                      'Hip', 'Shoulder', 'Face']
        joint_coord_pred = []
        for joint_id in list(range(0, 15)):
            hm_ind = hm[:, :, joint_id]
            hm_ind = cv2.resize(hm_ind, (new_size, new_size))
            joint_coord_pred.append(list(np.unravel_index(np.argmax(hm_ind), hm_ind.shape)))
            hm_ind = cv2.normalize(hm_ind, None, 0, 255, cv2.NORM_MINMAX)
            hm_ind = hm_ind.astype('uint8')
            hmb_ind = cv2.applyColorMap(hm_ind, colormap)
            hm_blend = cv2.addWeighted(image_blend, 0.65, hmb_ind, 0.35, 0, None)

            # Add joint name and joint circle
            cv2.putText(hm_blend, joint_name[joint_id], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            cv2.circle(hm_blend, (joint_coord_pred[joint_id][1], joint_coord_pred[joint_id][0]), 6, (255, 255, 255), 1)

            hm_list.append(hm_ind)
            hmb_list.append(hm_blend)

        # Stack grayscale heatmaps
        hm_stack = np.hstack(hm_list)
        # Convert grayscale stack to JET
        hm_stack_jet = cv2.applyColorMap(hm_stack, colormap)
        # Stack blended image+heatmaps
        hmb_stack = np.hstack(hmb_list)
        # print('fullstack: ' + str(np.shape(hmb_stack)))

        row1 = hmb_stack[:, 0:new_size * 6, :]
        row2 = hmb_stack[:, new_size * 6:new_size * 12, :]
        row3 = hmb_stack[:, new_size * 12::, :]
        black_square = np.zeros((new_size, new_size * 3, 3), np.int8)
        row3 = np.hstack([row3, black_square])

        hmb_fullstack = np.vstack((row1, row2, row3))
        hmb_fullstack = hmb_fullstack.astype('uint8')

        cv2.imshow('FULLSTACK - blend', hmb_fullstack)
        cv2.moveWindow("Stack - GRAY", 0, new_size * 4)
        # -----------------------------------------------------------------------------------------------------------

        # - OUTPUT COMBINED HEATMAP ---------------------------------------------------------------------------------
        output_heatmap = np.sum(stage_heatmap,axis=-1)                      # COLLATES ALL JOINTS TOGETHER

        output_heatmap = output_heatmap[-1]                                 # SELECTS LAST STAGE

        output_heatmap = output_heatmap[0]                                  # SELECTS FIRST IMAGE

        output_heatmap_256 = cv2.resize(output_heatmap,(256,256))                               # Resize GT HP
        output_heatmap_256 = cv2.normalize(output_heatmap_256, None, 0, 255, cv2.NORM_MINMAX)   # Normalise to [0-255]
        output_heatmap_256 = output_heatmap_256.astype('uint8')                                 # Convert to UNIT8
        output_heatmap_256 = cv2.applyColorMap(output_heatmap_256, colormap)                    # Apply JET Colormap
        # -----------------------------------------------------------------------------------------------------------

        # SHOW ALL IMAGES
        blended = cv2.addWeighted(demo_img_256, 0.65, batch_y_np_256, 0.35, 0, None)
        cv2.imshow('Image', demo_img_256);              cv2.moveWindow("Image", 1600, 10)         # IMAGE
        cv2.imshow('GT Heatmap', batch_y_np_256);       cv2.moveWindow("GT Heatmap", 1600, 360)   # GROUND TRUTH HEATMAP
        cv2.imshow('Out Heatmap', output_heatmap_256);  cv2.moveWindow("Out Heatmap", 1600, 730)  # PREDICTED HEATMAP
        cv2.imshow('Blended', blended);                 cv2.moveWindow("Blended", 1250, 10)       # BLENDED
        cv2.waitKey(100)


if __name__ == '__main__':
    tf.app.run()
