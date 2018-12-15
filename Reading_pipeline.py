# http://zangbo.me/2017/07/05/TensorFlow_9/
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
def read_batch_cpm(tfr_path, input_height, input_width, output_height, output_width, input_dim, output_dim, batch_size):
    # ---- This obtains a list of all images in the TFrecords ----------------------------------------------------------
    tfr_queue = tf.train.string_input_producer(tfr_path, num_epochs=None, shuffle=True)

    # multi_threads reading
    data_list = [
        read_and_decode_cpm(tfr_queue, input_height, input_width, output_height, output_width, input_dim, output_dim)
        for _ in range(2 * len(tfr_path))]
    # -----------------------------------------------------------------------------------------------------------------

    # ---- This shuffles the entire list and extracts a batch of size=batch_size
    batch_images, batch_labels, batch_images_orig = tf.train.shuffle_batch_join(data_list, batch_size=batch_size,
                                                                                capacity=1000 + 6 * batch_size,
                                                                                min_after_dequeue=100,  #100
                                                                                enqueue_many=True,
                                                                                name='batch_data_read')
    # -----------------------------------------------------------------------------------------------------------------
    return batch_images, batch_labels, batch_images_orig


# ----------------------------------------------------------------------------------------------------------------------
def read_and_decode_cpm(tfr_queue, input_height, input_width, output_height, output_width, input_dim, output_dim):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_labels = []
    queue_orig_images = []

    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw': tf.FixedLenFeature([], tf.string), 'hm_raw': tf.FixedLenFeature([], tf.string)})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [1,input_height, input_width, input_dim])
    #print(img)
    #plt.imshow(img)
    # depth_gray=([depth],[depth],[depth])
    # depth = tf.image.rgb_to_grayscale(depth_gray)

    # img = img[..., ::-1]
    # img = tf.image.random_contrast(img, 0.7, 1)
    # img = tf.image.random_brightness(img, max_delta=0.9)
    # img = tf.image.random_hue(img, 0.05)
    # img = tf.image.random_saturation(img, 0.7, 1.1)
    # img = img[..., ::-1]

    hm = tf.decode_raw(features['hm_raw'], tf.float32)
    hm = tf.reshape(hm, [1,output_height, output_width, output_dim])


    queue_images, queue_labels, queue_orig_images = pre_processing(img, hm)
  

    return queue_images, queue_labels, queue_orig_images


# ----------------------------------------------------------------------------------------------------------------------
def pre_processing(img, hm):
    # hmm = tf.cast(hm, tf.float32)
    imgg = tf.cast(img, tf.float32)

    imgg = (imgg/255)-0.5
    hm = hm

    return imgg, hm, img