from __future__ import print_function
from LIP_model import *
from utils import *
import tensorflow as tf
import os
import cv2
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Hide the warning messages about CPU/GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_CLASSES = 20
INPUT_SIZE = (384, 384)
DATA_DIRECTORY = 'datasets/examples/images'
# DATA_DIRECTORY = './datasets/examples'
DATA_LIST_PATH = 'datasets/examples/list/val.txt'
# DATA_LIST_PATH = './datasets/examples/list/val.txt'
NUM_STEPS = len(os.listdir(DATA_DIRECTORY))  # Number of images in the validation set.
RESTORE_FROM = './checkpoint/JPPNet-s2'
# RESTORE_FROM = './checkpoint/JPPNet-s2-pretrained'

    
CM_OUTPUT_DIR = 'outputs/cloth-masks'
if not os.path.exists(CM_OUTPUT_DIR):
    os.makedirs(CM_OUTPUT_DIR)
    
SM_OUTPUT_DIR = 'outputs/seg-masks'
if not os.path.exists(SM_OUTPUT_DIR):
    os.makedirs(SM_OUTPUT_DIR)
    
C_OUTPUT_DIR = 'outputs/cloths'
if not os.path.exists(C_OUTPUT_DIR):
    os.makedirs(C_OUTPUT_DIR)


def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH,
                             None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(
        image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(
        image_batch_origin, [int(h * 1.25), int(w * 1.25)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch},
                              is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075},
                              is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125},
                              is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(
            pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(
            parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(
            parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(
            pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(
            parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(
            parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(
            pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(
            parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(
            parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')

    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(
                                                parsing_out1_075, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(
                                                parsing_out2_075, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(
                                                parsing_out3_075, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)

    raw_output = tf.reduce_mean(
        tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(
        tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        try:
            parsing_ = sess.run(pred_all)
            if step % 1 == 0:
                print('step {:d}'.format(step))
                print(image_list[step])
            img_split = image_list[step].split('/')
            img_id = img_split[-1][:-4]
            
            # Merge all upper cloths categories into one class
            condition = np.isin(parsing_, [6, 7, 10])
            parsing_ = np.where(condition, 5, parsing_)
            
            # Saving segmentation mask
            '''seg_mask = Image.fromarray(np.asarray(parsing_[0, :, :, 0], dtype=np.uint8), mode='P')
            palette = np.array(utils.lip_label_colours).reshape(20*3,)
            seg_mask.putpalette(palette)
            seg_mask.save('{}/{}.png'.format(SM_OUTPUT_DIR, img_id))
            '''
            
            msk = decode_labels(parsing_, num_classes=N_CLASSES)
            parsing_im = Image.fromarray(msk[0])
            parsing_im.save('{}/{}.png'.format(SM_OUTPUT_DIR, img_id))
            
            
            # Saving cloth-mask
            cm_parsing_ = np.where(parsing_ == 5, parsing_, 0)
            cm_msk = decode_labels(cm_parsing_, num_classes=N_CLASSES, extra_param=True)
            cm_parsing_im = Image.fromarray(cm_msk[0]).convert('L')
            #cm_parsing_im.show()
            cm_parsing_im.save('{}/{}.jpg'.format(CM_OUTPUT_DIR, img_id))
            
            
            # Saving cloths
            up_cloths_mask = np.where(parsing_ == 5, True, False)
            up_cloths_mask_bd = np.broadcast_to(up_cloths_mask, shape=list(up_cloths_mask.shape[:-1]) + [3])
            up_cloths_mask_bd = np.squeeze(up_cloths_mask_bd, axis=0)
            ori_img = np.array(Image.open(image_list[step]))
            b_img = np.ones(shape=list(ori_img.shape[:-1]) + [3], dtype=ori_img.dtype) * 255
            ori_img = np.where(up_cloths_mask_bd, ori_img, b_img)
            ori_img = Image.fromarray(ori_img)
            ori_img.save('{}/{}.jpg'.format(C_OUTPUT_DIR, img_id))
            
            '''cv2.imwrite('{}/{}.png'.format(OUTPUT_DIR, img_id),
                        parsing_[0, :, :, 0])'''
        except Exception as err:
            print(err)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
