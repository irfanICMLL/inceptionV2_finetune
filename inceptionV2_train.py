#!/usr/local/bin/python
from __future__ import print_function
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import random
import numpy as np
import inception_v2_net
from shutil import copyfile
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.slim.nets import inception


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',                default= './train')
parser.add_argument('--val_dir',                  default= './test')
parser.add_argument('--log_dir',                  default= './logs')
parser.add_argument('--save_ckpt_dir',            default= './logs/saved_model')
parser.add_argument('--code_path',                default= './code_backup.py')

parser.add_argument('--batch_size',               default= 256,            type=int)
parser.add_argument('--num_workers',              default= 32,             type=int)

parser.add_argument('--learning_rate',            default= 0.1,            type=float)
parser.add_argument('--dropout_keep_prob',        default= 0.3,            type=float)
parser.add_argument('--weight_decay',             default= 1e-4,           type=float)
parser.add_argument('--learning_momentum',        default= 0.9,            type=float)
parser.add_argument('--batch_norm_decay_rate',    default= 0.95,           type=float)
parser.add_argument('--label_smoothing',          default= 0.1,            type=float)
parser.add_argument('--color_augm_probability',   default= 0.0,            type=float)
parser.add_argument('--lr_decay_rate',            default= 0.94,           type=float)
parser.add_argument('--clip_gradient_at',         default= 0.0,            type=float)
parser.add_argument('--smallest_side',            default= 256.0,          type=float)

parser.add_argument('--crop',                     default= 224,            type=int)
parser.add_argument('--num_epochs',               default= 999,            type=int)
parser.add_argument('--log_every_iterations',     default= 10,             type=int)
parser.add_argument('--check_val_acc_from_epoch', default= 0,              type=int) 
parser.add_argument('--val_acc_every_n_epochs',   default= 1,              type=int)
parser.add_argument('--batches_to_check',         default= 999,            type=int) # 999 = All the validation set
parser.add_argument('--decrease_lr_every_epoch',  default= 3,              type=int)
parser.add_argument('--train_iter_per_epoch',     default= 999,            type=int) # 999 = All the training set


NUM_TRAIN_IMAGES = 5994

def list_images(directory):
    #Get all the images and labels in directory/label/*.jpg
  
    labels = os.listdir(directory)
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op, batches_to_check):

    # Initialize the validation dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0

    for i in range(batches_to_check):

        try:
            correct_pred = sess.run(correct_prediction,{is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = list_images(args.train_dir)
    val_filenames, val_labels = list_images(args.val_dir)
    MAX_ITERATIONS = NUM_TRAIN_IMAGES // args.batch_size
    num_classes = len(set(train_labels))

    # Copy the code into the log_dir
    copyfile(args.code_path,args.log_dir+'/train_once.py')

    graph = tf.Graph()
    with graph.as_default():

        def apply_with_random_selector(x, func, num_cases):
            sel = tf.random_uniform([],maxval=num_cases,dtype=tf.int32)
            return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]

        def distort_color(image,color_ordering):
            
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

            return tf.clip_by_value(image, 0.0, 1.0)


        # Preprocessing (for both training and validation):
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = args.smallest_side
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)
            scale = tf.cond(tf.greater(height, width),lambda:smallest_side/width,lambda:smallest_side/height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)
            image = tf.image.resize_images(image, [new_height, new_width])  # (2)

            return image, label

        # Preprocessing (for training)
        def training_preprocess(image, label):

            image = tf.image.random_flip_left_right(image)
            image = tf.random_crop(image, [args.crop, args.crop, 3]) 

            image = image / 255.0

            if args.color_augm_probability > 0:
                r = tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32)
                do_color_distortion = tf.less(r,args.color_augm_probability)
                image = tf.cond(do_color_distortion,lambda:tf.identity(apply_with_random_selector(image,lambda x,ordering:distort_color(image,ordering),num_cases=4)),lambda:tf.identity(image))                

            image = image - 0.5
            image = image * 2.0

            return image, label

        # Preprocessing (for validation)
        def val_preprocess(image, label):

            image = tf.image.resize_image_with_crop_or_pad(image, args.crop, args.crop)

            image = image / 255.0 
            image = image - 0.5
            image = image * 2.0

            return image, label

        # ----------------------------------------------------------------------

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function,num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess,num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000) 
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,batched_train_dataset.output_shapes)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        def inception_arg_scope(weight_decay=1e-4,is_training=True,batch_norm_decay=0.95,batch_norm_epsilon=0.001):

            batch_norm_params = {
              'decay': batch_norm_decay,
              'epsilon': batch_norm_epsilon,
              'updates_collections': tf.GraphKeys.UPDATE_OPS,
              'is_training': is_training,
            }
            
            # Set weight_decay for weights in Conv and FC layers.
            with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d],weights_initializer=slim.variance_scaling_initializer(),activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params) as sc:
                    return sc


        with slim.arg_scope(inception_arg_scope(weight_decay=args.weight_decay,is_training=is_training,batch_norm_decay=args.batch_norm_decay_rate)):
            logits  = inception_v2_net.inception_v2(images,num_classes=num_classes,dropout_keep_prob=args.dropout_keep_prob,is_training=is_training)


        #Define the scopes that you want to exclude for restoration
        exclude = ['InceptionV2/Logits/Conv2d_1c_1x1']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        restore_ckpt = tf.contrib.framework.assign_from_checkpoint_fn('./inception_v2.ckpt',variables_to_restore)

        # ---------------------------------------------------------------------

        onehot_labels = tf.one_hot(labels,200,on_value=1.0,off_value=0.0,axis=-1)
        classif_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits,label_smoothing=args.label_smoothing)

        l2_loss = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = tf.losses.get_total_loss()

        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.Variable(args.learning_rate,trainable=False)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate,args.learning_momentum,use_nesterov=True)

        train_step = slim.learning.create_train_op(total_loss,optimizer,global_step=global_step,clip_gradient_norm=args.clip_gradient_at)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates], total_loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits,1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        best_val_accuracy = tf.Variable(0.0,trainable=False)

        tf.summary.scalar('1__total_loss', total_loss)
        tf.summary.scalar('2__train_accuracy', accuracy)
        tf.summary.scalar('3__val_accuracy', best_val_accuracy)
        tf.summary.scalar('4__classif_loss', classif_loss)
        tf.summary.scalar('5__l2_loss', l2_loss)
        tf.summary.scalar('6__learning_rate1', learning_rate)
        tf.summary.scalar('7__learning_rate2', learning_rate)

        # --------------------------------------------------------------------------

        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())
            restore_ckpt(sess)  # load the pretrained weights

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

            best_accuracy = 0.0

            for epoch in range(args.num_epochs):
                print('Starting learning epoch %d / %d' % (epoch, args.num_epochs-1))
                sess.run(train_init_op)

                iteration=0
                for _ in range(args.train_iter_per_epoch):

                    try:
                        _,summary,curr_step = sess.run([train_step,merged,global_step],{is_training:True})
                        train_writer.add_summary(summary,curr_step)

                        if iteration % args.log_every_iterations == 0:
                            print('E:%d/%d It:%d Step:%d Best:%f' % (epoch,args.num_epochs,iteration,curr_step-1,best_accuracy))

                        iteration += 1

                    except tf.errors.OutOfRangeError:
                        break

                if (epoch % args.decrease_lr_every_epoch == 0 and epoch is not 0):
                    curr_lr = sess.run(learning_rate,{is_training:False})
                    sess.run(tf.assign(learning_rate,curr_lr * args.lr_decay_rate))

                if (epoch >= args.check_val_acc_from_epoch) and ((epoch % args.val_acc_every_n_epochs) == 0):
                    curr_val_acc = check_accuracy(sess,correct_prediction,is_training,val_init_op,args.batches_to_check)
                    print('Val accuracy: %f\n' % curr_val_acc)

                    if curr_val_acc > best_accuracy:
                        curr_step = sess.run(global_step,{is_training:False})
                        saver.save(sess,args.save_ckpt_dir,global_step=curr_step)
                        best_accuracy = curr_val_acc
                        sess.run(tf.assign(best_val_accuracy,best_accuracy))

            train_writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
