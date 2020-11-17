import tensorflow as tf

from class_based.util import cross_entropy_loss, accuracy


"""Convolutional layers used by MAML model."""
seed = 123


def conv_block(inp, cweight, bweight, bn, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
    normed = bn(conv_output)
    normed = activation(normed)
    return normed


class ConvLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer = tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1',
                                       dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2',
                                       dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3',
                                       dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4',
                                       dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5',
                                    dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights

    def call(self, inp, weights):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']


"""MAML model code"""
import numpy as np
import tensorflow as tf
from functools import partial


class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        self.dim_hidden = num_filters
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]] * num_inner_updates
        losses_ts_post = [[]] * num_inner_updates
        accuracies_ts = [[]] * num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [
                    tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in
                    range(num_inner_updates)]

    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True,
                            meta_batch_size=25, num_inner_updates=1):
            """
              Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
              Args:
                inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                  labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                  labels used for evaluating the model after inner updates.
                  Should be shapes:
                    input_tr: [N*K, 784]
                    input_ts: [N*K, 784]
                    label_tr: [N*K, N]
                    label_ts: [N*K, N]
              Returns:
                task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            weights = self.conv_layers.conv_weights

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            #############################
            #### YOUR CODE GOES HERE ####
            # perform num_inner_updates to get modified weights
            # modified weights should be used to evaluate performance
            # Note that at each inner update, always use input_tr and label_tr for calculating gradients
            # and use input_ts and labels for evaluating performance

            # HINTS: You will need to use tf.GradientTape().
            # Read through the tf.GradientTape() documentation to see how 'persistent' should be set.
            # Here is some documentation that may be useful:
            # https://www.tensorflow.org/guide/advanced_autodiff#higher-order_gradients
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape

            wl = dict([(key, tf.identity(val)) for key, val in weights.items()])

            # task_output_tr_pre = self.conv_layers(input_tr, wl[0])
            # task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

            for i in range(num_inner_updates):
                with tf.GradientTape() as t:
                    t.watch(wl)
                    t.watch(input_tr)
                    pred_tr = self.conv_layers(input_tr, wl)
                    loss = self.loss_func(label_tr, pred_tr)

                grad = t.gradient(loss, wl)

                if i == 0:
                    task_output_tr_pre = pred_tr
                    task_loss_tr_pre = loss

                for key in wl.keys():
                    if self.learn_inner_update_lr:
                        wl[key] = wl[key] - self.inner_update_lr_dict[key] * grad[key]
                    else:
                        wl[key] = wl[key] - self.inner_update_lr * grad[key]

                pred_ts = self.conv_layers(input_ts, wl)
                ts_loss = self.loss_func(label_ts, pred_ts)
                task_outputs_ts.append(pred_ts)
                task_losses_ts.append(ts_loss)

            #############################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1),
                                            tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1),
                                                   tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre,
                           task_accuracies_ts, wl]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                                 False,
                                 meta_batch_size,
                                 num_inner_updates)
        out_dtype = [tf.float32, [tf.float32] * num_inner_updates, tf.float32, [tf.float32] * num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])
        K, N = meta_batch_size, input_tr.shape[1]
        # inp = (tf.reshape(input_tr, (N * K, 784)), tf.reshape(input_ts, (N * K, 784)), tf.reshape(label_tr, (N*K, N)),tf.reshape(label_ts, (N*K, N)))
        results = []
        for itr, its, ltr, lts in zip(input_tr, input_ts, label_tr, label_ts):
            result = task_inner_loop((itr, its, ltr, lts), meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
            results.append(result)

        wls = [result[-1] for result in results]
        concat_wls = {key: tf.reduce_mean([wl[key] for wl in wls]) for key in wls[0].keys()}
        return (tf.concat([result[i] for result in results], axis=0) for i in range(len(results[0])-1)), concat_wls
