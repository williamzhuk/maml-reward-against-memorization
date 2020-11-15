import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from class_based.dataloader import DataGenerator, get_datagen_as_dataset
from class_based.maml import MAML
import pickle


"""Model training code"""
"""
Usage Instructions:
  5-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
  20-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
  To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""


def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1, g_clip=0., g_weight=0.):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates, g_clip=g_clip,
                       g_weight=g_weight)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def meta_train_fn(model, exp_string, data_generator: DataGenerator,
                  n_way=5, meta_train_iterations=10000, meta_batch_size=25,
                  log=True, logdir='/tmp/data', k_shot=1, num_inner_updates=1, meta_lr=0.001, g_clip=0., g_weight=0.):
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    pre_accuracies, post_accuracies = [], []
    accs_itrs = []
    pre_accs_avgs = []
    post_accs_avgs = []
    val_accs = []
    val_itrs = []

    num_classes = data_generator.num_classes

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    dataset = get_datagen_as_dataset(data_generator, 'meta_train', meta_batch_size)

    for itr, (image_batches, label_batches) in zip(range(meta_train_iterations), dataset):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of training data and partition into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.

        # image_batches, label_batches = data_generator.sample_batch('meta_train', meta_batch_size)

        input_tr, label_tr = image_batches[:, :, :k_shot], label_batches[:, :, :k_shot]
        input_ts, label_ts = image_batches[:, :, k_shot:], label_batches[:, :, k_shot:]

        input_tr = tf.reshape(input_tr, ([meta_batch_size, n_way * k_shot, -1]))
        label_tr = tf.reshape(label_tr, ([meta_batch_size, n_way * k_shot, n_way]))
        input_ts = tf.reshape(input_ts, ([meta_batch_size, n_way * k_shot, -1]))
        label_ts = tf.reshape(label_ts, ([meta_batch_size, n_way * k_shot, n_way]))

        #############################

        inp = (input_tr, input_ts, label_tr, label_ts)

        result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size,
                                  num_inner_updates=num_inner_updates, g_clip=g_clip, g_weight=g_weight)

        if itr % SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            post_accuracies.append(result[-1][-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            pre_accs_avgs.append(np.mean(pre_accuracies))
            post_accs_avgs.append(np.mean(post_accuracies))
            accs_itrs.append(itr)
            print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (
                itr, pre_accs_avgs[-1], post_accs_avgs[-1])

            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            #############################
            #### YOUR CODE GOES HERE ####

            # sample a batch of validation data and partition it into
            # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
            # NOTE: The code assumes that the support and query sets have the same number of examples.

            image_batches, label_batches = data_generator.sample_batch('meta_val', meta_batch_size)
            input_tr, label_tr = image_batches[:, :, :k_shot], label_batches[:, :, :k_shot]
            input_ts, label_ts = image_batches[:, :, k_shot:], label_batches[:, :, k_shot:]

            input_tr = tf.reshape(input_tr, ([meta_batch_size, n_way * k_shot, -1]))
            label_tr = tf.reshape(label_tr, ([meta_batch_size, n_way * k_shot, n_way]))
            input_ts = tf.reshape(input_ts, ([meta_batch_size, n_way * k_shot, -1]))
            label_ts = tf.reshape(label_ts, ([meta_batch_size, n_way * k_shot, n_way]))

            #############################

            inp = (input_tr, input_ts, label_tr, label_ts)
            result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            print(
                'Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (
                    result[-2], result[-1][-1]))
            val_itrs.append(itr)
            val_accs.append(result[-1][-1])

    model_file = logdir + '/' + exp_string + '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)
    plt.plot(val_itrs, val_accs, label='maml_val')
    plt.plot(accs_itrs, pre_accs_avgs, label='pre_acc')
    plt.plot(accs_itrs, post_accs_avgs, label='post_acc')


# calculated for omniglot
NUM_META_TEST_POINTS = 600


def meta_test_fn(model, data_generator, n_way=5, meta_batch_size=25, k_shot=1,
                 num_inner_updates=1):
    num_classes = data_generator.num_classes

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []

    for _ in range(NUM_META_TEST_POINTS):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of test data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.

        image_batches, label_batches = data_generator.sample_batch('meta_test', meta_batch_size)
        input_tr, label_tr = image_batches[:, :, :k_shot], label_batches[:, :, :k_shot]
        input_ts, label_ts = image_batches[:, :, k_shot:], label_batches[:, :, k_shot:]

        input_tr = tf.reshape(input_tr, ([meta_batch_size, n_way * k_shot, -1]))
        label_tr = tf.reshape(label_tr, ([meta_batch_size, n_way * k_shot, n_way]))
        input_ts = tf.reshape(input_ts, ([meta_batch_size, n_way * k_shot, -1]))
        label_ts = tf.reshape(label_ts, ([meta_batch_size, n_way * k_shot, n_way]))

        #############################
        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96 * stds / np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))
    return means, stds, ci95


def run_maml(n_way=5, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             learn_inner_update_lr=True, resume=False, resume_itr=0, log=True, logdir='/tmp/data',
             data_path='./omniglot_resized', meta_train=True,
             num_train_chars=1100, num_val_chars=100,
             meta_train_iterations=10000, meta_train_k_shot=-1, meta_train_inner_update_lr=-1, g_clip=0., g_weight=0.):
    # call data_generator and get data with k_shot*2 samples per class
    data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot * 2, config={'data_folder': data_path},
                                   num_train_chars=num_train_chars, num_val_chars=num_val_chars)

    # set up MAML model
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    model = MAML(dim_input,
                 dim_output,
                 num_inner_updates=num_inner_updates,
                 inner_update_lr=inner_update_lr,
                 k_shot=k_shot,
                 num_filters=num_filters,
                 learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_' + str(n_way) + '.mbs_' + str(meta_batch_size) + '.k_shot_' + str(
        meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(
        meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

    if meta_train:
        return meta_train_fn(model, exp_string, data_generator,
                             n_way, meta_train_iterations, meta_batch_size, log, logdir,
                             k_shot, num_inner_updates, meta_lr, g_clip=g_clip, g_weight=g_weight)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        return meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    #

    # run currrrently doinng (.01, 1), (.1, 1), (1, 1), (10, 1), (100, 1)
    # other run (0, 0), (1, .01),  (1, .1) (1, 10), (1, 100)

    for g_clip, g_weight in [(-1, 50)]:
        plt.clf()
        num_train_chars = 128
        num_filters = 64
        run_maml(n_way=20,
                 k_shot=1,
                 num_inner_updates=1,
                 num_filters=num_filters,
                 num_train_chars=num_train_chars,
                 meta_batch_size=20,
                 g_clip=g_clip,
                 g_weight=g_weight
                 )
        plt.legend()
        plt.savefig('maml-nf=%d,nt=%d, gc=%.2f, gw=%.2f, custom.png' %
                    (num_filters, num_train_chars, g_clip, g_weight), format='png', dpi=600)
