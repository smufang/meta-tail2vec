import os
import numpy as np
import argparse
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(model, sess):
    prelosses, postlosses = [], []

    for iteration in range(50000):
        ops = [model.meta_op]

        if iteration % 20 == 0:
            ops.extend([model.summ_op, model.query_losses[0], model.query_losses[-1]])

        result = sess.run(ops)
        if iteration % 20 == 0:
            prelosses.append(result[2])
            postlosses.append(result[3])
            print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses))
            prelosses, postlosses = [], []


def test(model, sess, dataset_name):
    test_preds = []
    fw_5 = open('./data/' + dataset_name + '/result_5.csv', 'w')
    fw_3 = open('./data/' + dataset_name + '/result_3.csv', 'w')
    fw_1 = open('./data/' + dataset_name + '/result_1.csv', 'w')
    for i in range(250):
        if i % 100 == 1:
            print(i)
        ops = [model.test_query_preds, model.query_nodes]
        result, nodes = sess.run(ops)
        for n in range(4):
            fw_1.write(str(int(nodes[0][n][0])) + ' ')
            temp = [str(x) for x in result[0][n].tolist()]
            fw_1.write(' '.join(temp))
            fw_1.write('\n')
        for n in range(4):
            fw_3.write(str(int(nodes[2][n][0])) + ' ')
            temp = [str(x) for x in result[2][n].tolist()]
            fw_3.write(' '.join(temp))
            fw_3.write('\n')
        for n in range(4):
            fw_5.write(str(int(nodes[4][n][0])) + ' ')
            temp = [str(x) for x in result[4][n].tolist()]
            fw_5.write(' '.join(temp))
            fw_5.write('\n')
    fw_1.close()
    fw_3.close()
    fw_5.close()
    print('Done.')


def main():
    training = not args.test
    main_dir = './data/'
    dataset_name = 'flickr'
    kshot = 5
    meta_batchsz = 4
    k = 5

    db = DataGenerator(main_dir, dataset_name, kshot, meta_batchsz, 50000)
    if training:
        node_tensor, label_tensor, data_tensor = db.make_data_tensor(training=True)
        support_n = tf.slice(node_tensor, [0, 0, 0], [-1, kshot, -1], name='support_n')
        query_n = tf.slice(node_tensor, [0, kshot, 0], [-1, -1, -1], name='query_n')
        support_x = tf.slice(data_tensor, [0, 0, 0], [-1, kshot, -1], name='support_x')
        query_x = tf.slice(data_tensor, [0, kshot, 0], [-1, -1, -1], name='query_x')
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, kshot, -1], name='support_y')
        query_y = tf.slice(label_tensor, [0, kshot, 0], [-1, -1, -1], name='query_y')

    node_tensor, label_tensor, data_tensor = db.make_data_tensor(training=False)
    support_n_test = tf.slice(node_tensor, [0, 0, 0], [-1, kshot, -1], name='support_n_test')
    query_n_test = tf.slice(node_tensor, [0, kshot, 0], [-1, -1, -1], name='query_n_test')
    support_x_test = tf.slice(data_tensor, [0, 0, 0], [-1, kshot, -1], name='support_x_test')
    query_x_test = tf.slice(data_tensor, [0, kshot, 0], [-1, -1, -1], name='query_x_test')
    support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1, kshot, -1], name='support_y_test')
    query_y_test = tf.slice(label_tensor, [0, kshot, 0], [-1, -1, -1], name='query_y_test')

    model = MAML(128)

    model.build(support_n, support_x, support_y, query_n, query_x, query_y, k, meta_batchsz, mode='train')
    model.build(support_n_test, support_x_test, support_y_test, query_n_test, query_x_test, query_y_test, k,
                meta_batchsz, mode='test')
    model.summ_op = tf.summary.merge_all()

    all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
    for p in all_vars:
        print(p)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if os.path.exists(os.path.join('ckpt', 'checkpoint')):
        model_file = tf.train.latest_checkpoint('ckpt')
        print("Restoring model weights from ", model_file)
        saver.restore(sess, model_file)

    train(model, sess)
    test(model, sess, dataset_name)


if __name__ == "__main__":
    main()
