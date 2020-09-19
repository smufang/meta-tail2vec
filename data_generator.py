import numpy as np
import os
import random
import tensorflow as tf
import tqdm
import math
import csv


def cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings):
    weights = np.ndarray(shape=(len(travel),))
    index = 0
    for j in range(len(hop_num_list)):
        for k in range(hop_num_list[j]):
            weights[index] = math.exp(p_lambda * j)
            index += 1
    norm_weights = weights / weights.sum()
    index = 0
    temp_embeddings = np.zeros(shape=(len(travel), 128))
    for node in travel:
        temp_embeddings[index] = np.array(deepwalk_embeddings[node]).astype(np.float)
        index += 1
    embeddings = np.sum(np.multiply(temp_embeddings, norm_weights.reshape((-1, 1))), axis=0)
    return embeddings.tolist()


def generate_support_data(graph, source, deepwalk_embeddings, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if len(graph[frontier]) > node_size:
                node_children = np.random.choice(graph[frontier], node_size, replace=False)
            else:
                node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings)
    return deepwalk_embeddings[source], feature_embedding


def generate_query_data(graph, source, deepwalk_embeddings, s_n, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if travel_hop == 1:
                node_children = s_n
            else:
                if len(graph[frontier]) > node_size:
                    node_children = np.random.choice(list(graph[frontier]), node_size, replace=False)
                else:
                    node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings)
    return deepwalk_embeddings[source], feature_embedding


def write_task_to_file(s_n, q_n, g, emb, hop, size, p_lambda):
    task_data = []
    blank_row = [0.] * 257
    s_index = 0
    for n in s_n:
        oracle_embedding, embedding = generate_support_data(g, n, emb, hop=hop, node_max_size=size, p_lambda=p_lambda)
        task_data.append(list(n.split()) + oracle_embedding + embedding)
        s_index += 1
    while s_index < 5:
        task_data.append(blank_row)
        s_index += 1
    for n in q_n:
        oracle_embedding, embedding = generate_query_data(g, n, emb, s_n, hop=hop, node_max_size=size,
                                                          p_lambda=p_lambda)
        task_data.append(list(n.split()) + oracle_embedding + embedding)
    return task_data


class DataGenerator:
    def __init__(self, main_dir, dataset_name, kshot, meta_batchsz, total_batch_num=200):
        self.main_dir = main_dir
        self.kshot = kshot
        self.meta_batchsz = meta_batchsz
        self.total_batch_num = total_batch_num
        self.dataset_name = dataset_name
        self.hop = 2
        self.size1 = 50
        self.size2 = 25
        self.p_lambda = 0

        self.metatrain_file = self.main_dir + dataset_name + '/train.csv'
        self.metatest_file = self.main_dir + dataset_name + '/test.csv'

        self.graph_dir = self.main_dir + dataset_name + '/graph.adjlist'
        self.graph_dense_dir = self.main_dir + dataset_name + '/graph_dense.adjlist'
        self.emb_dir = self.main_dir + dataset_name + '/graph.embeddings'

        self.graph = dict()
        with open(self.graph_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.graph[temp[0]] = list()
                for n in range(1, len(temp)):
                    self.graph[temp[0]].append(temp[n])
        self.graph_dense = dict()
        with open(self.graph_dense_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.graph_dense[temp[0]] = set()
                for n in range(1, len(temp)):
                    self.graph_dense[temp[0]].add(temp[n])
        self.deepwalk_emb = dict()
        with open(self.emb_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.deepwalk_emb[temp[0]] = temp[1:]

    def make_data_tensor(self, training=True):
        num_total_batches = self.total_batch_num
        if training:
            file = self.metatrain_file
        else:
            file = self.metatest_file

        if training:
            if os.path.exists('./data/' + self.dataset_name + '/trainfile.csv'):
                pass
            else:
                all_data = []
                train_nodes = []
                with open(file, "r") as fr:
                    lines = fr.readlines()
                    for line in lines:
                        temp = list(line.strip('\n').split(','))
                        train_nodes.append(temp[0])
                for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                    query_node = random.sample(train_nodes, 1)
                    print(query_node)
                    support_node = random.sample(self.graph_dense[query_node[0]], self.kshot)
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.deepwalk_emb, self.hop,
                                                   (self.size1, self.size2), self.p_lambda)
                    all_data.extend(task_data)

                with open('./data/' + self.dataset_name + '/trainfile.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save train file list to trainfile.csv')
        else:
            if os.path.exists('./data/' + self.dataset_name + '/testfile.csv'):
                pass
            else:
                all_data = []
                test_nodes = []
                other_nodes = []
                with open(file, "r") as fr:
                    lines = fr.readlines()
                    for line in lines:
                        temp = list(line.strip('\n').split(','))
                        test_nodes.append(temp[0])
                for n in tqdm.tqdm(test_nodes, 'generating test episodes'):
                    query_node = list()
                    query_node.append(n)
                    print(query_node)
                    support_node = self.graph[query_node[0]]
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.deepwalk_emb,
                                                   self.hop, (self.size1, self.size2), self.p_lambda)
                    all_data.extend(task_data)
                with open('./data/' + self.dataset_name + '/testfile.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save test file list to testfile.csv')

        print('creating pipeline ops')
        if training:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + '/trainfile.csv'],
                                                            shuffle=False)
        else:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + '/testfile.csv'],
                                                            shuffle=False)
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        record_defaults = [0.] * 257
        row = tf.decode_csv(value, record_defaults=record_defaults)
        feature_and_label = tf.stack(row)

        print('batching data')
        examples_per_batch = 1 + self.kshot
        batch_data_size = self.meta_batchsz * examples_per_batch
        features = tf.train.batch(
            [feature_and_label],
            batch_size=batch_data_size,
            num_threads=1,
            capacity=256,
        )
        all_node_id = []
        all_label_batch = []
        all_feature_batch = []
        for i in range(self.meta_batchsz):
            data_batch = features[i * examples_per_batch:(i + 1) * examples_per_batch]
            node_id, label_batch, feature_batch = tf.split(data_batch, [1, 128, 128], axis=1)
            all_node_id.append(node_id)
            all_label_batch.append(label_batch)
            all_feature_batch.append(feature_batch)
        all_node_id = tf.stack(all_node_id)
        all_label_batch = tf.stack(all_label_batch)
        all_feature_batch = tf.stack(all_feature_batch)
        return all_node_id, all_label_batch, all_feature_batch
