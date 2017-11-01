# -*- coding: utf-8 -*-
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib as request
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify' + filename + '.Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print('Data size', len(words))

vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict(zip(list(zip(*count))[0], range(len(list(zip(*count))[0]))))
    data = list()
    un_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            un_count += 1
        data.append(index)
    count[0][1] = un_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, reverse_dictionary, dictionary, count


data, reverse_dictionary, dictionary, count = build_dataset(words)
del words

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert num_skips <= 2 * skip_window
    assert batch_size % num_skips == 0
    span = 2 * skip_window + 1
    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    # 初始化
    for i in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 移动窗口，获取批量数据
    for i in range(batch_size // num_skips):
        target = skip_window
        avoid_target = [skip_window]
        for j in range(num_skips):
            while target in avoid_target:
                target = np.random.randint(0, span - 1)
            avoid_target.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

with tf.Graph().as_default() as graph:
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels, inputs=embed,
                                             num_sampled=num_sampled, num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

    num_steps = 100001

    with tf.Session(graph=graph) as session:
        init.run()
        print("initialized")

        average_loss = 0.0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step", step, ":", average_loss)
                average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()