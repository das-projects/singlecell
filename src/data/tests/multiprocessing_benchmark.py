import sys
import time
from collections import defaultdict
from multiprocessing import Pool, Process

import numpy as np
import psutil
import scipy.signal
import tensorflow as tf

num_trials = 5

# Count the number of physical CPUs.
num_cpus = psutil.cpu_count(logical=False)
print('Using {} cores.'.format(num_cpus))


################################################
###### Benchmark 1: numerical computation ######
################################################


def f(args):
    image, random_filter = args
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]


pool = Pool(num_cpus)

filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]


def run_benchmark():
    image = np.zeros((3000, 3000))
    pool.map(f, zip(num_cpus * [image], filters))


durations1 = []
for _ in range(num_trials):
    start_time = time.time()

    run_benchmark()

    duration1 = time.time() - start_time
    durations1.append(duration1)
    print('Numerical computation workload took {} seconds.'.format(duration1))


###############################################
###### Benchmark 2: stateful computation ######
###############################################


def accumulate_prefixes(args):
    running_prefix_count, running_popular_prefixes, document = args
    for word in document:
        for i in range(1, len(word)):
            prefix = word[:i]
            running_prefix_count[prefix] += 1
            if running_prefix_count[prefix] > 3:
                running_popular_prefixes.add(prefix)
    return running_prefix_count, running_popular_prefixes


pool = Pool(num_cpus)

durations2 = []
for _ in range(num_trials):
    running_prefix_counts = [defaultdict(int) for _ in range(4)]
    running_popular_prefixes = [set() for _ in range(4)]

    start_time = time.time()

    for i in range(10):
        documents = [[np.random.bytes(20) for _ in range(10000)]
                     for _ in range(num_cpus)]
        results = pool.map(
            accumulate_prefixes,
            zip(running_prefix_counts, running_popular_prefixes, documents))
        running_prefix_counts = [result[0] for result in results]
        running_popular_prefixes = [result[1] for result in results]

    popular_prefixes = set()
    for prefixes in running_popular_prefixes:
        popular_prefixes |= prefixes

    duration2 = time.time() - start_time
    durations2.append(duration2)
    print('Stateful computation workload took {} seconds.'.format(duration2))


###################################################
###### Benchmark 3: expensive initialization ######
###################################################


def save_model():
    mnist = tf.keras.datasets.mnist.load_data()
    x_train, y_train = mnist[0]
    x_train = x_train / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    # Train the model.
    model.fit(x_train, y_train, epochs=1)
    # Save the model to disk.
    filename = '/tmp/model'
    model.save(filename)


# Train and save the model. This has to be done in a separate process because
# otherwise Python multiprocessing will hang when you try do run the code
# below.
p = Process(target=save_model)
p.start()
p.join()

filename = '/tmp/model'


def evaluate_next_batch(i):
    # Pin the process to a specific core if we are on Linux to prevent
    # contention between the different processes since TensorFlow uses
    # multiple threads.
    if sys.platform == 'linux':
        psutil.Process().cpu_affinity([i])
    model = tf.keras.models.load_model(filename)
    mnist = tf.keras.datasets.mnist.load_data()
    x_test = mnist[1][0] / 255.0
    return model.predict(x_test)


pool = Pool(num_cpus)

durations3 = []
for _ in range(num_trials):
    start_time = time.time()

    for _ in range(10):
        pool.map(evaluate_next_batch, range(num_cpus))

    duration3 = time.time() - start_time
    durations3.append(duration3)
    print('Expensive initialization workload took {} seconds.'.format(duration3))

print('Used {} cores.'.format(num_cpus))

print("""
Results:
- Numerical computation: {} +/- {}
- Stateful computation: {} +/- {}
- Expensive initialization: {} +/- {}
""".format(np.mean(durations1), np.std(durations1),
           np.mean(durations2), np.std(durations2),
           np.mean(durations3), np.std(durations3)))
