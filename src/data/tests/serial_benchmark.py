import time
from collections import defaultdict

import numpy as np
import psutil
import scipy.signal
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

num_trials = 5

# Count the number of physical CPUs.
num_cpus = psutil.cpu_count(logical=False)
print('Using {} cores.'.format(num_cpus))


################################################
###### Benchmark 1: numerical computation ######
################################################


def f(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]


filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]


def run_benchmark():
    image = np.zeros((3000, 3000))
    [f(image, filters[i]) for i in range(num_cpus)]


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


class StreamingPrefixCount(object):
    def __init__(self):
        self.prefix_count = defaultdict(int)
        self.popular_prefixes = set()

    def add_document(self, document):
        for word in document:
            for i in range(1, len(word)):
                prefix = word[:i]
                self.prefix_count[prefix] += 1
                if self.prefix_count[prefix] > 3:
                    self.popular_prefixes.add(prefix)

    def get_popular(self):
        return self.popular_prefixes


durations2 = []
for _ in range(num_trials):
    streaming_actors = [StreamingPrefixCount() for _ in range(num_cpus)]

    start_time = time.time()

    for i in range(num_cpus * 10):
        document = [np.random.bytes(20) for _ in range(10000)]
        streaming_actors[i % num_cpus].add_document(document)

    # Aggregate all of the results.
    results = [actor.get_popular() for actor in streaming_actors]
    popular_prefixes = set()
    for prefixes in results:
        popular_prefixes |= prefixes

    duration2 = time.time() - start_time
    durations2.append(duration2)
    print('Stateful computation workload took {} seconds.'.format(duration2))

###################################################
###### Benchmark 3: expensive initialization ######
###################################################

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


class Model(object):
    def __init__(self):
        # Load the model and some data.
        self.model = tf.keras.models.load_model(filename)
        mnist = tf.keras.datasets.mnist.load_data()
        self.x_test = mnist[1][0] / 255.0

    def evaluate_next_batch(self):
        # Note that we reuse the same data over and over, but in a
        # real application, the data would be different each time.
        return self.model.predict(self.x_test)


actor = Model()

durations3 = []
for _ in range(num_trials):
    start_time = time.time()

    for j in range(10):
        results = [actor.evaluate_next_batch() for _ in range(num_cpus)]

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
