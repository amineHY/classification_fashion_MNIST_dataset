# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from subprocess import check_output
from tensorflow.examples.tutorials.mnist import input_data

# get_ipython().run_cell_magic('bash', '', 'rm -r /tmp/tfmodels/fashion_mnist')
base_model_dir = "/tmp/tfmodels/fashion_mnist/"

# %% print the version of tensorflow and python
tf.logging.set_verbosity(tf.logging.INFO)
# !python - V
print(tf.__version__)

# DATA PREPARATION
print(check_output(["ls", "./data/fashion"]).decode("utf8"))
# DATA_SETS = input_data.read_data_sets("./data/")
DATA_SETS = input_data.read_data_sets(
    'data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
features_name = "pixels"
feature_columns = [tf.feature_column.numeric_column(features_name, shape=784)]


# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ## LINEAR CLASSIFIER
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %% Classification model
classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=10,
    model_dir=base_model_dir + "linear"
)
# %% Input function


def make_input_fn(data, batch_size, num_epochs, shuffle):
    return tf.estimator.inputs.numpy_input_fn(
        x={'pixels': data.images},
        y=data.labels.astype(np.int64),
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle)


# ## Training
classifier.train(input_fn=make_input_fn(DATA_SETS.train,
                                        batch_size=100,
                                        num_epochs=2,
                                        shuffle=True))

# ## Evaluation / Validation
accuracy_score = classifier.evaluate(
    input_fn=make_input_fn(
        DATA_SETS.test,
        batch_size=100,
        num_epochs=1,
        shuffle=False))['accuracy']

print('\nThe classification accuracy is : {0:2.2f} %\n'.format(
    accuracy_score*100))


# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ## DEEP LEARNING CLASSIFIER
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %% Classification model
deep_classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=10,
    hidden_units=[100, 75, 50],
    model_dir=base_model_dir + "deep"
)


# %% Training
deep_classifier.train(input_fn=make_input_fn(DATA_SETS.train,
                                             batch_size=100,
                                             num_epochs=2,
                                             shuffle=True))

# %% Evaluation / Validation
accuracy_score = deep_classifier.evaluate(
    input_fn=make_input_fn(
        DATA_SETS.test,
        batch_size=100,
        num_epochs=1,
        shuffle=False))['accuracy']

print('\nThe classification accuracy is : {0:2.2f}%\n'.format(
    accuracy_score*100))

# %% Prediction
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'pixels': DATA_SETS.test.images[5000:5005]},
    batch_size=1,
    num_epochs=1,
    shuffle=False)
predictions = deep_classifier.predict(input_fn=predict_input_fn)

for prediction in predictions:
    print("Predictions:    {} with probabilities {}\n".format(
        prediction["classes"], prediction["probabilities"]))
print('Expected answers values: {}'.format(
    DATA_SETS.test.labels[5000:5005]))


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
class_table = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

for i in range(5000, 5005):
    sample = np.reshape(DATA_SETS.test.images[i], (28, 28))
    plt.figure()
    plt.title("Correct label: {}".format(
        class_table[DATA_SETS.test.labels[i]]))
    plt.imshow(sample, 'gray')
    plt.show(True)


# ## Prepare the production of the algorithm
# This operation is done after the training of the  model
# it takes a snapshot of it and send it to production
feature_spec = {features_name: tf.FixedLenFeature(
    shape=[784], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    feature_spec)

classifier.export_savedmodel(export_dir_base=base_model_dir + '/export',
                             serving_input_receiver_fn=serving_fn)
