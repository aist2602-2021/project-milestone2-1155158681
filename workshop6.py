import tensorflow_datasets as tfds;

ds = tfds.load("fashion_mnist", split = "train",
               shuffle_files = True, batch_size = -1);

import tensorflow as tf;
from keras.models import Sequential;
from keras.layers import Dense;

model = Sequential();
model.add(tf.keras.layers.Flatten());
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255));

model.add(Dense(units = 10, activation = "softmax", input_dim = 64));
model.add(Dense(units = 64, activation = "relu", input_dim = 784));
model.add(Dense(units = 64, activation = "relu", input_dim = 784));
model.add(Dense(units = 64, activation = "relu", input_dim = 784));
model.add(Dense(units = 64, activation = "relu", input_dim = 784));
model.add(Dense(units = 64, activation = "relu", input_dim = 784));

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              optimizer = "sgd", metrics = "accuracy");

new_ds = tfds.as_numpy(ds);
x_train, y_train = new_ds["image"], new_ds["label"];
model.fit(x_train, y_train, epochs = 5, batch_size = 32);

test_ds, meta_info = tfds.load("fashion_mnist", split = "test", shuffle_files = False,
                               batch_size = -1, with_info = True);

test_ds = tfds.as_numpy(test_ds);
x_test, y_test = test_ds["image"], test_ds["label"]

import matplotlib.pyplot as plt;
import numpy as np;

img_idx = 0;
img = x_test[img_idx: img_idx + 1];
pred = model.predict(img, batch_size = 1)[0];
label = np.argmax(pred);

plt.imshow(np.squeeze(img), cmap = plt.get_cmap("gray"));
plt.show();

print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(meta_info.features["label"].names[label], pred[label]));

# Submission from 1155158681 (ChunHo Yip)
