import numpy as np

import tensorflow as tf

with open("train-images-idx3-ubyte", "rb") as binary_file:
    # Read the whole file at once
    dataset = binary_file.read()
    #print(data)
    np.savez("train_images.npz", data=dataset)


train_img = np.load("train_images.npz")['data']
#print(train_img)


with open("train-labels-idx1-ubyte", "rb") as binary_file:
    # Read the whole file at once
    dataset = binary_file.read()
    #print(data)
    np.savez("train_labels.npz", data=dataset)


train_lbl = np.load("train_labels.npz")['data']
#print(train_img)


with open("t10k-images-idx3-ubyte", "rb") as binary_file:
    # Read the whole file at once
    dataset = binary_file.read()
    #print(data)
    np.savez("test_images.npz", data=dataset)


test_img = np.load("test_images.npz")['data']
#print(train_img)


with open("t10k-labels-idx1-ubyte", "rb") as binary_file:
    # Read the whole file at once
    dataset = binary_file.read()
    #print(data)
    np.savez("test_labels.npz", data=dataset)


test_lbl = np.load("test_labels.npz")['data']
#print(train_img)

#X_train = np.load("train_images.npz")['data']
#print(X_train)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
