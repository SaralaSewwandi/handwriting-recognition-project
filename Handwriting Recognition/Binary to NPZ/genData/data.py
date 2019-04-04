import numpy as np

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

