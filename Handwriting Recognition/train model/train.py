

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
# Define if you want to download data from the original database or use the dataset one already provided and preprocessed
# Use:
# 'load': If you want to load the datase from the directory
# 'download': To download data from the database and process the images

dataset_load_method = 'load'

# Define if you want to save the dataset to a file
save_dataset = False

# Define if you want to load the trained classifiers from the directory
#load_classifiers = False
load_classifiers = True

# Define if you want to save the trained classifiers to a file
save_classifiers = True

# Define if you want to save classification test output to a file
save_results = True
if (save_results):
    result_output_file = open('result_output.txt','w') 

# Define if you want to print errors and warnings
enable_error_output = False

import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.tools as tls
import scipy
import sklearn
import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize

from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import data, color, exposure, measure
from skimage.transform import resize
from skimage.feature import hog
from sklearn.manifold import TSNE
from sklearn.externals import joblib

import urllib

from io import StringIO
import cv2
from scipy import ndimage
from difflib import SequenceMatcher
from sys import stdout
from IPython.display import clear_output

#function for displaying purposes
def print_percentage(prct, msg=None):
    if (prct > 100 or prct < 0):
        return
    clear_output(wait=True)
    if (msg == None):
        stdout.write("Progress: [")
    else:
        stdout.write(msg+" [")
    end = int(int(prct)/10)
    for i in range(0, end):
        stdout.write("=")
    for i in range(end, 10):
        stdout.write(" ")
    stdout.write("] "+str(prct)+"%")
    stdout.flush()


#read from the csv
df=pd.read_csv('first_and_last_names_fix.csv', sep=',',header=None)


def delborders(crop):
    cropf = ndimage.gaussian_filter(crop, 0.5)
    cropbin = (cropf<0.8)
    labeled, nr_objects = ndimage.label(cropbin)
    labels_to_delete = []
    for i in range(0, labeled.shape[1]):
        if (labeled[labeled.shape[0]-1][i] > 0):
            labels_to_delete.append(labeled[labeled.shape[0]-1][i])
    
    label_in_delete = False
    for x in range(0, labeled.shape[1]):
        for y in range(0, labeled.shape[0]):
            label_in_delete = False
            for l in range(0, len(labels_to_delete)):
                if (labeled[y][x] == labels_to_delete[l]):
                    label_in_delete = True
            
            if(label_in_delete):
                crop[y][x] = 1.0
    
    return crop



def getcrop(n):
    try: 
        urllib.request.urlretrieve(df[1][n], "temp.jpg")
    except urllib.error.URLError as e:
        return None, False
    img = cv2.imread('temp.jpg')
    imgh, imgw = img.shape[:-1]
    img_rgb = img.copy()
    template = cv2.imread('template.png')
    h, w = template.shape[:-1]

    template_match_success = False
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = .7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch collumns and rows
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        croph1 = pt[1]
        croph2 = pt[1]+h
        cropw = pt[0] + w
        template_match_success = True

    if (not template_match_success):
        #Template matching has failed so return...
        return img, False

    if (df[3][n] == 'first' or df[3][n] == 'last'):
        crop = img.copy()[max(croph1-6, 0):min(croph2+6, imgh), cropw:imgw]
    else:
        crop = img.copy()[max(min(croph2+4, imgh-1), 0):imgh, :]
        
    crop = color.rgb2gray(crop)
    if (df[3][n] == 'first_b' or df[3][n] == 'last_b'):
        crop = delborders(crop)
    return crop, True


def gen_dataset(n=df.shape[0]):
    data = []
    labels = []
    for i in range(1, n):
        crop, success = getcrop(i)
        if (success):
            data.append(crop)
            labels.append(df[2][i])
        else:
            if (enable_error_output):
                print("[WARNING] Template matching has failed for image: "+str(i))
        print_percentage((i*100/(n-1)), "Fetched "+str(i)+" images:")
    
    print_percentage(100, "Fetched "+str(n-1)+" images:")
    print("")
    print("Finished!")
    return data, labels


dataset=[]
labels=[]

# Load dataset from files
if (dataset_load_method == 'load'):
    dataset = np.load("./HandwrittenNames-dataset/HandwrittenNames_data.npz")['data']
    labels = np.load("./HandwrittenNames-dataset/HandwrittenNames_labels.npz")['data']


#extra functions
def get_labels(crop):
    img = crop.copy() # gray-scale image

    # You could smooth the image (to remove small objects) but we saw better results without using it...
    # blur_radius = 0.5
    # imgf = ndimage.gaussian_filter(img, blur_radius)

    threshold = 0.8

    # Find connected components
    labeled, nr_objects = ndimage.label(img<threshold) 
    #print("Number of objects is " +str(nr_objects))

    return labeled, nr_objects



def get_bboxes(labeled, nr_objects):
    bboxes = np.zeros((nr_objects, 2, 2), dtype='int')

    x1, y1, x2, y2 = 0, labeled.shape[0], 0, 0
    coord = 0
    cont = 0
    ytop, ybot = 0, 0
    nzero, firstb = False, False

    for x in range(0, labeled.shape[1]):
        nzero, firstb = False, False
        ytop, ybot = 0, 0
        for y in range(0, labeled.shape[0]):
            if (labeled[y][x] > 0):
                nzero = True
                if (not firstb):
                    ytop = y
                    firstb = True
                ybot = y

        if (nzero):
            if (ytop < y1):
                y1 = ytop
            if (ybot > y2):
                y2 = ybot
            if (coord == 0):
                x1 = x
                coord = 1
            elif (coord == 1):
                x2 = x
        elif ((not nzero) and (coord == 1)):
            bboxes[cont][0] = [x1, y1]
            bboxes[cont][1] = [x2, y2]
            cont += 1
            coord = 0
            x1, y1, x2, y2 = 0, labeled.shape[0], 0, 0

    bboxes = bboxes[0:cont]
    return bboxes, cont


def crop_characters(img, bboxes, n):
    characters = []
    for i in range(0, n):
        c = img.copy()[bboxes[i][0][1]:bboxes[i][1][1], bboxes[i][0][0]:bboxes[i][1][0]]
        if (c.shape[0] != 0 and c.shape[1] != 0):
            c = resize(c, (28, 28), mode='constant', cval=1.0, clip=True)
            characters.append((c<0.80).reshape(784))
    return characters, len(characters)


def labelsep(label):
    if (type(label) is str or type(label) is np.str_):
        decomposed_label = list(label)
        labels = []
        for i in range(0, len(decomposed_label)):
            if (decomposed_label[i] != ' '):
                labels.append(decomposed_label[i])
        return labels
    else:
        return []


def get_characters(image, label):
    labeled, nr_objects = get_labels(image)
    bboxes, n = get_bboxes(labeled, nr_objects)
    characters, n_chars = crop_characters(image, bboxes, n)
    labels = labelsep(label)
    return characters, labels[0:n_chars]

def get_characters_img_only(image):
    labeled, nr_objects = get_labels(image)
    bboxes, n = get_bboxes(labeled, nr_objects)
    characters, n_chars = crop_characters(image, bboxes, n)
    return characters


# Define if we want to print status of the training process
verbose_classifiers = True

# RBM Definition
rbm = BernoulliRBM(n_components=300, learning_rate=0.01, n_iter=45, random_state=0, verbose=verbose_classifiers)

# MLP Classifier Definition 
mlp = MLPClassifier(hidden_layer_sizes=(300,400,150), activation='relu', max_iter=5000, tol=0.0001, random_state=1, verbose=verbose_classifiers)
# We define the pipeline afterwards: MLP Classifier with RBM features
rbm_mlp_classifier = Pipeline(steps=[('rbm', rbm), ('mlp', mlp)])

X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

X_train_chars = []
Y_train_chars = []
Train_with_inconsistencies = []
z = 0
for i in range(0, len(X_train)):
    print_percentage(i*100/len(X_train), "Processing train image "+ str(i)+" :")
    characters, charlabels = get_characters(X_train[i], Y_train[i])
    if (len(characters) != len(charlabels) or len(characters) == 0 or len(charlabels) == 0):
        if (enable_error_output):
            print("[Warning] Input number "+str(i)+" inconsistent! Skipping this one...")
        Train_with_inconsistencies.append(i)
        z += 1
    else:
        X_train_chars.extend(characters)
        Y_train_chars.extend(charlabels)

print_percentage(100, "Processing train image "+ str(len(X_train))+" :")
print("")
print(str(100-(z*100/len(X_train)))+"% of the data in train batch correctly extracted.")



X_test_chars = []
Y_test_chars = []
Test_without_inconsistencies = []
z = 0
for i in range(0, len(X_test)):
    print_percentage(i*100/len(X_test), "Processing test image "+ str(i)+" :")
    characters, charlabels = get_characters(X_test[i], Y_test[i])
    if (len(characters) != len(charlabels) or len(characters) == 0 or len(charlabels) == 0):
        if (enable_error_output):
            print("[Warning] Input number "+str(i)+" inconsistent! Skipping this one...")
        z += 1
    else:
        X_test_chars.extend(characters)
        Y_test_chars.extend(charlabels)
        Test_without_inconsistencies.append(i)

print_percentage(100, "Processing train image "+ str(len(X_test))+" :")
print("")
print(str(100-(z*100/len(X_test)))+"% of the data in test batch correctly extracted.")


rbm_mlp_classifier.fit(X_train_chars[:], Y_train_chars)


# Save the classifiers if enabled:
if (save_classifiers):
	joblib.dump(rbm_mlp_classifier, 'RBM_MLP_classifier.pkl')


print("finished")
