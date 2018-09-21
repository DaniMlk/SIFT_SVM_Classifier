DATASET_PATH = './dataset'
import os
import cv2
from pylab import *
import scipy.stats as st
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
import itertools
import collections
import matplotlib as mpl
from utils.tqdm import tqdm
from mnist import MNIST
import inspect
import mnist
from sklearn.externals import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pdb
def make_log(sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    return ((x**2 + y**2 - (2.0 * sigma**2)) / sigma**4) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / normal

mnist = MNIST(DATASET_PATH, return_type='numpy')
train_img, train_lbl = mnist.load_training()

sift = cv2.xfeatures2d.SIFT_create()
descriptors = []
keypoints = []
image_ids = []

for image_id, pic in enumerate(train_img):
    kp, des = sift.detectAndCompute(pic.reshape((28, 28)).astype(np.uint8), None)
    if des is not None:
        keypoints.extend(list(kp))
        descriptors.extend(list(des))
        image_ids.extend([image_id] * len(kp))
        # pdb.set_trace()

from sklearn.cluster import KMeans, MiniBatchKMeans
descriptors = np.array(descriptors)

kmeans = MiniBatchKMeans(n_clusters=200, random_state=0, max_iter=1000, batch_size=1000).fit(descriptors)
joblib.dump(kmeans, '../kmeans5.pkl')
clusters = kmeans.labels_
features = np.zeros((len(train_img), 200))
for image_id, cluster in zip(image_ids, clusters):
    features[image_id, cluster] += 1
#     pdb.set_trace()

n_estimators = 20
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear'), max_samples=1.0 / n_estimators, n_estimators=n_estimators), n_jobs=-1)
clf.fit(features, train_lbl)
joblib.dump(clf, '../clf5.pkl')

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading

def tpmap(fn, *iterables, progress=True):
    with ThreadPoolExecutor() as executor:
        if progress:
            try:
                total = len(iterables[0])
            except TypeError:
                total = None
        ret = executor.map(fn, *iterables)
        if progress:
            ret = tqdm(ret, total=total)
        yield from ret

from threading import current_thread
thread_local = threading.local()

feature_chunks = np.split(features, 1000)

def run(chunk):
    try:
        clf = thread_local.clf
    except AttributeError:
        print('initializing')
        clf = joblib.load('../clf5.pkl')
        thread_local.clf = clf
    return clf.predict(chunk)

predicts = list(tpmap(run, feature_chunks))
predicts = np.concatenate(predicts)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import confusion_matrix
train_confusion = confusion_matrix(train_lbl, predicts)
plot_confusion_matrix(train_confusion, range(10), normalize=True)
plt.savefig('out5_train.png')
train_accuracy = np.sum(train_lbl == predicts) / len(predicts)
print(train_accuracy)

test_img, test_lbl = mnist.load_testing()
descriptors = []
keypoints = []
image_ids = []

for image_id, pic in tqdm(enumerate(test_img), total=len(test_img)):
    kp, des = sift.detectAndCompute(pic.reshape((28, 28)).astype(np.uint8), None)
    if des is not None:
        keypoints.extend(list(kp))
        descriptors.extend(list(des))
        image_ids.extend([image_id] * len(kp))

descriptors = np.array(descriptors)
clusters = kmeans.predict(descriptors)

test_features = np.zeros((len(test_img), 200))
for image_id, cluster in zip(image_ids, clusters):
    test_features[image_id, cluster] += 1


test_feature_chunks = np.split(test_features, 1000)

test_predicts = list(tpmap(run, test_feature_chunks))
test_predicts = np.concatenate(test_predicts)

test_confusion = confusion_matrix(test_lbl, test_predicts)
plot_confusion_matrix(test_confusion, range(10), normalize=True)
plt.savefig('out5_test.png')
test_accuracy = np.sum(test_lbl == test_predicts) / len(test_predicts)
print(test_accuracy)
