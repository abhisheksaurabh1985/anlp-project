# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics import confusion_matrix
# random shuffle
from random import shuffle
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

import numpy as np
from matplotlib import pyplot as plt

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

model = Doc2Vec.load('./medical.d2v')

log.info('Sentiment')
train_arrays = np.zeros((813+813, 100))
train_labels = np.zeros(813+813)

for i in xrange(813):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in xrange(0,813):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[813+i] = model.docvecs[prefix_train_neg]
    train_labels[813+i] = 0

print train_labels

import ipdb; ipdb.set_trace()

log.info('Fitting')

# split data into N kfolds
kf = KFold(train_labels.shape[0], n_folds=4)
fold_num = 1
for train_index, test_index in kf:
    x_train, x_val = train_arrays[train_index,:], train_arrays[test_index,:]
    y_train, y_val = train_labels[train_index], train_labels[test_index]
    
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

#import cPickle
# save the classifier
#with open('ReadyToGoClassifier.pkl', 'wb') as fid:
#    cPickle.dump(classifier, fid)
#
# load it again
#with open('ReadyToGoClassifier.pkl', 'rb') as fid:
#    classifier = cPickle.load(fid)
    print("K-fold #%d" % (fold_num))
    print classifier.score(x_val, y_val)
    pred = classifier.predict(x_val)
    print pred
    print accuracy_score(y_val, pred)
    print confusion_matrix(y_val, pred)
    fold_num = fold_num + 1

