# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics import confusion_matrix
# random shuffle
from random import shuffle
from sklearn.metrics import accuracy_score
# numpy

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

test_arrays = np.zeros((813+813, 100))
test_labels = np.zeros(813+813)

for i in xrange(813):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_train_pos]
    test_labels[i] = 1
    	
for i in xrange(0,813):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    test_arrays[813+i] = model.docvecs[prefix_train_neg]
    test_labels[813+i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

import cPickle
# save the classifier
with open('ReadyToGoClassifier.pkl', 'wb') as fid:
    cPickle.dump(classifier, fid)    

# load it again
#with open('ReadyToGoClassifier.pkl', 'rb') as fid:
#    classifier = cPickle.load(fid)

print classifier.score(test_arrays, test_labels)
pred = classifier.predict(test_arrays)
print pred
print accuracy_score(test_labels, pred)
print confusion_matrix(test_labels, pred)
