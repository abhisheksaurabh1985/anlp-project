# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

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
train_arrays = numpy.zeros((2331+321, 100))
train_labels = numpy.zeros(2331+321)

for i in xrange(2331):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1
    	
for i in xrange(0,321):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[2331+i] = model.docvecs[prefix_train_neg]
    train_labels[2331+i] = 0

print train_labels

test_arrays = numpy.zeros((2331+321, 100))
test_labels = numpy.zeros(2331+321)

for i in xrange(2331):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_train_pos]
    test_labels[i] = 1
    	
for i in xrange(0,321):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    test_arrays[2331+i] = model.docvecs[prefix_train_neg]
    test_labels[2331+i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(test_arrays, test_labels)
