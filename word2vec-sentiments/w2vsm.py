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
    #prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    #train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    #train_labels[12500 + i] = 0
    	
for i in xrange(2331,2331+321):
    #prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    #train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[i] = model.docvecs[prefix_train_neg]
    #train_labels[i] = 1
    train_labels[i] = 0
import ipdb; ipdb.set_trace()  # <--- *BAMF!*

print train_labels

test_arrays = numpy.zeros((2331+321, 100))
test_labels = numpy.zeros(2331+321)

for i in xrange(2331):
    prefix_test_pos = 'TEST_POS_' + str(i)
    #prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    #test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    #test_labels[12500 + i] = 0


for i in xrange(2331,2331+321):
    #prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    #test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[i] = model.docvecs[prefix_test_neg]
    #test_labels[i] = 1
    test_labels[i] = 0
import ipdb; ipdb.set_trace()
log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(test_arrays, test_labels)
