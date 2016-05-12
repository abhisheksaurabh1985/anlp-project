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

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences
        

log.info('source load')
sources = {'../bioscope/full_papers_neg_parsed.txt':'TEST_NEG', '../bioscope/full_papers_pos_parsed.txt':'TEST_POS', '../bioscope/full_papers_neg_parsed.txt':'TRAIN_NEG', '../bioscope/full_papers_pos_parsed.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in xrange(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

log.info('Model Save')
model.save('./medical.d2v')
model = Doc2Vec.load('./medical.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((813+813, 100))
train_labels = numpy.zeros(813+813)

for i in xrange(813):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1
    	
for i in xrange(0,813):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[813+i] = model.docvecs[prefix_train_neg]
    train_labels[813+i] = 0

print train_labels

test_arrays = numpy.zeros((813+813, 100))
test_labels = numpy.zeros(813+813)

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

print classifier.score(test_arrays, test_labels)
