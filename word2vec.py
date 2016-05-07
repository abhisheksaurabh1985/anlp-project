'''
This script used a pre-trained word2vec model to create training data for
    sentiment analysis.
    Note: requires a pre-trained model from:
    https://github.com/3Top/word2vec-api
'''

# General Python
import re

# Machine learning
from gensim.models.word2vec import Word2Vec as w
import numpy as np

def normalized(a, axis=-1, order=2):
    """ Normalize a numpy vector
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

# load model from file
MODEL_DIMENSIONALITY = 300
model = w.load_word2vec_format('/home/andrei/dev/word2vec-api/model.bin.gz', binary=True)

# read the pre-processed bioscope input file and get the polarity of each line
lines = [line.rstrip('\n') for line in open('bioscope/full_papers.txt')]
numLines = len(lines)

# create the sentence data matrix to be filled in later
sentenceData = np.zeros((MODEL_DIMENSIONALITY, numLines))
sentenceLabels = np.zeros(numLines)

for i in xrange(0, numLines):
    line = lines[i]
    polarity, sentence = line.split(' ', 1)
    polarity = int(polarity)

    # remove non alpha chars
    sentence = re.sub("[^a-zA-Z\s-]", "", sentence)
    # replace all '-' chars with ' '
    sentence = re.sub('-', ' ', sentence)

    # get the numpy value of the current word in the sentence
    print('polarity: %d, sentence: %s' % (polarity, sentence))
    words = sentence.split(' ')

    # get the total vector sum of the entire sentence
    s_val = np.zeros(MODEL_DIMENSIONALITY)
    for word in words:
        try:
            w_val = model[word]
        except:
            w_val = np.zeros(MODEL_DIMENSIONALITY)
        s_val = s_val + w_val
    s_val = normalized(s_val)

    sentenceData[:,i] = s_val
    sentenceLabels[i] = polarity
    print('step %d/%d' % (i, numLines))

# save the data and labels to files
np.save('bioscope/word2vec_sentence_data.npy', sentenceData)
np.save('bioscope/word2vec_sentence_labels.npy', sentenceLabels)
