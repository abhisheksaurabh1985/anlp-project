'''
SVM model using pre-trained word2vec data to detect sentiment analysis
    Note: requires a pre-trained model from:
    https://github.com/3Top/word2vec-api
'''

from sklearn import svm
from gensim.models.word2vec import Word2Vec as w

# load model from file
model = w.load_word2vec_format('/home/andrei/dev/word2vec-api/model.bin.gz', binary=True)

# read the pre-processed bioscope input file and get the polarity of each line
lines = [line.rstrip('\n') for line in open('bioscope/full_papers.txt')]
for line in lines:
    polarity, sentence = line.split(' ', 1)
    polarity = int(polarity)

    # get the numpy value of the current word in the sentence
    print('polarity: %d, sentence: %s' % (polarity, sentence))
    words = sentence.split(' ')
    for word in words:
        # strip non-alpha chars from the sentence
        w = filter(str.isalnum, word)
        try:
            w_val = model[w]
        except:
            print ":("
        print word
        print w_val
