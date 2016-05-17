'''
This script used a pre-trained word2vec model to create training data for
    sentiment analysis.
    Note: requires a pre-trained model from:
    https://github.com/3Top/word2vec-api
'''
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix

# load the data and labels to files
sentenceData = np.load('bioscope/word2vec_sentence_data.npy')
sentenceLabels = np.load('bioscope/word2vec_sentence_labels.npy')

# Create an SVM model with the default parameters
#TODO: tweak the gamma and C values
clf = svm.SVC()

# Fit the entire dataset to the data
#TODO: this needs to be split into train and validation data
clf.fit(np.transpose(sentenceData), np.transpose(sentenceLabels))

# evaluate the y-predicted values on the validation data
yp = clf.predict(np.transpose(sentenceData))

# Output the accuracy of predicion
diff = yp - sentenceLabels
diff = yp - sentenceLabels
diff = np.absolute(diff)
print("Prediction accuracy: %f" % (1-np.sum(diff)/len(sentenceLabels)))
print confusion_matrix(sentenceLabels, yp)
