import sys
import nltk
import itertools
from function_definitions import *
from nltk.tokenize import word_tokenize

# Read the annotated data in an Ordered Dictionary
txtFileName= './data/Annotations-1-120.txt'
triggersFileName = './data/negex_triggers.txt'

dictAnnotatedData= getTxtInDictionary(txtFileName)

# Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences.
tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:])
print len(tokenizedSentences)

triggers= getTriggers(triggersFileName)
cueTypeTags = extractCueTypeTags(tokenizedSentences, triggers)

# Tokenize 'concepts'
tokenizedConcepts = getTokens(dictAnnotatedData['Concept'][0:])
print len(tokenizedConcepts)

negations = dictAnnotatedData['Negation']

# Define B-I-O tags as per IOB2 convention. Three types of tags have been used viz. O (Others), B-X (Beginning of X)
# and I-X (Inside X) where X is 'CONCEPT'.
bioTags= ['O', 'B', 'I']
priorities= {'O':0, 'B':2, 'I':1}

# Training data for CRF with negated concepts
trainDataCRF = getTrainingDataForCRF(tokenizedSentences,
                                     tokenizedConcepts, negations,
                                     bioTags, priorities, ["Negated"], cueTypeTags)

prefix = "./output/"
# write the data to file
filename = prefix + 'trainNegatedCRF.csv'
writeCsvToFile(trainDataCRF, filename)
print "%s created" % filename

# Split data for training and testing
percentTestData= 25 # Only integer

fileNameCRFData= prefix + 'trainNegatedCRF.csv'
[dataCRF, trainingDataCRF, testDataCRF] = splitDataForValidation(fileNameCRFData, percentTestData)
# write data to file
trainingFileName = prefix + 'trainingNegatedDataCRF.txt'
writeLinesToFile(trainingDataCRF, trainingFileName)
print ("training file created: %s" % trainingFileName)

testFileName = prefix + 'testNegatedDataCRF.txt'
writeLinesToFile(testDataCRF, testFileName)
print ("test file created %s" % testFileName)
