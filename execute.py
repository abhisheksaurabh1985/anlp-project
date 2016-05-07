import sys
import nltk
import itertools
from function_definitions import *
from nltk.tokenize import word_tokenize

# Read the annotated data in an Ordered Dictionary
txtFileName= './data/Annotations-1-120.txt'
dictAnnotatedData= getTxtInDictionary(txtFileName)

# Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences.
tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:])
print len(tokenizedSentences)

# Tokenize 'concepts'
tokenizedConcepts = getTokens(dictAnnotatedData['Concept'][0:])
print len(tokenizedConcepts)

negations = dictAnnotatedData['Negation']

# Define B-I-O tags as per IOB2 convention. Three types of tags have been used viz. O (Others), B-X (Beginning of X)
# and I-X (Inside X) where X is 'CONCEPT'.
bioTags= ['O', 'BN', 'IN', 'BA', 'IA']
priorities= {'O':0, 'BN':2, 'IN':1, 'BA': 2, 'IA':1}

# Training data for CRF
[posTaggedTokens, indexConceptsInSentences, listBioTags, trainDataCRF] = getTrainingDataForCRF(tokenizedSentences,
                                                                                 tokenizedConcepts, negations,
                                                                                 bioTags, priorities)
# write the data to file
filename = './output/trainCRF.csv'
writeCsvToFile(trainDataCRF, filename)
print "%s created" % filename

# Split data for training and testing
fileNameCRFData= './output/trainCRF.csv'
percentTestData= 25 # Only integer
[dataCRF, trainingDataCRF, testDataCRF] = splitDataForValidation(fileNameCRFData, percentTestData)
# write data to file
writeLinesToFile(trainingDataCRF, './crf-test/trainingDataCRF.txt')
writeLinesToFile(testDataCRF, './crf-test/testDataCRF.txt')
