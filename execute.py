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
bioTags= ['O', 'B', 'I']
priorities= {'O':0, 'B':2, 'I':1}

# Training data for CRF
[posTaggedTokens, indexConceptsInSentences, listBioTags, trainDataCRF] = getTrainingDataForCRF(tokenizedSentences,
                                                                                 tokenizedConcepts, negations,
                                                                                 bioTags, priorities, "Affirmed")
# write the data to file
filename = './output/trainAffirmedCRF.csv'
writeCsvToFile(trainDataCRF, filename)
print "%s created" % filename

# Training data for CRF
[posTaggedTokens, indexConceptsInSentences, listBioTags, trainDataCRF] = getTrainingDataForCRF(tokenizedSentences,
                                                                                 tokenizedConcepts, negations,
                                                                                 bioTags, priorities, "Negated")
# write the data to file
filename = './output/trainNegatedCRF.csv'
writeCsvToFile(trainDataCRF, filename)
print "%s created" % filename


# Split data for training and testing
percentTestData= 25 # Only integer


fileNameCRFData= './output/trainNegatedCRF.csv'
[dataCRF, trainingDataCRF, testDataCRF] = splitDataForValidation(fileNameCRFData, percentTestData)
# write data to file
writeLinesToFile(trainingDataCRF, './crf-test/trainingNegatedDataCRF.txt')
writeLinesToFile(testDataCRF, './crf-test/testNegatedDataCRF.txt')

fileNameCRFData= './output/trainAffirmedCRF.csv'
[dataCRF, trainingDataCRF, testDataCRF] = splitDataForValidation(fileNameCRFData, percentTestData)
# write data to file
writeLinesToFile(trainingDataCRF, './crf-test/trainingAffirmedDataCRF.txt')
writeLinesToFile(testDataCRF, './crf-test/testAffirmedDataCRF.txt')