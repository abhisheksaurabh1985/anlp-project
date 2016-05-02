import sys
import nltk
import itertools
from function_definitions import *
from nltk.tokenize import word_tokenize

#csvFileName= './data/Annotations-1-120.csv'
#dictAnnotatedData1= getCSVInDictionary(csvFileName) # Read the annotated data in an Ordered Dictionary


txtFileName= './data/Annotations-1-120.txt'
dictAnnotatedData= getTxtInDictionary(txtFileName) # Read the annotated data in an Ordered Dictionary


tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:]) # Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences. 
print len(tokenizedSentences)

tokenizedConcepts = getTokens(dictAnnotatedData['Concept'][0:]) # Tokenize 'concepts'
print len(tokenizedConcepts)

# Define B-I-O tags as per IOB2 convention. Three types of tags have been used viz. O (Others), B-X (Beginning of X)
# and I-X (Inside X) where X is 'CONCEPT'.
bioTags= ['O', 'B', 'I']

# Training data for CRF
[posTaggedTokens, indexConceptsInSentences, listBioTags] = getTrainingDataForCRF(tokenizedSentences,
                                                                                 tokenizedConcepts,
                                                                                 bioTags)


        





