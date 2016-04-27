from function_definitions import *
from nltk.tokenize import word_tokenize

# Read the annotated data in an Ordered Dictionary
csvFileName= './negex.python/Annotations-1-120.csv'
dictAnnotatedData= getCSVInDictionary(csvFileName)

# Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences. 
tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:])
print len(tokenizedSentences)
