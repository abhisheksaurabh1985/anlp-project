import sys
import nltk
import itertools
from function_definitions import *
from nltk.tokenize import word_tokenize

# Read the annotated data in an Ordered Dictionary
csvFileName= './data/Annotations-1-120_small.csv'
dictAnnotatedData= getCSVInDictionary(csvFileName)

# Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences. 
tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:])
print len(tokenizedSentences)

'''
Data preparation for linear chain CRF
'''
# Get POS tags for each of the sentences
posTaggedTokens = []
for eachTokenizedSentence in tokenizedSentences:
    posTaggedTokens.append(nltk.pos_tag(eachTokenizedSentence))

# Tokenize 'concepts'
tokenizedConcepts = getTokens(dictAnnotatedData['Concept'][0:])
print len(tokenizedConcepts)

indexConceptsInSentences= []
for i in range(len(tokenizedConcepts)):
    temp = []
    temp.append(tokenizedSentences[i].index(tokenizedConcepts[i][0]))
    temp.append(tokenizedSentences[i].index(tokenizedConcepts[i][-1]))
    indexConceptsInSentences.append(temp)
    
# Define B-I-O tags as per IOB2 convention. Three types of tags have been used viz. O (Others), B-X (Beginning of X)
# and I-X (Inside X) where X is 'CONCEPT'.
bioTags= ['O', 'B-Concept', 'I-Concept']
listBioTags= []
for i in range(len(indexConceptsInSentences)):
    tempList = []
    tempList.append(list(itertools.repeat(bioTags[0],indexConceptsInSentences[i][0])))
    tempList.append(list(itertools.repeat(bioTags[1],1)))
    tempList.append(list(itertools.repeat(bioTags[2],indexConceptsInSentences[i][1]- indexConceptsInSentences[i][0])))
    tempList.append(list(itertools.repeat(bioTags[0],len(tokenizedSentences[i])- indexConceptsInSentences[i][1]- 1)))
    tempList = [val for sublist in tempList for val in sublist]
    listBioTags.append(tempList)
    
# Write token, POS and BIO tag in CSV
flatTokenizedSentences = []
for element in tokenizedSentences:
    for eachElement in element:
        flatTokenizedSentences.append(eachElement)
    flatTokenizedSentences.append('')

flatListPosTags= []
for eachPosTaggedSentence in posTaggedTokens:
    for eachPosTaggedToken in eachPosTaggedSentence:
        flatListPosTags.append(eachPosTaggedToken[1])
    flatListPosTags.append('')

flatListBioTags= []
for item in listBioTags:
    for eachItem in item:
        flatListBioTags.append(eachItem)
    flatListBioTags.append('')    

trainDataCRF= zip(flatTokenizedSentences, flatListPosTags, flatListBioTags)
with open('./output/trainCRF.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for row in trainDataCRF:
        writer.writerow(row)
        
# Read full articles from the bioscope dataset
bioScopeFullArticle= open('./output/full_papers.txt') 
listBioScopeFullArticle= []
for eachLine in bioScopeFullArticle:
    listBioScopeFullArticle.append(eachLine.decode('utf-8')) # Lines should be explicitly decoded to unicode. Else, python throws UnicodeDecode Error.
tokensBioScope= getTokens(listBioScopeFullArticle)
print len(tokensBioScope)

# Merge the lists to obtain a bigger dataset
trainingData = tokenizedSentences + tokensBioScope
##trainingData = tokenizedSentences

# Import the built-in logging module and configure it so that Word2Vec creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 2    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 5           # Context window size                                                                                    
downsampling = 1e-15   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(trainingData, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_2minwords_2context"
model.save(model_name)

