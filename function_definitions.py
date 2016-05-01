from collections import OrderedDict
import csv
import nltk
from nltk.tokenize import word_tokenize
import itertools

# External libraries
from nltk.tokenize import word_tokenize

def getCSVInDictionary(csvFileName):
    '''
    Description: Read a multi column CSV file with first row as header. File content is stored in a dictionary which the function returns. 
    Input: File path along with the file name.
    Output: An ordered dictionary with file header as KEY. Dictionary values are a list which hold the column values corresponding to each header.
    For e.g. {'header A': ['val_1_header_A', 'val_2_header_A'], 'header B': ['val_1_header_B', 'val_1_header_B']}   
    '''
    reader= csv.DictReader(open(csvFileName))
    result= OrderedDict()
    for row in reader:
        for column, value in row.iteritems():
            result.setdefault(column, []).append(value)
    return result
    
def getTokens(listSentences):
    '''
    Description: Function to get tokens from sentences. Word2Vec expects single sentences, each one of them as a list of words.
    Input: List of sentences.
    Output: A nested list with the inner list containing a sentence in a tokenized form. All the tokens are in lowercase.
    '''
    tokenizedSentences = []
    for eachSentence in listSentences:
        tokenizedSentences.append(word_tokenize(eachSentence.lower()))
    return tokenizedSentences
        
def getTrainingDataForCRF(tokenizedSentences, tokenizedConcepts, bioTags):
    # Get POS tags for each of the sentences
    posTaggedTokens = []
    for eachTokenizedSentence in tokenizedSentences:
        posTaggedTokens.append(nltk.pos_tag(eachTokenizedSentence))

    indexConceptsInSentences= []
    for i in range(len(tokenizedConcepts)):
        temp = []
        temp.append(tokenizedSentences[i].index(tokenizedConcepts[i][0]))
        temp.append(tokenizedSentences[i].index(tokenizedConcepts[i][-1]))
        indexConceptsInSentences.append(temp)
        
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
    
    return posTaggedTokens, indexConceptsInSentences, listBioTags
