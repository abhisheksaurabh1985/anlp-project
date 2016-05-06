from __future__ import division
from collections import OrderedDict
from nltk.tokenize import word_tokenize
import itertools
import math
import csv
import nltk
import re
import random

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

def getTxtInDictionary(txtFileName):
    reader= csv.DictReader(open(txtFileName), delimiter="	")
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
        tokenizedSentences.append(word_tokenize(re.sub("/", " ", eachSentence.lower())))
    return tokenizedSentences

def sublistIndex(sublist, origlist):
    origstr = ' '.join(map(str, origlist))
    ind =  origstr.index(' '.join(map(str, sublist)))
    return len(origstr[:ind].split(' ')) - 1


        
def getTrainingDataForCRF(tokenizedSentences, tokenizedConcepts, bioTags):

    indexConceptsInSentences= []
    exceptions = []
    for i in range(len(tokenizedConcepts)):
        temp = []
        start = tokenizedConcepts[i][0]
        end = tokenizedConcepts[i][-1]
        if(start in tokenizedSentences[i]):
            if(end in tokenizedSentences[i]):
                # this will not work if there are multiple occurences of concept in a sentence!
                ind = sublistIndex(tokenizedConcepts[i], tokenizedSentences[i])
                temp.append(ind)
                temp.append(ind + len(tokenizedConcepts[i]) - 1)
                indexConceptsInSentences.append(temp)
            else:
                print("concept end '%s' is not in %i sentence; ignore this sentence" % (end, i))
                exceptions.append(i)
        else:
            print("concept start '%s' is not in %i sentence; ignore this sentence" % (start, i))
            exceptions.append(i)

    for index in sorted(exceptions, reverse=True):
        del tokenizedSentences[index]

    # Get POS tags for each of the sentences
    posTaggedTokens = []
    for i in range(len(tokenizedSentences)):
        posTaggedTokens.append(nltk.pos_tag(tokenizedSentences[i]))

    listBioTags= []
    for i in range(len(indexConceptsInSentences)):
        tempList = []
        startIndex = indexConceptsInSentences[i][0]
        endIndex = indexConceptsInSentences[i][1]
        tempList.append(list(itertools.repeat(bioTags[0],startIndex)))
        tempList.append(list(itertools.repeat(bioTags[1],1)))
        tempList.append(list(itertools.repeat(bioTags[2],endIndex- startIndex)))
        tempList.append(list(itertools.repeat(bioTags[0],len(tokenizedSentences[i])- endIndex- 1)))
        tempList = [val for sublist in tempList for val in sublist]
        listBioTags.append(tempList)

    # Write token, POS and BIO tag in CSV
    flatTokenizedSentences = []
    for i in xrange(len(tokenizedSentences)):
        for eachElement in tokenizedSentences[i]:
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
        writer = csv.writer(csvfile, delimiter=" ")
        for row in trainDataCRF:
            if(row[0].strip()==row[1].strip()==row[2].strip() == ''):
                print(row)
                print(" ignored")
                continue
            writer.writerow(row)
    
    return posTaggedTokens, indexConceptsInSentences, listBioTags


def splitDataForValidation(fileNameCRFData, percentTest):
    dataCRF= []
    blankLine = [' \n']
    with open(fileNameCRFData) as f:
        temp= []
##        pdb.set_trace()
        for line in f:
##            print line
            if line.strip():
                temp.append(line)	
            else:
                dataCRF.append(temp + blankLine)
                temp= []                

    print len(dataCRF)
    # Split into train and test
    countSentencesTest = math.ceil((percentTest/ 100)* len(dataCRF))
    random.shuffle(dataCRF)
    trainingDataCRF= dataCRF[0:len(dataCRF)- int(countSentencesTest)]
    testDataCRF = [x for x in dataCRF if x not in trainingDataCRF]
    print len(trainingDataCRF)
    print len(testDataCRF)
    # Write in text file
##    try:
##        with open('./output/trainingDataCRF.txt', 'w') as file:
##            file.writelines(''.join(i) for i in trainingDataCRF)
##
##        with open('./output/testDataCRF.txt', 'w') as file:
##            file.writelines(''.join(i) for i in testDataCRF)
##    except Exception:
##        pass
    with open('./crf-test/trainingDataCRF.txt', 'w') as file:
        for i in trainingDataCRF:
            if i is not None:
                file.writelines(''.join(i))

    with open('./crf-test/testDataCRF.txt', 'w') as file:
        for i in testDataCRF:
            if i is not None:
                file.writelines(''.join(i))

    return dataCRF, trainingDataCRF, testDataCRF
