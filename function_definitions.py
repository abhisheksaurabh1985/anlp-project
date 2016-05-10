from __future__ import division
from collections import OrderedDict
from nltk.tokenize import word_tokenize
import itertools
import math
import csv
import nltk
import re
import random
import numpy as np

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

def getTriggers(triggersFileName):
    triggers = []
    for line in open(triggersFileName):
        triggers.append(line.strip().split('\t\t'))
    return triggers

def getTokens(listSentences):
    '''
    Description: Function to get tokens from sentences. Word2Vec expects single sentences, each one of them as a list of words.
    Input: List of sentences.
    Output: A nested list with the inner list containing a sentence in a tokenized form. All the tokens are in lowercase.
    '''
    tokenizedSentences = []
    for eachSentence in listSentences:
        tokenizedSentences.append(getTokensFromSentence(eachSentence))
    return tokenizedSentences

def getTokensFromSentence(sentence):
    return  word_tokenize(re.sub("[/_]", " ", sentence.lower()))

def sublistIndex(sublist, origlist):
    origstr = ' '.join(map(str, origlist))
    substr = ' '+' '.join(map(str, sublist))+' '
    if not (substr in origstr):
       return -1
    ind =  origstr.index(substr)
    return len(origstr[:ind].split(' '))

def mergeListsOfTags(lists, priorities):
    res = []
    for i in xrange(len(lists[0])):
        tag = lists[0][i]
        for l in lists[1:]:
            if(priorities[l[i]] > priorities[tag]):
                tag = l[i]
        res.append(tag)
    return res

def writeLinesToFile(data, fileName):
    with open(fileName, 'w') as file:
        for i in data:
            if i is not None:
                file.writelines(''.join(i))

def writeCsvToFile(data, fileName):
    with open(fileName, 'wb') as csvfile:
        for row in data:
            csvfile.write(" ".join(row) + "\n")

def getTrainingDataForCRF(tokenizedSentences_orig, tokenizedConcepts,
                          negations_orig, bioTags, priorities, consideredConcepts,
                          cueTypeTags_orig):
    indexConceptsInSentences= []
    exceptions = []
    negations = list(negations_orig)
    tokenizedSentences = list(tokenizedSentences_orig)
    cueTypeTags = list(cueTypeTags_orig)

    for i in range(len(tokenizedConcepts)):
        temp = []
        if(negations[i] == consideredConcepts):
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
        else:
            indexConceptsInSentences.append([0,0])

     # Get rid of the ignored sentences
    for index in sorted(exceptions, reverse=True):
        del tokenizedSentences[index]
        del negations[index]
        del cueTypeTags[index]

    # Get POS tags for each of the sentences
    posTaggedTokens = []
    for i in range(len(tokenizedSentences)):
        posTaggedTokens.append(nltk.pos_tag(tokenizedSentences[i]))

    listBioTags= []
    tagB = bioTags[1]
    tagI = bioTags[2]

    for i in range(len(indexConceptsInSentences)):
        if(negations[i] == consideredConcepts):
            startIndex = indexConceptsInSentences[i][0]
            endIndex = indexConceptsInSentences[i][1]

            tempList = list(itertools.repeat(bioTags[0],len(tokenizedSentences[i])))
            tempList[startIndex] = tagB
            if(startIndex != endIndex):
                for j in xrange(startIndex + 1, endIndex+1):
                    tempList[j] = tagI
        else:
            tempList = list(itertools.repeat(bioTags[0],len(tokenizedSentences[i])))
        listBioTags.append(tempList)

    listTriggerTags= []
    for i in range(len(cueTypeTags)):
        tempList = list(itertools.repeat('NONE',len(tokenizedSentences[i])))
        for tag in cueTypeTags[i].keys():
           for indexes in cueTypeTags[i][tag]:
               tempList[indexes[0]] = tag + "-B"
               if(indexes[0] != indexes[1]):
                   for j in xrange(indexes[0]+1, indexes[1]+1):
                       tempList[j] = tag + "-I"
        listTriggerTags.append(tempList)

    uniqueSentenses = list(np.unique(tokenizedSentences))

    # Write token, POS and BIO tag in CSV
    flatTokenizedSentences = []
    for i in xrange(len(uniqueSentenses)):
        for eachElement in uniqueSentenses[i]:
            flatTokenizedSentences.append(eachElement)
        flatTokenizedSentences.append('')

    flatListPosTags= []
    for i in xrange(len(uniqueSentenses)):
        ind = tokenizedSentences.index(uniqueSentenses[i])
        for eachPosTaggedToken in posTaggedTokens[ind]:
            flatListPosTags.append(eachPosTaggedToken[1])
        flatListPosTags.append('')

    flatListBioTags= []
    for i in xrange(len(uniqueSentenses)):
        selectedBioTagsList = [listBioTags[k] for k, j in enumerate(tokenizedSentences) if j == uniqueSentenses[i]]
        selectedBioTags = mergeListsOfTags(selectedBioTagsList, priorities)
        for item in selectedBioTags:
            flatListBioTags.append(item)
        flatListBioTags.append('')

    flatCueTypeTags = []
    for i in xrange(len(uniqueSentenses)):
        ind = tokenizedSentences.index(uniqueSentenses[i])
        for eachTriggerTag in listTriggerTags[ind]:
            flatCueTypeTags.append(eachTriggerTag)
        flatCueTypeTags.append('')
    flatIsPunctuation = []
    for sentence in uniqueSentenses:
        for token in sentence:
            if(isPunctuation(token)):
                flatIsPunctuation.append('PUNCT')
            else:
                flatIsPunctuation.append('NONE')
        flatIsPunctuation.append('')

    trainDataCRF= zip(flatTokenizedSentences, flatListPosTags, flatCueTypeTags, flatIsPunctuation, flatListBioTags)

    return trainDataCRF


def isPunctuation(token):
    matched = re.match("[^\w]*", token)
    return matched != None and matched.group(0) == token

def splitDataForValidation(fileNameCRFData, percentTest):
    dataCRF= []
    with open(fileNameCRFData, 'r') as f:
        temp= []
        for line in f:
            temp.append(line)
            if not line.strip():
                dataCRF.append(temp)
                temp= []

    print len(dataCRF)
    # Split into train and test
    countSentencesTest = math.ceil((percentTest/ 100)* len(dataCRF))
    random.shuffle(dataCRF)
    trainingDataCRF= dataCRF[0:len(dataCRF)- int(countSentencesTest)]
    testDataCRF = [x for x in dataCRF if x not in trainingDataCRF]

    print len(trainingDataCRF)
    print len(testDataCRF)

    return dataCRF, trainingDataCRF, testDataCRF


def extractCueTypeTags(tokenizedSentences, triggers):
    tagsExtracted = []
    triggerTokens = []
    for i in xrange(len(triggers)):
       triggerTokens.append(getTokensFromSentence(triggers[i][0]))

    for sentence in tokenizedSentences:
        tagsForSentence = {}
        for i in xrange(len(triggers)):
            ind = sublistIndex(triggerTokens[i], sentence)
            if(ind != -1):
                tag = triggers[i][1][1:-1]
                if not(tag in tagsForSentence.keys()):
                   tagsForSentence[tag] = []
                tagsForSentence[tag].append((ind, ind + len(triggerTokens[i])-1))

        tagsExtracted.append(tagsForSentence)

    return tagsExtracted
