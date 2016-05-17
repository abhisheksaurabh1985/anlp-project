from __future__ import division
from collections import OrderedDict
from nltk.tokenize import word_tokenize
import itertools
import math
import csv
import re
import random
import numpy as np
from nltk.corpus import conll2000
from ChunkParser import *

defaultTag = "-"
# Training a chunk parser on conll200 corpus
train_sents = conll2000.chunked_sents('train.txt')
# training the chunker, ChunkParser is a class defined in the next slide
NPChunker = ChunkParser(train_sents)


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
    beginIndices =  [i for i, x in enumerate(origlist) if x == sublist[0]]
    badBeginIndices = [] # indices of list beginIndices
    for i in xrange(len(beginIndices)):
        ind = beginIndices[i]
        for j in xrange(1, len(sublist)):
            if (j + ind) >=len(origlist) or origlist[j+ind] != sublist[j]:
                badBeginIndices.append(i)
                break
    for i in sorted(badBeginIndices,reverse=True):
        del beginIndices[i]
    if len(beginIndices) == 0:
        return -1
    return beginIndices[0]

def mergeListsOfTags(lists, priorities):
    if len(lists) == 0:
        return lists
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

def getTrainingDataForCRF(tokenizedSentences_orig, tokenizedConcepts_orig,
                          negations_orig, bioTags, priorities,
                          cueTypeTags_orig):

    negations = list(negations_orig)
    tokenizedSentences = list(tokenizedSentences_orig)
    cueTypeTags = list(cueTypeTags_orig)
    tokenizedConcepts = list(tokenizedConcepts_orig)
    (uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences, cueTypeTagsForUniqueSentences) =\
        makeUnique(tokenizedSentences, tokenizedConcepts, negations, cueTypeTags)

    listBioTags = []
    listNegTags = []
    tagB = bioTags[1]
    tagI = bioTags[2]

    notFoundConceptIndices = []

    # finding indices for unique concepts in unique sentences
    for i in range(len(uniqueSentenses)):
        sentence = uniqueSentenses[i]
        listsBioTagsForSentence = []
        listsNegTagsForSentence = []
        for j in xrange(len(conceptsForUniqueSentences[i])):
            concept = conceptsForUniqueSentences[i][j]
            startInd = sublistIndex(concept, sentence)
            endInd = startInd + len(concept) - 1
            tempList = list(itertools.repeat(bioTags[0],len(sentence)))
            tempListNeg = list(itertools.repeat(defaultTag,len(sentence)))

            if (startInd == -1):
                print("concept '%s' is not in sentence '%s'; ignore this concept" % (' '.join(concept), ' '.join(sentence)))
                notFoundConceptIndices.append((i, j))
                listsBioTagsForSentence.append(tempList)
                listsNegTagsForSentence.append(tempListNeg)
                continue

            if negationsForUniqueSentences[i][j] == "Negated":
                negTag = "N"
            else:
                negTag = "A"

            tempList[startInd] = tagB
            if(startInd != endInd):
                for j in xrange(startInd + 1, endInd+1):
                    tempList[j] = tagI
            for j in xrange(startInd, endInd+1):
                tempListNeg[j] = negTag

            listsBioTagsForSentence.append(tempList)
            listsNegTagsForSentence.append(tempListNeg)
        listBioTags.append(mergeListsOfTags(listsBioTagsForSentence, priorities))
        listNegTags.append(mergeListsOfTags(listsNegTagsForSentence, {defaultTag: 0, "N":1, "A":1}))

    for indices in sorted(notFoundConceptIndices, reverse=True):
        del conceptsForUniqueSentences[indices[0]][indices[1]]
        del negationsForUniqueSentences[indices[0]][indices[1]]

    listCueTypeTags= []
    for i in range(len(cueTypeTagsForUniqueSentences)):
        cueTypeTagsForSentence = cueTypeTagsForUniqueSentences[i]
        sentence = uniqueSentenses[i]
        tempList = list(itertools.repeat(defaultTag,len(sentence)))
        for cueTypeTagsSet in cueTypeTagsForSentence:
            for tag in cueTypeTagsSet.keys():
               for indexes in cueTypeTagsSet[tag]:
                   tempList[indexes[0]] = tag + "-B"
                   if(indexes[0] != indexes[1]):
                       for j in xrange(indexes[0]+1, indexes[1]+1):
                           tempList[j] = tag + "-I"
        listCueTypeTags.append(tempList)


    # Write token, POS and BIO tag in CSV
    flatTokenizedSentences = []
    flatListPosTags = []
    flatListBioTags = []
    flatListNegTags = []
    flatCueTypeTags = []
    flatIsPunctuation = []

    for i in xrange(len(uniqueSentenses)):
        sentence = uniqueSentenses[i]
        posTaggedTokens = nltk.pos_tag(sentence)
        for j in xrange(len(sentence)):
            token = sentence[j]

            flatTokenizedSentences.append(token)
            flatListPosTags.append(posTaggedTokens[j][1])
            flatListBioTags.append(listBioTags[i][j])
            flatListNegTags.append(listNegTags[i][j])
            flatCueTypeTags.append(listCueTypeTags[i][j])
            if(isPunctuation(token)):
                flatIsPunctuation.append('PUNCT')
            else:
                flatIsPunctuation.append(defaultTag)
        flatTokenizedSentences.append('')
        flatListPosTags.append('')
        flatListBioTags.append('')
        flatListNegTags.append('')
        flatCueTypeTags.append('')
        flatIsPunctuation.append('')

    flatChunkTags = getChunks(uniqueSentenses)

    flatSegmentTags = getSegmentTags(uniqueSentenses)

    trainDataCRF= zip(flatTokenizedSentences, flatListPosTags, flatCueTypeTags,
                      flatIsPunctuation, flatChunkTags, flatSegmentTags,
                      flatListBioTags, flatListNegTags)

    return (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences)


def isPunctuation(token):
    matched = re.match("[^\w]*", token)
    return matched != None and matched.group(0) == token

def getCRFDataFromFile(fileNameCRFData, verbose =False):
    dataCRF= []
    with open(fileNameCRFData, 'r') as f:
        temp= []
        for line in f:
            temp.append(line)
            if not line.strip():
                dataCRF.append(temp)
                temp= []

    if verbose: print len(dataCRF)
    return dataCRF

def kfoldCrossValidation(fileNameCRFData, k, verbose=False):
    dataCRF = getCRFDataFromFile(fileNameCRFData)
    # Split into train and test
    trainingDataCRF = []
    testDataCRF = []

    random.shuffle(dataCRF)
    l = len(dataCRF)
    s = int(l/k)

    for i in xrange(k):
        if i == k-1:
            endTestInd = l-1
        else:
            endTestInd = (i+1)*s
        trainFold = dataCRF[0:s*i] + dataCRF[endTestInd:l-1]
        testFold = dataCRF[s*i:endTestInd]
        trainingDataCRF.append(trainFold)
        testDataCRF.append(testFold)

    if verbose:
        print len(trainingDataCRF)
        print len(testDataCRF)

    return dataCRF, trainingDataCRF, testDataCRF

def splitDataForValidation(fileNameCRFData, percentTest, verbose=False):
    dataCRF = getCRFDataFromFile(fileNameCRFData)
    # Split into train and test
    countSentencesTest = math.ceil((percentTest/ 100)* len(dataCRF))
    random.shuffle(dataCRF)

    trainingDataCRF= dataCRF[0:len(dataCRF)- int(countSentencesTest)]
    testDataCRF = [x for x in dataCRF if x not in trainingDataCRF]

    if verbose:
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


def getChunks(tokenizedSentences):

    chunksTags = []
    for sentence in tokenizedSentences:
        inputSentence = nltk.pos_tag(sentence)
        chunksForSentence = NPChunker.parse(inputSentence)

        for chunk in chunksForSentence:
            if(type(chunk) == nltk.tree.Tree):
                if (len(chunk) == 0):
                    continue
                label = chunk.label()
                chunksTags.append(label + "-B")
                chunksTags.extend([label + "-I"]*(len(chunk)-1))
            else:
                chunksTags.append(defaultTag)
        chunksTags.append('')
    return chunksTags

def getSegmentTags(sentences):
    tags = []
    for sentence in sentences:
        segmNum = 0
        for i in xrange(len(sentence)):
            token = sentence[i]
            if token in [',','.',':',';','-']:
                tags.append(defaultTag)
                if i > 0:
                    segmNum += 1
            else:
                tags.append(str(segmNum))

        tags.append('')
    return tags

def getConfusionMatrix(tokenizedConcepts, concepts, tokenizedSentences, sentences, negations, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ignored = 0
    for k in xrange(len(sentences)):
        sentence = sentences[k]
        labelsPredicted = labels[k]
        conceptsPredicted = concepts[k]
        indices = [i for i, x in enumerate(tokenizedSentences) if x == sentence]

        conceptsReal = [tokenizedConcepts[i] for i in indices]
        labelsReal = [negations[i] for i in indices]

        conceptsReal = [item for sublist in conceptsReal for item in sublist]
        labelsReal = [item for sublist in labelsReal for item in sublist]
        for j in xrange(len(conceptsReal)):
            concept = conceptsReal[j]
            if not(concept in conceptsPredicted):
                print("concept %s not found"%' '.join(concept))
                ignored+=1
                continue
            ind = conceptsPredicted.index(concept)
            if (labelsPredicted[ind] == labelsReal[j]):
                if(labelsPredicted[ind] == "Negated"):
                    TN+=1
                else:
                    TP+=1
            else:
                if(labelsPredicted[ind] == "Negated"):
                    FN+=1
                else:
                    FP+=1

    return ([[TP, FN], [FP, TN]], ignored)


def makeUnique(sentences, concepts, negations, cueTypeTags):
    uniqueSentenses = list(map(list, set(map(tuple, np.unique(sentences)))))
    conceptsForUniqueSentences = []
    negationsForUniqueSentences = []
    cueTypeTagsForUniqueSentence = []

    for sentence in uniqueSentenses:
        indices = [k for (k, j) in enumerate(sentences) if j == sentence]
        allConceptsForSentence = [concepts[i] for i in indices]
        allNegationsForSentence = [negations[i] for i in indices]
        allCueTypeTags = [cueTypeTags[i] for i in indices]
        uniqueConcepts, uniqueNegations, uniqueCueTypeTags =\
            mergeConcepts(allConceptsForSentence, allNegationsForSentence, allCueTypeTags)
        conceptsForUniqueSentences.append(uniqueConcepts)
        negationsForUniqueSentences.append(uniqueNegations)
        cueTypeTagsForUniqueSentence.append(uniqueCueTypeTags)

    return (uniqueSentenses, conceptsForUniqueSentences,
            negationsForUniqueSentences, cueTypeTagsForUniqueSentence)

def mergeConcepts(concepts, negations, cueTypeTags):
    resultConcepts = []
    resultNegations = []
    resultCueTypeTags = []
    subconceptsIndices = []
    for i in xrange(len(concepts)):
        concept = concepts[i]

        # first check if this concept is subconcept of other
        for c in concepts:
            ind = sublistIndex(concept, c)
            if c != concept and ind != -1:
                subconceptsIndices.append(i)
                break
    # get rid of subconcepts
    for i in xrange(len(concepts)):
        if not (i in subconceptsIndices):
            resultConcepts.append(concepts[i])
            resultNegations.append(negations[i])
            resultCueTypeTags.append(cueTypeTags[i])

    # leave only unique concepts
    uniqueConcepts = list(map(list, set(map(tuple,resultConcepts))))
    indices = []
    for concept in uniqueConcepts:
        indices.append(resultConcepts.index(concept))
    uniqueNegations = [resultNegations[i] for i in indices]
    uniqueCueTypeTags = [resultCueTypeTags[i] for i in indices]
    return (uniqueConcepts, uniqueNegations, uniqueCueTypeTags)



