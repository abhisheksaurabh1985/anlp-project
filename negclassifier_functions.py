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
from negclassifies_classes import *
from subprocess import Popen, PIPE

# Default tag which is used if the feature is absent
defaultTag = "-"


def getTxtInDictionary(txtFileName):
    '''
    Read a multi column CSV file with first row as header. File content is stored in a dictionary which the function returns.
    :param txtFileName: File path along with the file name.
    :return: An ordered dictionary with file header as KEY. Dictionary values are a list which hold the column values corresponding to each header.
    For e.g. {'header A': ['val_1_header_A', 'val_2_header_A'], 'header B': ['val_1_header_B', 'val_1_header_B']}
    '''
    reader= csv.DictReader(open(txtFileName), delimiter="	")
    result= OrderedDict()
    for row in reader:
        for column, value in row.iteritems():
            result.setdefault(column, []).append(value)
    return result

def getTriggers(triggersFileName):
    '''
    Read a trigger table from a specified file;
    Each line of the trigger file should be in format 'trigger tag'
    :param triggersFileName: name of a file
    :return: list, each element of it is a list of 2 elements: trigger and tag (both strings)
    '''
    triggers = []
    for line in open(triggersFileName):
        triggers.append(line.strip().split('\t\t'))
    return triggers

def getTokens(listSentences):
    '''
    Function to get tokens from sentences. Word2Vec expects single sentences, each one of them as a list of words.
    :param listSentences: list of sentences
    :return: A nested list with the inner list containing a sentence in a tokenized form. All the tokens are in lowercase.
    '''
    tokenizedSentences = []
    for eachSentence in listSentences:
        tokenizedSentences.append(getTokensFromSentence(eachSentence))
    return tokenizedSentences

def getTokensFromSentence(sentence):
    '''
    Retrive a list of tokens in a lower case from a string
    :param sentence: string
    :return: list of strings
    '''
    return  word_tokenize(re.sub("[/_]", " ", sentence.lower()))

def sublistIndex(sublist, origlist):
    '''
    Search for a sublist position in a list
    :param sublist:
    :param origlist:
    :return: begin index of first occurence of sublist
    '''
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
    '''
    Merge several lists of tags of the same lengths according to the priorities,
    if several different tags are present then the one with the higher priorities is chosen
    :param lists: list of lists to merge
    :param priorities: dictionary, for each tag has an integer value
    :return: merged list
    '''
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
    '''
    Write list of strings to a file
    :param data: list of strings
    :param fileName:
    :return: -
    '''
    with open(fileName, 'w') as file:
        for i in data:
            if i is not None:
                file.writelines(''.join(i))

def writeCsvToFile(data, fileName):
    '''
    Write list of strings to a csv file
    :param data: list of strings
    :param fileName:
    :return: -
    '''
    with open(fileName, 'wb') as csvfile:
        for row in data:
            csvfile.write(" ".join(row) + "\n")

def getTrainingDataForCRF(tokenizedSentences_orig, tokenizedConcepts_orig,
                          negations_orig, bioTags, priorities,
                          cueTypeTags_orig):
    '''
    Turn a list of tokenized sentences with tokenized concepts and negation labels into a list
    of features for each token. Also extracts unique sentences along with concepts and negation labels.
    :param tokenizedSentences_orig: list each element of which is a list of tokens which represents
    the sentence
    :param tokenizedConcepts_orig: list each element of which is a list of tokens which represents
    the sentence
    :param negations_orig: list each element of which is the negation label: "Affiremed" or "Negated
    :param bioTags: list of possible tags for tagging the concepts (like ['B', 'I', 'O'])
    :param priorities: dictionary of priorities of bioTags for merging, see function 'mergeListsOfTags'
    :param cueTypeTags_orig: preextracted tags for prenegeations and postnegations cues
    :return: a tuple of 4 values:
     trainDataCRF: list of tuples with extracted features: (token, postag, ...)
     uniqueSentenses: list the tokenized sentences from the input which are unique
     conceptsForUniqueSentences: list of tokenized concepts which correspond to the unique sentences
     negationsForUniqueSentences: list of negation labels which correspond to the unique sentences
    '''
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
    '''
    Determine whether the token is a punctuation mark
    :param token: token
    :return: boolean
    '''
    matched = re.match("[^\w]*", token)
    return matched != None and matched.group(0) == token

def getCRFDataFromFile(fileNameCRFData, verbose =False):
    '''
    Read crf data from file. The input file should be in format accepted by crf++
    The output is the list each element of which is a list of tokens from the same sentence
    along with its features
    :param fileNameCRFData:
    :param verbose:
    :return:  list each element of which is a list of tokens from the same sentence
    along with its features
    '''
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
    '''
    Split input file for crf++ into k folds
    :param fileNameCRFData: input file name
    :param k: number of folds to split the data
    :param verbose:
    :return: tuple of three elements:
     dataCRF - list each element of which is a list of tokens from the same sentence
     trainingDataCRF - list with the same structure as first (dataCRF) but for k training sets
     testDataCRF - list for k test sets
    '''
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
    '''
    Split input file for crf++ into test and training part
    :param fileNameCRFData: input file name
    :param percentTest: percentage for test split
    :param verbose:
    :return: tuple of three elements:
     dataCRF - list each element of which is a list of tokens from the same sentence
     trainingDataCRF - list with the same structure as first (dataCRF) but for training set
     testDataCRF - list for test set
    '''
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
    '''
    Extract prenegation and postnegation cues for tokens
    :param tokenizedSentences: list, each element of it is a list of tokens for the sentence
    :param triggers: triggers which are considered as prenegations and postnegations
    :return: list
        for each tokenized sentence the output list contains a list of corresponding cues
    '''
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


# Training a chunk parser on conll200 corpus
train_sents = conll2000.chunked_sents('train.txt')

# training the chunker, ChunkParser is a class defined in the next slide
NPChunker = ChunkParser(train_sents)

def getChunks(tokenizedSentences):
    '''
    For a list of tokenized sentences create the list of corresponding syntactic chunks
    (as noun phrase, verb phrase and others)
    :param tokenizedSentences:
    :return:
        list of chunk tags
        for each chunk tag postfix is added :
            "-B" for the first token in chunk
            "-I" for other tokens
        if the token doesn't have an associated chunk the default tag is assigned
    '''

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
    '''
    For a list of tokenized sentences create a list of segment tags.
    Each tag is an integer sequential number of the segment.
    Segment are separated by punctuation marks.
    :param sentences:  list of tokenized sentences
    :return: list of segment tags
    '''
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

def getConfusionMatrix(allConcepts, concepts, allSentences, sentences, allNegations, negations):
    '''
    Computes confusion matrix for the task of binary classification of the concepts
    :param allConcepts: list of all tokenized concepts present in the dataset
    :param concepts: list of tokenized concepts for which the prediction is made
    :param allSentences: list of all tokenized sentences present in the dataset
    :param sentences: list of tokenized sentencets for which the predicion is made
    :param allNegations: list of negation labels for each known concepts
    :param negations: list of predicted negation labels
    :return: a tuple:
     list of lists which corresponds to confusion matrix
     number of ignored concepts i.e. concepts not found in original dataset
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ignored = 0
    for k in xrange(len(sentences)):
        sentence = sentences[k]
        labelsPredicted = negations[k]
        conceptsPredicted = concepts[k]
        indices = [i for i, x in enumerate(allSentences) if x == sentence]

        conceptsReal = [allConcepts[i] for i in indices]
        labelsReal = [allNegations[i] for i in indices]

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
    '''
    From the list of tokenized sentences, tokenized concepts, negation labels and pre- and postnegation cue tags
    extract a set of unique sentences along with the set of unique concepts (each of which is not a subset of
    the other concept for this sentence), negation labels and cue tags
    :param sentences: list of tokenized sentences
    :param concepts: list of tokenized concepts
    :param negations: list of corresponding negations
    :param cueTypeTags: list of corresponding pre- and postnegation cue tags
    :return:
    '''
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
    '''
    For a list of tokenized concepts for a sentence perform two operations:
        1) get rid of all concepts which are subconcepts of others (subset of tokens)
        2) get rid of repeated concepts, i.e. concepts which contain the same set of
        tokens in the same order
    :param concepts: list of tokenized concepts
    :param negations: list of negation tags
    :param cueTypeTags: list of prenegation and postnegation cue tags
    :return: list of filtered tokenized concepts along with corresponding negation tags and cue type tags
    '''
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


def classifyConcepts(filename):
   '''
   From the output file of crf++ collect the sentences, concepts and predicted labels
   :param filename: output file from crf++
   :return: tuple:
   list of tokenized sentences
   list of tokenized concepts
   list of predicted labels
   '''
   concepts = []
   sentences = []
   labels = []

   #  number of features
   N = 9

   sentence = []
   concept = []
   labelTags = []
   currentConcepts = []
   currentLabels = []

   conceptTags = ['O','B','I']

   f = open(filename)

   for line in f:
      if not line.split():
         sentences.append(sentence)
         sentence = []
         if len(concept):
            currentConcepts.append(concept)
            currentLabels.append(getLabelFromTags(labelTags))
         concepts.append(currentConcepts)
         currentConcepts = []
         concept = []
         labels.append(currentLabels)
         currentLabels = []
         labelTags = []
      else:
         parts = re.split("\t|\n", line)
         conceptPart = parts[N-3]
         classPart = parts[N-1]
         tokenPart = parts[0]

         sentence.append(tokenPart)

         if conceptPart == conceptTags[1]:
            if len(concept) > 0:
               currentConcepts.append(concept)
               concept = []
               currentLabels.append(getLabelFromTags(labelTags))
               labelTags = []
            concept = [tokenPart]
            labelTags.append(classPart)
         elif conceptPart == conceptTags[2]:
            concept.append(tokenPart)
            labelTags.append(classPart)
         else:
            if len(concept) > 0:
               currentConcepts.append(concept)
               concept = []
               currentLabels.append(getLabelFromTags(labelTags))
               labelTags = []

   f.close()

   return (concepts, sentences, labels)

def getLabelFromTags(labelTags):
   '''
   From a list of negation tags predicted for a concept make 1 prediction about
   whether the concept is affirmed of negated.
   :param labelTags: lis of tags predicted for a concept
   :return: one of values : "Affirmed", "Negated"
   '''
   if "N" in labelTags:
      return "Negated"
   return "Affirmed"


def createInputForCRF(inputFilename, outputFilename):
   '''
   Read the original dataset and make crf input from it
   :param inputFilename: original dataset
   :param outputFilename: file to write the input for crf
   :return: a tuple
    trainDataCRF - list of tuples with features for train data, each tuple looks like ('token', 'postag', ...)
    uniqueSentenses - list of unique tokenized sentences
    conceptsForUniqueSentences - list of unique set concepts for each of unique sentences
    (each concept is not a subset of other concepts)
    negationsForUniqueSentences - list of negation labels for each concept
   '''

   # Read the annotated data in an Ordered Dictionary
   txtFileName= './data/Annotations-1-120_orig.txt'
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
   (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences) = \
      getTrainingDataForCRF(tokenizedSentences,
                            tokenizedConcepts, negations,
                            bioTags, priorities, cueTypeTags)
   # write the data to file
   writeCsvToFile(trainDataCRF, outputFilename)
   print "%s created" % outputFilename
   return (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences)

def trainCRF(crfCmd, templatePath, trainingFileName, modelPath):
   '''
   Execute the command to run crf script for training
   :param crfCmd: list with params to run a training script, for ex.
    ['./crf_learn', '-c', str(crfCParam), '-f', str(crfFParam), "-a", "MIRA"]
   :param templatePath:
    path to the template file for the crf
   :param trainingFileName:
    path to the training file for the crf
   :param modelPath:
    path where crf will save the model
   :return: a tuple:
    returncode from a process, integer
    stderr, error output from a process, string
    stout, output from a process, string
   '''
   otherParams = [templatePath, trainingFileName, modelPath]
   # Run the tagger and get the output
   p = Popen(crfCmd + otherParams, stdin=PIPE, stdout=PIPE, stderr=PIPE)
   (stdout, stderr) = p.communicate()
   return (p.returncode, stderr, stdout)

def testCRF(crfCmd, modelPath, testFileName, outputFileName):
   '''
   Run the script to evalutate the model. The output is written to
   :param crfCmd: list of parameters to run the test script, for ex.
    ['./crf_test']
   :param modelPath: path to the model
   :param testFileName: file to perform test
   :param outputFileName: file to save a model
   :return: -
   '''
   otherParams = ['-m', modelPath, testFileName]
   p = Popen(crfCmd + otherParams, stdin=PIPE, stdout=PIPE, stderr=PIPE)
   (stdout, stderr) = p.communicate()

   predictions = stdout
   f = open(outputFileName,'w')
   f.write(predictions)
   f.close()


