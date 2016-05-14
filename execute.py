from function_definitions import *
from subprocess import Popen, PIPE
from crftest.extract_concepts_found import *
import numpy as np

# Read the annotated data in an Ordered Dictionary
txtFileName= './data/Annotations-1-120.txt'
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

prefix = "./output/"
# write the data to file
filename = prefix + 'trainNegatedCRF.csv'
writeCsvToFile(trainDataCRF, filename)
print "%s created" % filename


# evaluate crf
crfPath = ""
crfLearn = "crf_learn"
crfTest = "crf_test"
templatePaths = ["./crftest/template", "./crftest/template2"]
modelPath = prefix + "model"

crfC = [5.0, 10.0, 15.0, 20.0]
crfF = [1, 2, 3, 5, 10, 15]

outputFileName = prefix + "output.csv"
outputConceptsFileName = prefix + "concepts.csv"

useMira = [False, True]

confMatrixes = []
# Split data for training and testing
percentTestData= 25 # Only integer

# number of times to split dataset randomly to compute average accuracy
N = 10

trainingFileNameTemplate = prefix + 'trainingNegatedDataCRF-%d.txt'
testFileNameTemplate = prefix + 'testNegatedDataCRF-%d.txt'
# first - split the data
for i in xrange(N):
   [dataCRF, trainingDataCRF, testDataCRF] = splitDataForValidation(filename, percentTestData)
   writeLinesToFile(trainingDataCRF, trainingFileNameTemplate % i)
   writeLinesToFile(testDataCRF, testFileNameTemplate % i)

specificities = []
sensitivities = []
precisions = []
accuracies = []
f1scores = []
params = []


# then - check all sets of parameters
for crfCParam in crfC:
   for crfFParam in crfF:
      for useMiraParam in useMira:
         for templatePath in templatePaths:
            recallForParams = []
            precisionForParams = []
            crfCmd = [crfLearn]
            crfParams = ['-c', str(crfCParam), '-f', str(crfFParam)]
            if (useMiraParam):
               crfParams.extend(["-a", "MIRA"])
            crfCmd.extend(crfParams)
            crfParams.extend(["template", templatePath])
            print crfParams
            confMatForParams = []
            ignored = 0
            for i in xrange(N):
               crfCmd.extend([templatePath, trainingFileNameTemplate % i, modelPath])
               # Run the tagger and get the output
               p = Popen(crfCmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
               (stdout, stderr) = p.communicate()
               if p.returncode != 0:
                  print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))
                  break

               crfCmd = [crfTest, '-m', modelPath, (testFileNameTemplate % i)]
               p = Popen(crfCmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
               (stdout, stderr) = p.communicate()

               predictions = stdout
               f = open(outputFileName,'w')
               f.write(predictions)
               f.close()

               # (nonOTagAccuracy, accuracy, class_res) = count_out(outputFileName)
               (concepts, sentences, labels) = classify_concepts(outputFileName)

               # confusion matrix
               # [TP FN
               #  FP TN]
               confMatrix, ignoredC = getConfusionMatrix(conceptsForUniqueSentences, concepts,
                                               uniqueSentenses, sentences,
                                               negationsForUniqueSentences, labels)
               ignored += ignoredC

               confMatForParams.append(np.array(confMatrix))
            sumConfMatrix = sum(confMatForParams)
            confMatrixes.append(sumConfMatrix)

            # specificity TN/(TN+FP)
            specificity = sumConfMatrix[1,1]/float(sumConfMatrix[1,0]+sumConfMatrix[1,1])
            print "specificity / true negative rate %s" % str(specificity)

            # sensitivity TP/(TP+FN) - recall
            sensitivity = sumConfMatrix[0,0]/float(sumConfMatrix[0,0]+sumConfMatrix[0,1])
            print "sensitivity  / true positive rate / recall %s" % str(sensitivity)

            # precision TP/(TP+FP)
            precision = sumConfMatrix[0,0]/float(sumConfMatrix[0,0]+sumConfMatrix[1,0])
            print "precision %s" % str(precision)

            # F1-score 2*TP/(2*TP+FP+FN)
            f1score = 2*sumConfMatrix[0,0]/float(2*sumConfMatrix[0,0]+sumConfMatrix[1,0]+sumConfMatrix[0,1])
            print "f1 score %s" % str(f1score)

            # accuracy (TP+TN)/(TP+FP+FN)
            accuracy = (sumConfMatrix[0,0]+sumConfMatrix[1,1])/float(sum(sum(sumConfMatrix)))
            print "accuracy %s" % str(accuracy)

            # print "ignored %f" % (ignored/float(sum(sum(sumConfMatrix))))
            # print "ignored %d" % (ignored)

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)
            precisions.append(precision)
            f1scores.append(f1score)
            params.append(crfParams)


maxSpecificity = max(specificities)
print("max specificity / true negative rate %f, for params %s" % (maxSpecificity, ' '.join(params[specificities.index(maxSpecificity)])))
maxSensitivity = max(sensitivities)
print("max sensitivity / true positive rate / recall %f, for params %s" % (maxSensitivity, ' '.join(params[sensitivities.index(maxSensitivity)])))
maxPrecision = max(sensitivities)
print("max precision %f, for params %s" % (maxPrecision, ' '.join(params[precisions.index(maxPrecision)])))
maxF1score = max(f1scores)
print("max f1score %f, for params %s" % (maxF1score, ' '.join(params[f1scores.index(maxF1score)])))
maxAccuracy = max(accuracies)
print("max accuracy %f, for params %s" % (maxAccuracy, ' '.join(params[accuracies.index(maxAccuracy)])))
