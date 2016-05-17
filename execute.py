from function_definitions import *
from subprocess import Popen, PIPE
from crftest.extract_concepts_found import *
import os
import numpy as np
import operator

def getMax(my_list):
   index, value = max(enumerate(my_list), key=operator.itemgetter(1))
   return (value,index)

def createInputForCRF(filename):

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
   writeCsvToFile(trainDataCRF, filename)
   print "%s created" % filename
   return (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences)

def trainCRF(crfCmd, templatePath, trainingFileName, modelPath):
   otherParams = [templatePath, trainingFileName, modelPath]
   # Run the tagger and get the output
   p = Popen(crfCmd + otherParams, stdin=PIPE, stdout=PIPE, stderr=PIPE)
   (stdout, stderr) = p.communicate()
   return (p.returncode, stderr, stdout)

def testCRF(crfCmd, modelPath, testFileName, outputFileName):
   otherParams = ['-m', modelPath, testFileName]
   p = Popen(crfCmd + otherParams, stdin=PIPE, stdout=PIPE, stderr=PIPE)
   (stdout, stderr) = p.communicate()

   predictions = stdout
   f = open(outputFileName,'w')
   f.write(predictions)
   f.close()

evalMeasureKeys = ['specificity', 'sensitivity', 'precision', 'accuracy', 'f1score']

def computeEvalMeasures(confusionMatrix):
    # specificity TN/(TN+FP)
   specificity = confusionMatrix[1,1]/float(confusionMatrix[1,0]+confusionMatrix[1,1])

   # sensitivity TP/(TP+FN) - recall
   sensitivity = confusionMatrix[0,0]/float(confusionMatrix[0,0]+confusionMatrix[0,1])

   # precision TP/(TP+FP)
   precision = confusionMatrix[0,0]/float(confusionMatrix[0,0]+confusionMatrix[1,0])

   # F1-score 2*TP/(2*TP+FP+FN)
   f1score = 2*confusionMatrix[0,0]/float(2*confusionMatrix[0,0]+confusionMatrix[1,0]+confusionMatrix[0,1])

   # accuracy (TP+TN)/(TP+FP+FN)
   accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/float(sum(sum(confusionMatrix)))

   measures = [specificity, sensitivity, precision, accuracy, f1score]
   res = {}
   for i in xrange(len(measures)):
      res[evalMeasureKeys[i]] = measures[i]
   return res



if __name__ == "__main__":
   prefix = "./output/"
   # write the data to file
   filename = prefix + 'trainNegatedCRF.csv'
   (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences) = \
      createInputForCRF(filename)

   # evaluate crf
   crfPath = ""
   crfLearn = "crf_learn"
   crfTest = "crf_test"
   templateFolder = "./templates"
   templatePaths = [os.path.join(templateFolder, f) for f in os.listdir(templateFolder) if os.path.isfile(os.path.join(templateFolder, f))]
   # templatePaths = [templateFolder+"/template8"]
   modelPath = prefix + "model"

   crfC = [1.0, 2.0]
   crfF = [1, 2, 3]

   outputFileName = prefix + "output.csv"
   outputConceptsFileName = prefix + "concepts.csv"

   useMira = [True]

   # inner cross-validation to choose hyper-parameters
   k_inner = 10
   # outer cross-validation to evaluate performance
   k_outer = 5

   bestParamsDict = {}
   for k in evalMeasureKeys:
      bestParamsDict[k] = []

   [dataCRF, trainingDataCRF_outer, testDataCRF_outer] = kfoldCrossValidation(filename, k_outer)

   filenameForValidationTemplate = prefix + 'validationNegatedDataCRF - %d.txt'
   filenameForFinalTestTemplate = prefix + 'testFinalNegatedDataCRF - %d.txt'
   trainingFileNameTemplate = prefix + 'trainingNegatedDataCRF-%d.txt'
   testFileNameTemplate = prefix + 'testNegatedDataCRF-%d.txt'

   for i_outer in xrange(k_outer):
      writeLinesToFile(trainingDataCRF_outer[i_outer], filenameForValidationTemplate % i_outer)
      writeLinesToFile(testDataCRF_outer[i_outer], filenameForFinalTestTemplate % i_outer)


   for i_outer in xrange(k_outer):

      print "\nprocessing outer %d fold\n" % i_outer
      confMatrixes = []

      [dataCRF, trainingDataCRF, testDataCRF] = kfoldCrossValidation(filenameForValidationTemplate%i_outer, k_inner)
      for i in xrange(k_inner):
         writeLinesToFile(trainingDataCRF[i], trainingFileNameTemplate % i)
         writeLinesToFile(testDataCRF[i], testFileNameTemplate % i)

      evalMeasures = {}
      for k in evalMeasureKeys:
         evalMeasures[k] = []

      params = []

      # check all sets of parameters
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
                  for i in xrange(k_inner):
                     (returncode, stderr, stdout) = trainCRF(crfCmd, templatePath, trainingFileNameTemplate%i, modelPath)
                     if returncode != 0:
                        print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))
                        break
                     testCRF([crfTest], modelPath, (testFileNameTemplate % i), outputFileName)

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

                  currentMeasures = computeEvalMeasures(sumConfMatrix)

                  for k in currentMeasures.keys():
                     print "%s  %s" % (k, currentMeasures[k])
                     evalMeasures[k].append(currentMeasures[k])

                  params.append(crfParams)

      bestParamsDict_inner = {}

      for i in xrange(len(evalMeasureKeys)):
         maxMeasure, ind = getMax(evalMeasures[evalMeasureKeys[i]])
         print("\nmax %s %f, for params %s" %
            (evalMeasureKeys[i], maxMeasure, ' '.join(params[ind])))
         otherMeasures = {}
         for k in evalMeasureKeys:
            otherMeasures[k] = evalMeasures[k][ind]
         print ("other measures: %s" % str(otherMeasures))
         bestParamsDict_inner[evalMeasureKeys[i]] = (maxMeasure, params[ind])


      # test with the best parameters
      for k in bestParamsDict_inner.keys():
         print "\ntesting for best parameters for %s" % k
         paramsBest = bestParamsDict_inner[k][1]
         print "parameters %s" % ' '.join(paramsBest)
         crfCmd = paramsBest[:-2]
         (returncode, stderr, stdout) = trainCRF([crfLearn] + crfCmd, paramsBest[-1], filenameForValidationTemplate % i_outer, modelPath)
         if returncode != 0:
            print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))

         testCRF([crfTest], modelPath, filenameForFinalTestTemplate % i_outer, outputFileName)
         (concepts, sentences, labels) = classify_concepts(outputFileName)

         confMatrix, ignoredC = getConfusionMatrix(conceptsForUniqueSentences, concepts,
                                                     uniqueSentenses, sentences,
                                                     negationsForUniqueSentences, labels)
         measures = computeEvalMeasures(np.array(confMatrix))
         for k_m in measures.keys():
            print "%s  %s" % (k_m, measures[k_m])
         bestParamsDict[k].append(measures[k])


   print bestParamsDict
   for k in evalMeasureKeys:
      print "avg %s = %f" % (k, sum(bestParamsDict[k])/float(k_outer))


