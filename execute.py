from negclassifier_functions import *
from negclassifies_classes import *
import os
import numpy as np

if __name__ == "__main__":
   # prefix for all the output files
   prefix = "./output/"
   txtFileName= './data/Annotations-1-120_orig.txt'

   # write the data to file
   filename = prefix + 'trainNegatedCRF.csv'
   # parse the input file and create an input for crf++
   (trainDataCRF, uniqueSentenses, conceptsForUniqueSentences, negationsForUniqueSentences) = \
      createInputForCRF(txtFileName, filename)

   # evaluate crf
   crfPath = ""
   crfLearn = "crf_learn"
   crfTest = "crf_test"
   templateFolder = "./templates"
   modelPath = prefix + "model"

   # Parameters for the grid search
   templatePaths = [os.path.join(templateFolder, f) for f in os.listdir(templateFolder)
                    if os.path.isfile(os.path.join(templateFolder, f))]
   crfC = [1.0, 2.0]
   crfF = [1, 2, 3]
   useMira = [False, True]

   outputFileName = prefix + "output.csv"
   outputConceptsFileName = prefix + "concepts.csv"

   # inner cross-validation to choose hyper-parameters
   k_inner = 10
   # outer cross-validation to evaluate performance
   k_outer = 5

   evaluator = KFoldEvaluator()

   # split data into k_outer folds
   [dataCRF, trainingDataCRF_outer, testDataCRF_outer] = kfoldCrossValidation(filename, k_outer)

   # filenames for outer folds
   filenameForValidationTemplate = prefix + 'validationNegatedDataCRF - %d.txt'
   filenameForFinalTestTemplate = prefix + 'testFinalNegatedDataCRF - %d.txt'

   # filenames for inner folds
   trainingFileNameTemplate = prefix + 'trainingNegatedDataCRF-%d.txt'
   testFileNameTemplate = prefix + 'testNegatedDataCRF-%d.txt'

   # create separate files for each run of simulation
   for i_outer in xrange(k_outer):
      writeLinesToFile(trainingDataCRF_outer[i_outer], filenameForValidationTemplate % i_outer)
      writeLinesToFile(testDataCRF_outer[i_outer], filenameForFinalTestTemplate % i_outer)

   # loop of outer cross-validation
   for i_outer in xrange(k_outer):
      print "\nprocessing outer %d fold\n" % i_outer
      confMatrixes = []

      [dataCRF, trainingDataCRF, testDataCRF] = kfoldCrossValidation(filenameForValidationTemplate%i_outer, k_inner)
      for i in xrange(k_inner):
         writeLinesToFile(trainingDataCRF[i], trainingFileNameTemplate % i)
         writeLinesToFile(testDataCRF[i], testFileNameTemplate % i)

      params = []

      # check all sets of parameters
      for crfCParam in crfC:
         for crfFParam in crfF:
            for useMiraParam in useMira:
               for templatePath in templatePaths:

                  # crf params for training command
                  crfCmd = [crfLearn]
                  crfParams = ['-c', str(crfCParam), '-f', str(crfFParam)]
                  if (useMiraParam):
                     crfParams.extend(["-a", "MIRA"])
                  crfCmd.extend(crfParams)

                  confMatForParams = []
                  ignored = 0
                  for i in xrange(k_inner):
                     # training for ith inner fold
                     (returncode, stderr, stdout) = trainCRF(crfCmd, templatePath, trainingFileNameTemplate%i, modelPath)
                     if returncode != 0:
                        print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))
                        break

                     # testing for the ith fold
                     testCRF([crfTest], modelPath, (testFileNameTemplate % i), outputFileName)

                     (concepts, sentences, labels) = classifyConcepts(outputFileName)

                     # confusion matrix
                     # [TP FN
                     #  FP TN]
                     confMatrix, ignoredC = getConfusionMatrix(conceptsForUniqueSentences, concepts,
                                                     uniqueSentenses, sentences,
                                                     negationsForUniqueSentences, labels)
                     ignored += ignoredC

                     confMatForParams.append(np.array(confMatrix))

                  crfParams.extend(["template", templatePath])
                  print crfParams

                  # summing all the confusion matricies from the inner cross validation
                  sumConfMatrix = sum(confMatForParams)
                  confMatrixes.append(sumConfMatrix)

                  # adding the current averaged parameters to the set of inner cross validation parameters
                  currentMeasures = evaluator.computeEvalMeasures(sumConfMatrix)
                  evaluator.appendToCurrent(currentMeasures, crfParams)

      evaluator.computeBestFromCurrent()

      # test with the best parameters
      for k in evaluator.bestParamsDict_inner.keys():

         print "\ntesting for best parameters for %s" % k
         paramsBest = evaluator.bestParamsDict_inner[k][1]

         print "parameters %s" % ' '.join(paramsBest)
         crfCmd = paramsBest[:-2]
         (returncode, stderr, stdout) = trainCRF([crfLearn] + crfCmd,
                                                 paramsBest[-1],
                                                 filenameForValidationTemplate % i_outer,
                                                 modelPath)
         if returncode != 0:
            print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))

         testCRF([crfTest], modelPath, filenameForFinalTestTemplate % i_outer, outputFileName)
         (concepts, sentences, labels) = classifyConcepts(outputFileName)

         confMatrix, ignoredC = getConfusionMatrix(conceptsForUniqueSentences, concepts,
                                                     uniqueSentenses, sentences,
                                                     negationsForUniqueSentences, labels)
         # adding the computed parameters to set of outer best parameters
         measures = evaluator.computeEvalMeasures(np.array(confMatrix))
         evaluator.appendToBest(measures, k)


   print evaluator.bestParamsDict
   for k in evaluator.evalMeasureKeys:
      print "avg %s = %f" % (k, sum(evaluator.bestParamsDict[k])/float(k_outer))


