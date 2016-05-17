from function_definitions import *
from subprocess import Popen, PIPE
from crftest.count_out import *
from crftest.extract_concepts_found import *

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
trainDataCRF = getTrainingDataForCRF(tokenizedSentences,
                                     tokenizedConcepts, negations,
                                     bioTags, priorities, ["Negated", "Affirmed"], cueTypeTags)

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
crfF = [1, 2, 3,  5, 10, 15]

outputFileName = prefix + "output.csv"
outputConceptsFileName = prefix + "concepts.csv"

useMira = [False, True]

recalls = []
precisions = []
fscores = []
parameters = []

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
               open(outputFileName,'w').write(predictions)

               (nonOTagAccuracy, accuracy, class_ref) = count_out(outputFileName)
               (counts, percentage) = do_extract_concepts(outputFileName, outputConceptsFileName)
               recallForParams.append(percentage['recall'])
               precisionForParams.append(percentage['precision'])
            avgRecall = sum(recallForParams)/float(N) #average recall for N runs
            avgPrecision = sum(precisionForParams)/float(N)
            recalls.append(avgRecall)
            precisions.append(avgPrecision)

            fscore = 2*(avgPrecision*avgRecall)/(avgPrecision+avgRecall)
            fscores.append(fscore)
            parameters.append(crfParams)
            print "recall %f" % avgRecall
            print "precision %f" % avgPrecision
            print "fscore %f" % fscore

bestRecall = max(recalls)
print "best recall: %f" % bestRecall
ind = recalls.index(bestRecall)
print "with precision: %f, with fscore: %f, for parameters %s\n" % (precisions[ind], fscores[ind], ' '.join(parameters[ind]))

bestPrecision = max(precisions)
print "best precision: %f" % bestPrecision
ind = precisions.index(bestPrecision)
print "with recall: %f, with fscore: %f, for parameters: %s\n" % (recalls[ind], fscores[ind], ' '.join(parameters[ind]))

bestFScore = max(fscores)
print "best fscore %f" % bestFScore
ind = fscores.index(bestFScore)
print "with precision: %f, recall: %f, for parameters: %s" % (recalls[ind], precisions[ind],
                                                              ' '.join(parameters[ind]))