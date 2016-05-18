import nltk
import operator

class ChunkParser(nltk.ChunkParserI):
   '''
   Parser which is train using conll corpors
   '''
   def __init__(self, train_sents):
      '''
      getting training data and perform training
      :param train_sents: tagged training data
      :return:
      '''
      train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
      self.tagger = nltk.TrigramTagger(train_data)
   def parse(self, sentence):
      pos_tags = [pos for (word,pos) in sentence]
      tagged_pos_tags = self.tagger.tag(pos_tags)
      chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
      conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
      in zip(sentence, chunktags)]
      return nltk.chunk.conlltags2tree(conlltags)


class KFoldEvaluator:
   '''
   Class to evaluate five performance measures using nested cross-validation
   '''
   def __init__(self):
      self.evalMeasureKeys = ['specificity', 'sensitivity', 'precision', 'accuracy', 'f1score']
      self.bestParamsDict = {}
      for k in self.evalMeasureKeys:
         self.bestParamsDict[k] = []

      self.currentMeasures = {}
      for k in self.evalMeasureKeys:
         self.currentMeasures[k] = []
      self.params = []

   @staticmethod
   def getMax(myList):
      '''
      get max value from the list and and its index
      :param myList:
      :return:
      '''
      index, value = max(enumerate(myList), key=operator.itemgetter(1))
      return (value,index)


   def computeEvalMeasures(self, confusionMatrix):
      '''
      Compute five performance measures from a confusion matrix
      :param confusionMatrix: 2x2 np.array
      :return: dictionary with 5 keys corresponding to specificity, sensitivity, precision, accuracy, f1score
      '''
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
         res[self.evalMeasureKeys[i]] = measures[i]

      return res

   def appendToBest(self, measures, k):
      '''
      append measure to the current dictionary of the best params in the outer cross validation
      :param measures:
      :param k: key for the dictionary
      :return:
      '''
      for k_m in measures.keys():
         print "%s  %s" % (k_m, measures[k_m])
      self.bestParamsDict[k].append(measures[k])

   def appendToCurrent(self, measures, crfParams):
      '''
      Append the measure and corresponding parameters
      to the current set of parameters for the inner cross validation
      :param measures:
      :param crfParams:
      :return:
      '''
      for k in measures.keys():
         print "%s  %s" % (k, measures[k])
         self.currentMeasures[k].append(measures[k])
      self.params.append(crfParams)

   def computeBestFromCurrent(self):
      '''
      compute the best measure value and set of parameters for the outer cross validation
      :return:
      '''
      self.bestParamsDict_inner = {}

      for i in xrange(len(self.evalMeasureKeys)):
         maxMeasure, ind = self.getMax(self.currentMeasures[self.evalMeasureKeys[i]])
         print("\nmax %s %f, for params %s" %
            (self.evalMeasureKeys[i], maxMeasure, ' '.join(self.params[ind])))
         otherMeasures = {}
         for k in self.evalMeasureKeys:
            otherMeasures[k] = self.currentMeasures[k][ind]
         print ("other measures: %s" % str(otherMeasures))
         self.bestParamsDict_inner[self.evalMeasureKeys[i]] = (maxMeasure, self.params[ind])
