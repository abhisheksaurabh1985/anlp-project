import re
import sys


def extract_concepts(tags, filename):
   file = open(filename)
   concepts_found = []
   concepts_initial = []
   concepts_found_current = []
   concepts_initial_current = []
   initial = ""
   found = ""

   real_tag_n = 7
   predicted_tag_n = real_tag_n+1

   for line in file:

      if not line.strip():
         concepts_found.append(concepts_found_current)
         concepts_initial.append(concepts_initial_current)
         concepts_found_current = []
         concepts_initial_current = []
         initial = ""
         found = ""
      else:
         parts = re.split("\t|\n", line)

         if (parts[predicted_tag_n] == tags[1]):
            if(found.strip()):
               concepts_found_current.append(found.strip())
            found =  parts[0] + " "
         elif (parts[predicted_tag_n] == tags[2]):
            found = found + parts[0] + " "
         elif (parts[predicted_tag_n] == tags[0]):
            if(found.strip()):
               concepts_found_current.append(found.strip())

         if (parts[real_tag_n] == tags[1]):
            if(initial.strip()):
               concepts_initial_current.append(initial.strip())
            initial =  parts[0] + " "
         elif (parts[real_tag_n] == tags[2]):
            initial = initial + parts[0] + " "
         elif (parts[real_tag_n] == tags[0]):
            if(initial.strip()):
               concepts_initial_current.append(initial.strip())
               initial = ""
   file.close()
   return (concepts_found, concepts_initial)

def classify_concepts(filename):
   concepts = []
   sentences = []
   labels = []

   N = 9

   sentence = []
   concept = []
   labelTags = []
   currentConcepts = []
   currentLabels = []

   conceptTags = ['O','B','I']
   classTags = ['-','N','A']

   f = open(filename)

   for line in f:
      if not line.split():
         sentences.append(sentence)
         sentence = []
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
   if "N" in labelTags:
      return "Negated"
   return "Affirmed"


def count_found_concepts(sentences, concepts_initial, concepts_found, prefix, verbose = False):
   found_from_initial = 0
   initial_from_found = 0
   total_initial = 0
   total_found = 0
   for i in xrange(len(sentences)):
      for j in xrange(len(concepts_initial[i])):
         total_initial += 1
         if concepts_initial[i][j] in concepts_found[i]:
            found_from_initial += 1
      for j in xrange(len(concepts_found[i])):
         total_found +=1
         if concepts_found[i][j] in concepts_initial[i]:
            initial_from_found += 1

   counts = {'total_initial': total_initial, 'total_found': total_found,
             'found_from_initial': found_from_initial, 'initial_from_found': initial_from_found}
   percentages = {'recall': found_from_initial/float(total_initial),
                  'precision': initial_from_found/float(total_found)}
   if verbose:
      print ("total initial %s concepts %d" % (prefix, total_initial))
      print ("recall: found concepts from initial %s concepts %d; percenatge %f" %
             (prefix, found_from_initial, percentages['found_from_initial']))
      print ("total found %s concepts %d" % (prefix, total_found))
      print ("precision: initial concepts from found %s concepts %d; percenatge %f" %
             (prefix, initial_from_found, percentages['initial_from_found']))

   return (counts, percentages)

def get_concepts(filename, sentences, concepts_initial, concepts_found):
   f = open(filename, "w")
   separator = "\t"
   f.write("Original concept" + separator + "Found concept" + separator + "Sentence\n")
   for i in xrange(len(sentences)):
      for k in xrange(len(concepts_initial[i])):
         if (len(concepts_found[i]) == 0):
            found = ''
         elif (k >= len(concepts_found[i])):
            found = concepts_found[i][-1]
         else:
            found = concepts_found[i][k]
         f.write(concepts_initial[i][k] + separator + found + separator +sentences[i]+ "\n" )
   f.close()


def do_extract_concepts(filename, outputFileName, verbose = False):
   sentences = []
   (concepts_found, concepts_initial) = extract_concepts(['-','N','A'], filename)
   file = open(filename)

   sentence = ""
   for line in file:
      if not line.strip():
         sentences.append(sentence[1:])
         sentence = ""
      else:
         parts = re.split("\t|\n", line)
         if(parts[0]=="."):
            sep = ""
         else:
            sep = " "
         sentence = sentence + sep + parts[0]
   file.close()

   if verbose: print ("total sentences %d\n" % len(concepts_initial))

   (counts, percentages) = count_found_concepts(sentences, concepts_initial, concepts_found, "", verbose)
   get_concepts(outputFileName, sentences, concepts_initial, concepts_found)
   return (counts, percentages)


if __name__ == "__main__":
   if(len(sys.argv) <3 ):
      print "input file is not specified"

   filename = sys.argv[1]
   do_extract_concepts(filename, sys.argv[2], True)