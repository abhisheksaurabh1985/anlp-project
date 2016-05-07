import re

def extract_concepts(tags, filename):
   file = open(filename)
   concepts_found = []
   concepts_initial = []
   concepts_found_current = []
   concepts_initial_current = []
   initial = ""
   found = ""
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

         if (parts[3] == tags[1]):
            if(found.strip()):
               concepts_found_current.append(found.strip())
            found =  parts[0] + " "
         elif (parts[3] == tags[2]):
            found = found + parts[0] + " "
         elif (parts[3] == tags[0]):
            if(found.strip()):
               concepts_found_current.append(found.strip())

         if (parts[2] == tags[1]):
            if(initial.strip()):
               concepts_initial_current.append(initial.strip())
            initial =  parts[0] + " "
         elif (parts[2] == tags[2]):
            initial = initial + parts[0] + " "
         elif (parts[2] == tags[0]):
            if(initial.strip()):
               concepts_initial_current.append(initial.strip())
               initial = ""
   file.close()
   return (concepts_found, concepts_initial)

def count_found_concepts(sentences, concepts_initial, concepts_found, prefix):
   partial = 0
   equal = 0
   subconcepts = 0
   outerconcepts = 0
   total = 0
   for i in xrange(len(sentences)):
      for j in xrange(len(concepts_initial[i])):
         total += 1
         found = ""
         if(len(concepts_found[i])>j):
            found = concepts_found[i][j]
         initial = concepts_initial[i][j]
         if(initial and initial == found):
            equal += 1
         if(initial and initial in found and initial != found):
            outerconcepts +=1
            partial += 1
         if(found and found in initial and found != initial):
            subconcepts += 1
            partial += 1

   print ("total %s concepts %d" % (prefix, total))
   print ("equal %s concepts %d; percenatge %f" % (prefix, equal, equal/float(total)))
   print ("partial %s concepts %d; percenatge %f" % (prefix, partial, partial/float(total)))
   print ("sub %s concepts %d; percenatge %f" % (prefix, subconcepts, subconcepts/float(total)))
   print ("outer %s concepts %d; percenatge %f\n" % (prefix, outerconcepts, outerconcepts/float(total)))

def print_concepts(filename, sentences, concepts_initial, concepts_found):
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


filename = "test_out.csv"
sentences = []
(concepts_found_a, concepts_initial_a) = extract_concepts(['O','BA', 'IA'], filename)
(concepts_found_n, concepts_initial_n) = extract_concepts(['O','BN', 'IN'], filename)
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

print ("total sentences %d\n" % len(concepts_initial_a))

count_found_concepts(sentences, concepts_initial_a, concepts_found_a, "affirmed")
concept_file = "concepts_a.csv"
print_concepts(concept_file, sentences, concepts_initial_n, concepts_found_a)


count_found_concepts(sentences, concepts_initial_n, concepts_found_n, "negated")
concept_file = "concepts_n.csv"
print_concepts(concept_file, sentences, concepts_initial_n, concepts_found_n)


