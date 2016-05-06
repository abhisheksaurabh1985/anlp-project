import csv
import re

out_csv = open("test_out.csv")
sentences = []
concepts_found = []
concepts_initial = []
sentence = ""
concept_found = ""
concept_initial = ""
for line in out_csv:

   if not line.strip():
      sentences.append(sentence[1:])
      concepts_found.append(concept_found)
      concepts_initial.append(concept_initial)
      sentence = ""
      concept_initial = ""
      concept_found = ""
   else:
      parts = re.split("\t|\n", line)
      if(parts[0]=="."):
         sep = ""
      else:
         sep = " "
      sentence = sentence + sep + parts[0]
      if (parts[3] != "O"):
         concept_found = concept_found +  parts[0] + " "
      if (parts[2] != "O"):
         concept_initial = concept_initial + parts[0] + " "
sentences
print ("total concepts %d" % len(concepts_initial))
print ("equal concepts %d" % len([i for i in xrange(len(concepts_initial)) if
                                  concepts_initial[i] == concepts_found[i]]))
print ("partial concepts %d" %
       len([i for i in xrange(len(concepts_initial)) if
            (concepts_found[i] in concepts_initial[i] and concepts_found[i])
            or (concepts_initial[i] in concepts_found[i] and concepts_initial[i])
            and (concepts_initial[i] != concepts_found[i])]))
print ("sub concepts %d" % len([i for i in xrange(len(concepts_initial)) if
                                concepts_found[i] in concepts_initial[i] and concepts_found[i]
                                and (concepts_initial[i] != concepts_found[i])]))
print ("outer concepts %d" % len([i for i in xrange(len(concepts_initial)) if
                                  concepts_initial[i] in concepts_found[i] and concepts_initial[i]
                                  and (concepts_initial[i] != concepts_found[i])]))
out_csv.close()

concept_file = "concepts.csv"

f = open(concept_file, "w")
separator = "\t"
f.write("Original concept" + separator + "Found concept" + separator + "Sentence\n")
for i in xrange(len(sentences)):
   f.write(concepts_initial[i] + separator + concepts_found[i] + separator +sentences[i]+ "\n" )



