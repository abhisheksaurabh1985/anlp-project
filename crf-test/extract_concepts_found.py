import csv
import re

out_csv = open("test_out.csv")
sentences = []
concepts_found = []
concepts_initial = []
sentence = ""
concepts_found_current = []
concepts_initial_current = []
initial = ""
found = ""
for line in out_csv:

   if not line.strip():
      sentences.append(sentence[1:])
      concepts_found.append(concepts_found_current)
      concepts_initial.append(concepts_initial_current)
      sentence = ""
      concepts_found_current = []
      concepts_initial_current = []
      initial = ""
      found = ""
   else:
      parts = re.split("\t|\n", line)
      if(parts[0]=="."):
         sep = ""
      else:
         sep = " "
      sentence = sentence + sep + parts[0]


      if (parts[3] == "B"):
         if(found.strip()):
            concepts_found_current.append(found.strip())
         found =  parts[0] + " "
      elif (parts[3] == "I"):
         found = found + parts[0] + " "
      elif (parts[3] == "O"):
         if(found.strip()):
            concepts_found_current.append(found.strip())

      if (parts[2] == "B"):
         if(initial.strip()):
            concepts_initial_current.append(initial.strip())
         initial =  parts[0] + " "
      elif (parts[2] == "I"):
         initial = initial + parts[0] + " "
      elif (parts[2] == "O"):
         if(initial.strip()):
            concepts_initial_current.append(initial.strip())
            initial = ""





sentences
print ("total sentences %d" % len(concepts_initial))

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

print ("total concepts %d" % total)
print ("equal concepts %d; percenatge %f" % (equal, equal/float(total)))
print ("partial concepts %d; percenatge %f" % (partial, partial/float(total)))
print ("sub concepts %d; percenatge %f" % (subconcepts, subconcepts/float(total)))
print ("outer concepts %d; percenatge %f" % outerconcepts, outerconcepts/float(total))
out_csv.close()

concept_file = "concepts.csv"

f = open(concept_file, "w")
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


