import re
inputFilename = "negative.txt"
f = open(inputFilename, 'r')

outputFilename = "full_papers_neg_parsed.txt"
out = open(outputFilename, 'w')
for line in f:
   lineProcessed = re.sub("[^\w]", " ", line)
   out.write(lineProcessed.lower() + "\n")

f.close()
out.close()