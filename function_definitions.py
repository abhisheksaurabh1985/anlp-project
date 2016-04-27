from collections import OrderedDict
import csv

# External libraries
from nltk.tokenize import word_tokenize

def getCSVInDictionary(csvFileName):
    reader= csv.DictReader(open(csvFileName))
    result= OrderedDict()
    for row in reader:
        for column, value in row.iteritems():
            result.setdefault(column, []).append(value)
    return result
    
def getTokens(listSentences):
    tokenizedSentences = []
    for eachSentence in listSentences:
        tokenizedSentences.append(word_tokenize(eachSentence))
    return tokenizedSentences
        
    

