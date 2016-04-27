from collections import OrderedDict
import csv

# External libraries
from nltk.tokenize import word_tokenize

def getCSVInDictionary(csvFileName):
    '''
    Description: Read a multi column CSV file with first row as header. File content is stored in a dictionary which the function returns. 
    Input: File path along with the file name.
    Output: An ordered dictionary with file header as KEY. Dictionary values are a list which hold the column values corresponding to each header.
    For e.g. {'header A': ['val_1_header_A', 'val_2_header_A'], 'header B': ['val_1_header_B', 'val_1_header_B']}   
    '''
    reader= csv.DictReader(open(csvFileName))
    result= OrderedDict()
    for row in reader:
        for column, value in row.iteritems():
            result.setdefault(column, []).append(value)
    return result
    
def getTokens(listSentences):
    '''
    Description: Function to get tokens from sentences. Word2Vec expects single sentences, each one of them as a list of words.
    Input: List of sentences.
    Output: A nested list with the inner list containing a sentence in a tokenized form. All the tokens are in lowercase.
    '''
    tokenizedSentences = []
    for eachSentence in listSentences:
        tokenizedSentences.append(word_tokenize(eachSentence.lower()))
    return tokenizedSentences
        
    

