from function_definitions import *
from nltk.tokenize import word_tokenize

# Read the annotated data in an Ordered Dictionary
csvFileName= './negex.python/Annotations-1-120.csv'
dictAnnotatedData= getCSVInDictionary(csvFileName)

# Word2Vec expects single sentences, each one of them as a list of words. Generate tokens from sentences. 
tokenizedSentences= getTokens(dictAnnotatedData['Sentence'][0:])
print len(tokenizedSentences)

# Import the built-in logging module and configure it so that Word2Vec creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 2    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 2           # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(tokenizedSentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_2minwords_2context"
model.save(model_name)

