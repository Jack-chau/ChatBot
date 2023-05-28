import nltk
import numpy as np
# nltk.download( 'punkt' ) ## Download pretrained tokenizer
from nltk.stem.porter import PorterStemmer ## For stemming

stemmer = PorterStemmer( )

def tokenize( sentence ) :
    return nltk.word_tokenize( sentence )

def stem( word ) :
    return stemmer.stem( word.lower( ) )

def bag_of_words( tonkenizedSentecnce, allWords ) :
    tonkenizedSentecnce = [ stem( word ) for word in tonkenizedSentecnce ]
    bag = np.zeros( len( allWords ), dtype = np.float32 )
    for idx, word in enumerate( allWords ) :
        if word in tonkenizedSentecnce :
            bag[ idx ] = 1.0
    return bag

# sentence = [ "hello", "how", "are", "you" ]
# words = [ "hi", "hello", "I", "you", "bye", "thank", "cool" ]
# bag = bag_of_words( sentence, words )
# print( bag )