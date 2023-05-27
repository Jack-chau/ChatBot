import nltk
# nltk.download( 'punkt' ) ## Download pretrained tokenizer
from nltk.stem.porter import PorterStemmer ## For stemming

stemmer = PorterStemmer( )

def tokenize( sentence ) :
    return nltk.word_tokenize( sentence )

def stem( word ) :
    return stemmer.stem( word.lower( ) )

def bag_of_words( tonkenized_sentecnce, all_words ) :
    pass

a = "How long does shipping take?"
print ( a ) 

print( tokenize( a ) )

words = [ "Organize", "organizes", "organizing" ]
stemmed_words = [ stem( word ) for word in words ]
print( stemmed_words )



