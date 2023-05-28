import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork

with open( 'intents.json', 'r' ) as file : # read json file in dictionary form
    intents = json.load( file )
# print( type( intents ) )

all_words = []
keys = []
keysValues = []

# indexing for the key ['intents'] in intents.json
for intent in intents[ 'intents' ] : 
    tag = intent[ 'tag' ] # for every 'tag' in intents.json, we store them inside an tags list
    keys.append( tag ) 
    for pattern in intent[ 'patterns' ] : # second for loop to extract all content inside 'patterns' list
        tokenizedSentense = tokenize( pattern )    # for every content, apply tonkenization --> all_word
        all_words.extend( tokenizedSentense ) # tokenize will return a list, that's why we use extend to extract all words
        keysValues.append( ( tag, tokenizedSentense ) )
    
ignore_words = [ '?', '!', '.', ',' ]
all_words = [ stem( word ) for word in all_words if word not in ignore_words ]
all_words = sorted( set( all_words ) ) # use set to get all unique word
keys = sorted( set( keys ) )

y_train = []
X_train = []

for ( key, value ) in keysValues :
    # print( f"key : { key }, value: { value }" )
    bag = bag_of_words( value, allWords = all_words )
    X_train.append( bag )

    label = keys.index( key )
    y_train.append( label ) # CrossEntropyLoss

X_train = np.array( X_train )
y_train = np.array( y_train )

class ChatDataset( Dataset ) :
    def __init__( self ) :
        self.n_samples = len( X_train )
        self.x_data = X_train
        self.y_data = y_train
    
    # dataset[idx]
    def __getitem__( self, index ) :
        return self.x_data[ index ], self.y_data[ index ]
    
    def __len__( self ) :
        return self.n_samples
    
    # Hyperperameters
batch_size = 8
hiddenSize = 8
inputSize = len( X_train[ 0 ] )
numberOfClasses = len( keys )
learningRate = 0.001
numEpochs = 1000
 
dataset = ChatDataset( )
train_loader = DataLoader( dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2 )
device = torch.device( 'cuda' if torch.cuda.is_available( ) else 'cpu' )
model = NeuralNetwork( inputLayer = inputSize, hiddenLayer = hiddenSize , Classes = numberOfClasses ).to( device )

# loss and optimizer
criterion = nn.CrossEntropyLoss( )
optimizer = torch.optim.Adam( model.parameters( ), lr = learningRate )
 
for epoch in range( numEpochs ) :
    for ( words, labels ) in train_loader : 
        words = words.to( device )
        labels = labels.to( device )
        # forward
        outputs = model( words )
        loss = criterion( outputs, labels )
        # backward and optimizer step
        optimizer.zero_grad( )
        loss.backward( )
        optimizer.step( )

    if ( epoch + 1 ) % 100 == 0 :
        print( f"epoch { epoch + 1 } / { numEpochs }, loss = { loss.item( ) }" )

print( f" final loss, loss= epoch { loss.item( ) }")

data = {
    "modelState" : model.state_dict( ),
    "inputSize" : inputSize,
    "hiddenSize" : hiddenSize,
    "numberOfClasses" : numberOfClasses,
    "all_words" : all_words,
    "keys" : keys,
}

FILE = "data.pth"

torch.save( data, FILE )

print( f'training complete. file saved to {FILE}')
