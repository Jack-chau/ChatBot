import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize

device = torch.device( 'cuda' if torch.cuda.is_available( ) else 'cpu' )

with open( 'intents.json', 'r' ) as f : 
    intents = json.load( f )

FILE = "data.pth"

data = torch.load ( FILE )
inputSize = data[ "inputSize" ]
hiddenSize = data[ "hiddenSize" ]
numberOfClasses = data[ "numberOfClasses" ]
all_words = data[ "all_words"]
keys = data[ "keys"]
model_state = data[ "modelState"]

model = NeuralNetwork( inputLayer = inputSize, hiddenLayer = hiddenSize , Classes = numberOfClasses ).to( device )
model.load_state_dict( model_state )
model.eval( )

bot_name = "Sam"
print( "Let's chat! type 'quit' to exit" )

while True :
    sentence = input( 'You: ' )
    if sentence == "quit" :
        break
    sentence = tokenize( sentence )
    X = bag_of_words( sentence, all_words )
    X = X.reshape( 1, X.shape[0] )
    X = torch.from_numpy( X )

    output = model( X )

    _, predicted = torch.max( output, dim=1 )
    key = keys[ predicted.item( ) ]
    
    probs = torch.softmax( output, dim =1 )
    prob = probs[ 0 ][ predicted.item( ) ]

    if prob.item( ) > 0.75 :
        for intent in intents[ "intents" ] :
            if key == intent[ 'tag' ] :
                print( f"{ bot_name }: { random.choice( intent[ 'responses' ] ) } " )
    else : 
        print( "I don't understant..." )
