import torch
import torch.nn as nn

class NeuralNetwork( nn.Module ) :
    def __init__( self, inputLayer, hiddenLayer, Classes ) :
        super( NeuralNetwork, self ).__init__(  )
        self.l1 = nn.Linear( inputLayer, hiddenLayer )
        self.l2 = nn.Linear( hiddenLayer, hiddenLayer )
        self.l3 = nn.Linear( hiddenLayer, Classes )
        self.relu = nn.ReLU( )

    def forward( self, x ) :
        out = self.l1( x )
        out = self.relu( out )
        out = self.l2( out )
        out = self.relu( out )
        out = self.l3( out )
        # no actication an no softmax
        return out
    
