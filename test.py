from nltk_utils import tokenize

x = [ 'I have an apple', 'bee', 'cat', 'dog' ]
y = [ 'egg', 'fish' ]
list = []
for w in x :
    z = tokenize( w )
    list.extend( z )

print( list )