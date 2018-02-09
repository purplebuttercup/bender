
import nltk
from PyDictionary import PyDictionary
from nltk.corpus import wordnet as wn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from urllib.request import urlopen
from urllib.request import URLError

sentence = "Check whether run-flat tyres have already been fitted"

#break sentence in words
tokens = nltk.word_tokenize(sentence)
print(tokens)

#assign part-of-speach tags
tagged = nltk.pos_tag(tokens)

#generate dictionary
dictionary = PyDictionary()

for t in tokens:
    print(t)
    syns = dictionary.synonym(t) #get synonyms
    print(syns)
    #r = "http://words.bighugelabs.com/api/2/79918f9feb35ec3fb2b72016cfb85aba/"+t[0]+"/json "

    #try:
    #    response = urlopen(r)
    #    syn_full = response.read()
    #    print(syn_full)


    #except URLError as e:
    #    print('error ', e)