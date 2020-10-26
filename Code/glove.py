# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:02:52 2019

@author: bipin
"""

import math
import numpy as np
import scipy
import itertools
from random import shuffle


# implementation of Glove 
     
     #First we need to build the co occurence matrix  :
     #WE do this by taking a window of k accross the corpus and see how many
     #times a word occurs in the window 
window_size = 10 
def build_cooccur(songs , corpus,min_count = None  ):
    size_of_matrix = len( songs)
    id2word = dict((i, word) for word, (i, _) in songs.iteritems())

    # Collect cooccurrences internally as a sparse matrix
    cooccurrences = scipy.sparse.lil_matrix((size_of_matrix,size_of_matrix),dtype=np.float64)
#    Take each line in the corpus split it ,and build a sequence of word ids
    for i, line in enumerate(corpus):
        tokens = line.strip().split()
        token_ids = [songs[word][0] for word in tokens]
        #    For each word ( word id ) in the sentence check the context words 
        #    To the left of this word 
        for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)
#                For each word id j we will add on weight to the cell Xij
#                Words which are nearer to each other have higher Xij values
#                Where the X is the co-occurence matrix .
            
#                We also know that the co-coocurence matrix is symmetric . 
#                context word apperaing to the left of the thw word 
#                is same as it apperaing to the right of it
            for left_i, left_id in enumerate(context_ids):
            # Distance from center word
                distance = contexts_len - left_i
    
                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)
    
                # Build co-occurrence matrix symmetrically (pretend
                # we are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment
    
#"""        
#     a cooccurrence pair) is of the form
#       
#    (i_main, i_context, cooccurrence)
#    
#    where `i_main` is the ID of the main word in the cooccurrence and
#    `i_context` is the ID of the context word, and `cooccurrence` is the
#    `X_{ij}` cooccurrence value 
#    
#    """
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,cooccurrences.data)):
        if min_count is not None and songs[id2word[i]][1] < min_count:
            continue
        for data_idx, j in enumerate(row):
            if min_count is not None and songs[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter (songs, data):
    learning_rate=0.05
    x_max=100
    alpha=0.75 
#    """
#    Run a single iteration of GloVe training using the given
#    cooccurrence data and the previously computed weight vectors /
#    biases and accompanying gradient histories.
#    
#    """
    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)
    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - math.log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost
                  
def train_glove(songs, cooccurrences, vector_size=100,iterations=25, **kwargs):
    size_1 = len(songs )

#    We build two word vectors for each word: one for the word as
#    the main (center) word and one for the word as a context word.
#    All elements are initialized randomly in the range (-0.5, 0.5].
    
    W = (np.random.rand(size_1 * 2, vector_size) - 0.5) / float(vector_size + 1)
#    The above is aweight matrix 
#    Now we add the biases 
    biases = (np.random.rand(size_1 * 2) - 0.5) / float(vector_size + 1)
    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((size_1 * 2, vector_size),dtype=np.float64)
    
    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(size_1 * 2, dtype=np.float64)
    
#    Now we begin the iterations : 
    data = [(W[i_main], W[i_context +size_1],
             biases[i_main : i_main + 1],
             biases[i_context + size_1 : i_context + size_1 + 1],
             gradient_squared[i_main], gradient_squared[i_context + size_1],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + size_1
                                     : i_context + size_1 + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]
    for i in range(iterations):
        cost = run_iter(songs, data, **kwargs)
    return W

f=open("5000 word.txt","r") #List of words which forms the vocabulary for the Glove

contents=f.read()
vocabulary = dict()
words=contents.split(',')
for j in range(len(words)):
    words[j]=words[j].strip()  
    vocabulary[words[j]]= j
    
#Assuming we have the corpus 
#Which is the collections lines with words for iterating through the window
# So corpus is just like a file which has been read
cooccurrences = build_cooccur(vocabulary)
# actually has to be cooccurrences = build_cooccur(vocabulary,corpus)
W = train_glove(vocabulary, cooccurrences )