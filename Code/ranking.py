import math
import pickle
import numpy as np
import scipy
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

ps = PorterStemmer()        

if __name__=="__main__":
    
    N=27143
    f=open("inver_ind.pkl","rb") #Open the inverted index created in build_index.py using pickle
    inverted_index=pickle.load(f)
    query1=input("Enter the song lyrics: ").lower().split() #Enter the search query
    query =[]
    for w in query1:
        query.append(ps.stem(w))
    print( query )  # Verifying whether it was correctly stemmed or not
    rankin = dict()
    for i in query:
        if (i in inverted_index): # if the term exists in the inverted index
            for j in inverted_index[i]: 
#                iterate through the list of (songID,frequency) pairs and update the songID's score
#                by adding (1 + (log10 of the frequency of the current term))
                if( j[0] not in rankin.keys() ):
                    rankin[j[0]] = (1+ math.log(j[1],10))* math.log((N / len(inverted_index[i])),10)
                else :
                    updaterank = {j[0]:rankin[j[0]]+(1+ math.log(j[1],10))* math.log((N / len(inverted_index[i])),10)}
                    rankin.update(updaterank)
                
    if( len(rankin ) == 0 ):
        print( " No Matches Found")
    else :
        k=10
        for w in sorted(rankin, key=rankin.get, reverse=True):
            print (w)
            k-=1
            if(k<=0):
                 break 
  

#    final_rank=list() #The final ranking list containing songIDs in descending order of score as calculated in ranking dictionary
#    for key, value in sorted(ranking.items(), key=lambda kv: kv[1], reverse=True): #Sort ranking dictionary by value in descending order
#        final_rank.append(key)
#    for i in range(min(len(final_rank),12)): #Print the top 10 songIDs according to score 
#        print(final_rank[i])
#    if not final_rank: #If terms dont match
#        print("No match found")
                 
    
                 
     
     
     
                
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        