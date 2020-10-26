import csv #To handle comma separated values
import pickle #To serialize the inverted index



if __name__=="__main__":
    
    f=open("5000 word.txt","r") #List of words
    songs=dict() #Dictionary with key as song id and value as a list of tuples containing (wordID,frequency) pairs
#    print( songs)
    inverted_index=dict() #The final inverted index
    contents=f.read()
    words=contents.split(',')
    for j in range(len(words)):
        words[j]=words[j].strip()  
        inverted_index[words[j]]=list() #Initialize Inverted Index's values as empty lists
    with open('test.csv','rt')as f: #Open Dataset
        data=csv.reader(f)
        for row in data:
            songs[row[0]]=row[2:] #Make songs a dictionary with key as song id and value as a list of tuples containing (wordID,frequency) pairs
    for k in songs.keys():
        for val in songs[k]:
            word, freq = map(int,val.split(':'))
            tup=(k,freq)
            inverted_index[words[word-1]].append(tup) #Add (SongID,frequency) pairs to the Inverted Index
    print( songs)
    f=open("inver_ind.pkl","wb") #Serialize the inverted index so as to prevent reprocessing the dataset
    pickle.dump(inverted_index,f)
    f.close()