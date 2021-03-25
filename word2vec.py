#!/usr/bin/env python
# coding: utf-8

# In[311]:


import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit

nltk.download('stopwords')
nltk.download('punkt')
import random


# In[312]:


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from


# In[313]:


def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = False
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    # ... load in first 15,000 rows of unlabeled data file.  You can load in
    # more if you want later (and should do this for the final homework)
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    #fullconts = fullconts[1:15000]  # (TASK) Use all the data for the final submission
    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]





    print ("Generating token stream...")
    #... populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.
    fullrec = []
    min_count = 50
    origcounts = Counter()
    word_lst = nltk.word_tokenize(fullconts[0])
    stop_words = set(stopwords.words('english'))
    fullrec = [word for word in word_lst if word not in stop_words]
    origcounts.update(fullrec)





    print ("Performing minimum thresholding..")
    #... populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = []
    for token in fullrec:
        if origcounts[token] >= min_count:
            fullrec_filtered.append(token)
        else:
            fullrec_filtered.append('<UNK>')       
    wordcounts.update(fullrec_filtered)


    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered






    print ("Producing one-hot indicies")
    #... sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = sorted(set(fullrec))
    wordcodes = dict((c, i) for i, c in enumerate(uniqueWords))
    for n in range(len(fullrec)):
        fullrec[n] = wordcodes[fullrec[n]]






    #... close input file handle
    handle.close()



    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.
    #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows
    
    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec


# In[314]:


@jit
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


# In[315]:


fullsequence = loadData('unlabeled-data.txt')


# In[316]:


len(fullsequence)


# In[317]:


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    #global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0


    print ("Generating exponentiated count vectors")
    #... for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = [v**exp_power for v in wordcounts.values()]
    max_exp_count = sum(exp_count_array)



    print ("Generating distribution")

    #... compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = [c/max_exp_count for c in exp_count_array]





    print ("Filling up sampling table")
    #... create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.

    cumulative_dict = {}
    table_size = 1e7
    key = 0
    for i in range(len(prob_dist)):
        size = int(np.floor(prob_dist[i]*table_size))
        for number in range(size):
            cumulative_dict[key] = wordcodes[uniqueWords[i]]
            key+=1



    return cumulative_dict


# In[318]:


samplingTable=negativeSampleTable(fullsequence,uniqueWords,wordcounts)


# In[319]:


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    #... randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    for i in range(num_samples):
        token_idx = random.randint(0,len(samplingTable)-1)
        while samplingTable[token_idx] == context_idx:
            token_idx = random.randint(0,len(samplingTable)-1)
        results.append(samplingTable[token_idx])
        
    return results


# In[320]:


@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, context_words,W1,W2,negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
        #... implement gradient descent. Find the current context token from context_words
        #... and the associated negative samples from negative_indices. Run gradient descent on both
        #... weight matrices W1 and W2.
        #... compute the total negative log-likelihood and store this in nll_new.
        #... You don't have to use all the input list above, feel free to change them
        
    for i in range(len(context_words)):
        #w_j = []
        #w_j +=[(context_idx, 1)]
        #w_neg=[]       
        #for x in range(num_samples):
            #neg_idx = negative_indices[i*num_samples+x]
            #w_neg +=[(neg_idx, 0)]
        #w_j = w_c + w_neg  
        
        h = np.copy(W1[center_token])
        context_idx = context_words[i]
        wj = [(context_idx, 1)] + [(negative_indices[i*num_samples+x], 0) for x in range(num_samples)]       
        W1_sum = np.zeros(hidden_size)
        
        for idx,v in wj:
            vh = np.dot(W2[idx],h)
            W1_sum += (sigmoid(vh)-v)*W2[idx]
            W2[idx] = W2[idx]-learning_rate*(sigmoid(vh)-v)*h   
            
        W1[center_token] = h - learning_rate*W1_sum
        
        update = 0
        
        for k in range(1,len(wj)):
            update += np.log(sigmoid(-np.dot(W2[k],W1[center_token])))
            
        nll_new = -np.log(sigmoid(np.dot(W2[context_idx],W1[center_token])))- update
        
    return [nll_new]


# In[321]:


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0




    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            if iternum%10000==0:
                print ("Negative likelihood: ", nll)				
                nll_results.append(nll)
                nll = 0


            #... determine which token is our current input. Remember that we're looping through mapped_sequence
            center_token = mapped_sequence[i]
            #... don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            if uniqueWords[center_token] == '<UNK>':
                continue
                



            iternum += 1
            cti=[]
            neg_idx = []
            #... now propagate to each of the context outputs
            for k in range(0, len(context_window)):

                #... Use context_window to find one-hot index of the current context token.
                context_index = mapped_sequence[i+context_window[k]]
                cti.append(context_index)
                #... construct some negative samples
                negative_indices = generateSamples(context_index, num_samples)
                neg_idx+=negative_indices
                
            #context_words = [mapped_sequence[x] for x in cti]

                #... You have your context token and your negative samples.
                #... Perform gradient descent on both weight matrices.
                #... Also keep track of the negative log-likelihood in variable nll.
            [nll_new] = performDescent(num_samples, learning_rate, center_token, cti ,W1,W2,neg_idx)
            nll+=nll_new



    for nll_res in nll_results:
        print (nll_res)
    return [W1,W2]



# In[456]:


def load_model():
    handle = open("saved_W1.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]


# In[445]:


def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()


# In[446]:


word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)


# In[325]:


train = train_vectors()


# In[ ]:


def morphology(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [word_seq[0], # suffix averaged
	embeddings[wordcodes[word_seq[1]]]]
	vector_math = vectors[0]+vectors[1]
	#... find whichever vector is closest to vector_math
	#... Use the same approach you used in function prediction() to construct a list
	#... of top 10 most similar words to vector_math. Return this list.


# In[485]:


def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    #... find whichever vector is closest to vector_math
    #... Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.
    top10_analogy = []

    for w in uniqueWords:
        if w not in word_seq:
            vector_other = embeddings[wordcodes[w]] 
            cosine_similarity = 1-cosine(vector_other,vector_math)
            top10_analogy.append({"word": w, "score": cosine_similarity})
        
    top10_analogy.sort(key = lambda x: x['score'], reverse = True)
    top10_analogy = top10_analogy[:10]
    
    return top10_analogy


# In[486]:


analogy(['fresh','fish','delicious'])


# In[487]:


analogy(['coffee','sugar','tea'])


# In[489]:


analogy(['matcha','milk','latte'])


# In[490]:


analogy(['tart','eggs','bread'])


# In[478]:


analogy(['pasta','tomato','noodle'])


# In[458]:


def get_neighbors(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    #... search through all uniqueWords and for each token, compute its similarity to target_word.
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
    top10 = []
    ohidx = wordcodes[target_word]
    vector_t = word_embeddings[ohidx]

    for w in uniqueWords:
        if w != target_word:
            vector_other = word_embeddings[wordcodes[w]] 
            cosine_similarity = 1-cosine(vector_other,vector_t)
            top10.append({"word": w, "score": cosine_similarity})
        
    top10.sort(key = lambda x: x['score'], reverse = True)
    top10 = top10[:10]
    
    return top10
    


# In[459]:


[word_embeddings, proj_embeddings] = load_model()


# In[460]:


get_neighbors('costco')


# In[467]:


get_neighbors('tart')


# In[465]:


with open('prob7_output.txt','w') as output:
    output.write("target word, similar word, simliar score\n")
    target_words = ['fresh','meat','bbq','cream','coffee','sushi','oh','rainbow','unicorn','opaque']
    for tw in target_words:
        results = get_neighbors(tw)
        for i in range(len(results)):
            output.write(tw+","+results[i]["word"]+","+str(results[i]["score"])+"\n")
output.close()


# In[464]:


with open('intrinsic-test.csv') as test:
    file = csv.reader(test,delimiter = ',')
    lines = []
    for line in file:
        lines+=[line]
        
    with open('intrinsic-result.csv','w') as output:
        out = csv.writer(output)
        out.writerow(['ID','sim'])
        
        for k in range(1,len(lines)):
            
            word1 = word_embeddings[wordcodes[lines[k][1]]]
            word2 = word_embeddings[wordcodes[lines[k][2]]]
            cosine_similarity = 1-cosine(word1,word2)
            out.writerow([lines[k][0],cosine_similarity])
            
output.close()

    


# In[496]:


with open('intrinsic-test.csv') as test:
    file = csv.reader(test,delimiter = ',')
    lines = []
    for line in file:
        lines+=[line]
    


# In[497]:


lines


# In[ ]:


if __name__ == '__main__':
    if len(sys.argv)==2:
        filename = sys.argv[1]  
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))



        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        #train_vectors(preload=False)
        [word_embeddings, proj_embeddings] = load_model()








        #... we've got the trained weight matrices. Now we can do some predictions
        #...pick ten words you choose
        targets = ["good", "bad", "food", "apple",'tasteful','unbelievably','uncle','tool','think']
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (prediction(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
            print ("\n")



        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        print (analogy(["apple", "fruit", "banana"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.
        #... this is the optional task, if you don't want to finish it, common lines from 545 to 556

        s_suffix = [word_embeddings[wordcodes["banana"]] - word_embeddings[wordcodes["bananas"]]]
        others = [["apples", "apple"],["values", "value"]]
        for rec in others:
            s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
        s_suffix = np.mean(s_suffix, axis=0)
        print (morphology([s_suffix, "apples"]))
        print (morphology([s_suffix, "pears"]))






    else:
        print ("Please provide a valid input filename")
        sys.exit()

