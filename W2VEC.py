#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras, os, pickle, re, sklearn, string, tensorflow
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Embedding
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split


print('Keras version: \t\t%s' % keras.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)


# In[2]:


# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download('punkt')   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence,               remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')# Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import re
from bs4 import BeautifulSoup


# In[4]:


# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
def review_to_wordlist( review, remove_stopwords=False ):
# Function to convert a document to a sequence of words,
# optionally removing stop words. Returns a list of words.
#
# 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
#
# 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
#
# 3. Convert words to lower case and split them
    words = review_text.lower().split()
#
# 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
#
# 5. Return a list of words
    return(words)


# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download('punkt')
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
# Function to split a review into parsed sentences. Returns a
# list of sentences, where each sentence is a list of words
#
# 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
#
# 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
# If a sentence is empty, skip it
        if len(raw_sentence) > 0:
# Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
#
# Return the list of sentences (each sentence is a list of words,
# so this returns a list of lists
    return sentences



# In[5]:


def read_files(path):
    documents = list()
# Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename),encoding='utf-8') as f:
                doc = f.read()
                # doc = clean_text(doc)
                documents.append(doc)
# Read in all lines in a txt file
    if os.path.isfile(path):
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(line)
    return documents



pos_train = read_files('/home/workspace/aclImdb/train/pos')
neg_train = read_files('/home/workspace/aclImdb/train/neg')
pos_test = read_files('/home/workspace/aclImdb/test/pos')
neg_test = read_files('/home/workspace/aclImdb/test/neg')
train1 = pos_train + neg_train
test1 = pos_test + neg_test

#docs = negative_docs + positive_docs
l_train = [1 for _ in range(len(pos_train))] + [0 for _ in range(len(neg_train))]
l_test = [1 for _ in range(len(pos_train))] + [0 for _ in range(len(neg_test))]
train = np.column_stack((train1,l_train))
test = np.column_stack((test1,l_test))


# In[6]:


sentences = [] # Initialize an empty list of sentences
num_reviews = 25000
print ("Parsing sentences from training set")
for review in range( 0, num_reviews ):
    sentences += review_to_sentences(train[review][0], tokenizer)
'''
print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
'''


# In[47]:


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

def myhashfxn(obj):
    return hash(obj) % (2 ** 32)

# Set values for various parameters
num_features = 400    # Word vector dimensionality                      
min_word_count = 60   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


#**********************************************************



# Initialize and train the model (this will take some time)
from gensim.models import word2vec
#model = word2vec.Word2Vec(hashfxn=myhashfxn)

print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling,hashfxn=myhashfxn)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[48]:


model.doesnt_match("man woman child kitchen".split())


# In[49]:


model.most_similar("man")


# In[50]:


#Vector Averaging
import numpy as np # Make sure that numpy is imported
def makeFeatureVec(words, model, num_features):
# Function to average all of the word vectors in a given
# paragraph
#
# Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
#
    nwords = 0.
#
# Index2word is a list that contains the names of the words in
# the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
#
# Loop over each word in the review and, if it is in the model's
# vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
#
# Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
# Given a set of reviews (each one a list of words), calculate
# the average feature vector for each one and return a 2D numpy array
#
# Initialize a counter
    counter = 0
#
# Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
# Loop through the reviews
    for review in reviews:
        if counter%1000 == 0:
            print ("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
#
# Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


# In[51]:


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
clean_train_reviews = []
for review in train1:
    clean_train_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))
    
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test1:
    clean_test_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))


# In[52]:


#clustering
from sklearn.cluster import KMeans
import time
start = time.time() # Start time
# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )#**************************************************
idx = kmeans_clustering.fit_predict( word_vectors )
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")


# In[53]:


# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))


# In[54]:


val= list(word_centroid_map.values())
key = list(word_centroid_map.keys())
# For the first 10 clusters
for cluster in range(0,10):
#
# Print the cluster number
    print ("\nCluster %d" % cluster)
#
# Find all of the words for that cluster number, and print them out
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( val[i] == cluster ):
            words.append(key[i])
    print(words)


# In[55]:


def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# In[56]:


train_centroids = np.zeros( (len(train1), num_clusters), dtype="float32" )
# Transform the training set reviews into bags of centroids
counter = 0

for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1
# Repeat for test reviews
test_centroids = np.zeros(( len(test1), num_clusters), dtype="float32" )
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1


# In[57]:


l_test1 = np.asarray(l_test)

from sklearn.ensemble import RandomForestClassifier
#Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators = 100)#******************************
# Fitting the forest may take a few minutes
print ("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,l_train)
result = forest.predict(test_centroids)
# Write the test results

print ('Test Accuracy: %.2f'%forest.score(test_centroids, l_test1))


# In[58]:


from sklearn import linear_model, datasets

LGR = linear_model.LogisticRegression()

LGR.fit(train_centroids, l_train)
result1 = LGR.predict(test_centroids)

print ('Test Accuracy: %.2f'%LGR.score(test_centroids, l_test1))


# In[59]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(random forrest  Classifier)')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[60]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result1)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(Logistic Regression Classifier)')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[21]:


'''
from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [50,100,200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,300,None],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_centroids, l_train)

CV_rfc.best_params_
'''


# In[ ]:





# In[ ]:





# In[22]:


'''
# Grid search cross validation
from sklearn.linear_model import LogisticRegression

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(train_centroids,l_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
'''


# In[20]:


from sklearn.model_selection import validation_curve
l_test1 = np.asarray(l_test)

from sklearn.ensemble import RandomForestClassifier
# Create range of values for parameter
param_range = [150,500,600,700,800,900,1500]

# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(RandomForestClassifier(max_depth =20), 
                                             train_centroids, 
                                             l_train, 
                                             param_name="n_estimators", 
                                             param_range=param_range,
                                             cv=3, 
                                             scoring="accuracy")


# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()


# In[26]:


# Create range of values for parameter
param_range =[50,80,90,100,120,140]

# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(RandomForestClassifier(n_estimators =20), 
                                             train_centroids, 
                                             l_train,
                                             param_name="max_depth", 
                                             param_range=param_range,
                                             cv=3, 
                                             scoring="accuracy")


# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()


# In[27]:


# Create range of values for parameter
param_range =[1,2,3,4,5,10]

# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(linear_model.LogisticRegression(), 
                                             train_centroids, 
                                             l_train,
                                             param_name="C", 
                                             param_range=param_range,
                                             cv=3, 
                                             scoring="accuracy")
# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With  Logestic Regression")
plt.xlabel("Number Of parameter C")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()


# In[ ]:





# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, datasets

forest = RandomForestClassifier(n_estimators = 700,max_depth=80)#******************************
forest = forest.fit(train_centroids,l_train)




from sklearn import linear_model, datasets

LGR = linear_model.LogisticRegression(penalty='l1', C=1.0)

LGR2 = linear_model.LogisticRegression(penalty='l2', C=1.0)#*************************************************

# we create an instance of Neighbours Classifier and fit the data.
LGR.fit(train_centroids, l_train)
LGR2.fit(train_centroids, l_train)


# Use the random forest to make sentiment label predictions
result = forest.predict(test_centroids)
result1 = LGR.predict(test_centroids)
result2 = LGR2.predict(test_centroids)


print ('Test Accuracy: %.2f'%forest.score(test_centroids, l_test1))
print ('Test Accuracy: %.2f'%LGR.score(test_centroids, l_test1))
print ('Test Accuracy: %.2f'%LGR2.score(test_centroids, l_test1))


# In[34]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(random forrest  Classifier)')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[31]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result1)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(Logistic Regression Classifier(L1))')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[32]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result2)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(Logistic Regression Classifier(L2))')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[61]:


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

def myhashfxn(obj):
    return hash(obj) % (2 ** 32)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


#**********************************************************



# Initialize and train the model (this will take some time)
from gensim.models import word2vec
#model = word2vec.Word2Vec(hashfxn=myhashfxn)

print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling,hashfxn=myhashfxn)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[62]:


model.doesnt_match("man woman child kitchen".split())


# In[63]:


#Vector Averaging
import numpy as np # Make sure that numpy is imported
def makeFeatureVec(words, model, num_features):
# Function to average all of the word vectors in a given
# paragraph
#
# Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
#
    nwords = 0.
#
# Index2word is a list that contains the names of the words in
# the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
#
# Loop over each word in the review and, if it is in the model's
# vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
#
# Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
# Given a set of reviews (each one a list of words), calculate
# the average feature vector for each one and return a 2D numpy array
#
# Initialize a counter
    counter = 0
#
# Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
# Loop through the reviews
    for review in reviews:
        if counter%1000 == 0:
            print ("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
#
# Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


# In[64]:


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
clean_train_reviews = []
for review in train1:
    clean_train_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))
    
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test1:
    clean_test_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))


# In[65]:


#clustering
from sklearn.cluster import KMeans
import time
start = time.time() # Start time
# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )#**************************************************
idx = kmeans_clustering.fit_predict( word_vectors )
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")


# In[66]:


# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))


# In[67]:


val= list(word_centroid_map.values())
key = list(word_centroid_map.keys())
# For the first 10 clusters
for cluster in range(0,10):
#
# Print the cluster number
    print ("\nCluster %d" % cluster)
#
# Find all of the words for that cluster number, and print them out
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( val[i] == cluster ):
            words.append(key[i])
    print(words)


# In[68]:


def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# In[69]:


train_centroids = np.zeros( (len(train1), num_clusters), dtype="float32" )
# Transform the training set reviews into bags of centroids
counter = 0

for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1
# Repeat for test reviews
test_centroids = np.zeros(( len(test1), num_clusters), dtype="float32" )
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1


# In[70]:


l_test1 = np.asarray(l_test)

from sklearn.ensemble import RandomForestClassifier
#Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators = 700,max_depth=80)#******************************
# Fitting the forest may take a few minutes
print ("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,l_train)
result = forest.predict(test_centroids)
# Write the test results

print ('Test Accuracy: %.2f'%forest.score(test_centroids, l_test1))


# In[71]:


from sklearn import linear_model, datasets

LGR = linear_model.LogisticRegression(penalty='l1', C=1.0)

LGR.fit(train_centroids, l_train)
result1 = LGR.predict(test_centroids)

print ('Test Accuracy: %.2f'%LGR.score(test_centroids, l_test1))


# In[72]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(random forrest  Classifier)')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[73]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

false_positive_rate, true_positive_rate, thresholds = roc_curve(l_test1, result1)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic(random forrest  Classifier)')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




