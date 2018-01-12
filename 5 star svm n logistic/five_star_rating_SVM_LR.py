import csv
import pandas as pd
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer        
import numpy as np
import random
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm
import time

start_time=time.time()

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):     
    text = re.sub("\d+", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("@[^\s]+:/"," ",text)
    #text = re.sub("?","",text)
    text = re.sub('[\s]+', ' ', text)
    text = re.sub("  ", " ", text)
    text = re.sub("haha","", text)
    text = re.sub("hahaha","", text)    
    text = text.strip('\'"?,.??')    
    text.replace('?',"")
    text.replace('??',"")    
   
    text = re.sub("`","", text)
    text = re.sub("#([^\s]+,$!&*)_.","", text)
    
    tokens = nltk.word_tokenize(text)
    
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)

one_star_size = 7896
two_star_size = 5009
three_star_size=6734
four_star_size=15264
five_star_size=24495

test_file = 'one_star_test.csv'#reading psitive test data
train_file = 'training_data.csv'#reading train data


test_data = pd.read_csv(test_file, header=None)#Reading the positive tweets from test data file
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()#reading texts and sentiments from training data file
train_data.columns = ["Sentiment","Text"] #Assigning sentiments to tweets


df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist()) #vectorizing and storing in corpora
data_feature = df.toarray() #feature vector is converted into an array
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)#count of features

#85:15 split for cross validation
X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        
        train_size=0.80, 
        random_state=1234)
        
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#defining the training model by setting parameter values
log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
#fitting the model
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)
print 'Logisitic Cross-Validation'
print(classification_report(y_test, y_pred))
#defining parameters for testing
log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)

#predicting on the test data
test_pred = log_model.predict(data_feature[len(train_data):])

spl = random.sample(xrange(len(test_pred)), one_star_size)
#counting the number of correct one star prediction
onelogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 1:
        onelogcount = onelogcount + 1
    #print sentiment, text


#training and testing SVM on the same psoitive test data
log_model = svm.SVC()
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = svm.SVC()
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)

# get predictions
test_pred = log_model.predict(data_feature[len(train_data):])

# sample some of them
spl = random.sample(xrange(len(test_pred)), one_star_size)

# print text and labels
oneSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 1:
        oneSVMcount = oneSVMcount + 1
print("onesvm")
   

##########################################################################################################################

##########################################################################################################################

test_file = 'two_star_test.csv'
train_file = 'training_data.csv'


test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature = df.toarray()
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), two_star_size)


twologcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 2:
        twologcount = twologcount + 1


log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])

spl = random.sample(xrange(len(test_pred)), two_star_size)


twoSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 0:
        twoSVMcount = twoSVMcount + 1

################################################################################################################
        
###############################################################################################################
test_file = 'three_star_test.csv'
train_file = 'training_data.csv'
test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature = df.toarray()
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000, max_iter = 200)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), three_star_size)


threelogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 3:
        threelogcount = threelogcount + 1
   

log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), three_star_size)


threeSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 3:
        threeSVMcount = threeSVMcount + 1
    
####################################################################################################################
    
####################################################################################################################
test_file = 'four_star_test.csv'
train_file = 'training_data.csv'
test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature = df.toarray()
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000, max_iter = 200)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), four_star_size)

fourlogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 4:
        fourlogcount = fourlogcount + 1
   
log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), four_star_size)


fourSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 4:
        fourSVMcount = fourSVMcount + 1
###################################################################################################################
        
####################################################################################################################
test_file = 'five_star_test.csv'
train_file = 'training_data.csv'


test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature = df.toarray()
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000, max_iter = 200)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), five_star_size)


fiveLogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 5:
        fiveLogcount = fiveLogcount + 1
  

log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

log_model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=500)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])


spl = random.sample(xrange(len(test_pred)), five_star_size)


fiveSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 5:
        fiveSVMcount = fiveSVMcount + 1
   
##################################################################################################################
        
##################################################################################################################


accuracysvm=(oneSVMcount+twoSVMcount+fiveSVMcount+fourSVMcount+threeSVMcount)/float(one_star_size+two_star_size+three_star_size+four_star_size+five_star_size)
print 'Support Vector Machine accuracy', accuracysvm

accuracylog=(onelogcount+twologcount+threelogcount+fourlogcount+fiveLogcount)/float(one_star_size+two_star_size+three_star_size+four_star_size+five_star_size)
print 'logistic regression accuracy', accuracylog

f1_score_LR= 2*((onelogcount+twologcount+threelogcount+fourlogcount+fiveLogcount))/float(2*(onelogcount+twologcount+threelogcount+fourlogcount+fiveLogcount)+(one_star_size-onelogcount)+(two_star_size-twologcount)+(three_star_size-threelogcount)+(four_star_size-fourlogcount)+(five_star_size-fiveLogcount))
print 'logistic regression F1 score', f1_score_LR

f1_score_svm=2*(oneSVMcount+twoSVMcount+fiveSVMcount+fourSVMcount+threeSVMcount)/float(2*(oneSVMcount+twoSVMcount+fiveSVMcount+fourSVMcount+threeSVMcount)+(one_star_size-oneSVMcount)+(two_star_size-twoSVMcount)+(three_star_size-threeSVMcount)+(four_star_size-fourSVMcount)+(five_star_size-fiveSVMcount))
print 'svm f1 scre', f1_score_svm

end_time=time.time()

total_time=end_time-start_time

print 'total time for execution', total_time