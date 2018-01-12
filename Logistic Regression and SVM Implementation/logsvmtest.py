import numpy as np
import pandas as pd
import re, nltk
import random
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression       


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
    
    text = re.sub('[\s]+', ' ', text)
    text = re.sub("  ", " ", text)
    text = re.sub("haha","", text)
    text = re.sub("hahaha","", text)    
    text = text.strip('\'"?,.??')    
    text.replace('?',"")
    text.replace('??',"")    
    text = re.sub("`","", text)
    text = re.sub("#([^\s]+,$!&*)_.","", text)
    
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)

Pos_Test = 39759
Neg_Test = 19639

#reading train and test data
test_file = 'Yelp_positive_test.csv'
train_file = 'Yelp_training_data.csv'


test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"] 
print(train_data.Text)

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
#converting feature vectore into an array
data_feature = df.toarray() 
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

#cross validation
X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)
        
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#defining the training model 
log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000,max_iter = 200)
#fitting the model
log_model = log_model.fit(X=X_train, y=y_train)
#predicting on the test part for cross validation
y_pred = log_model.predict(X_test)
print 'Logisitic Cross-Validation'
print(classification_report(y_test, y_pred))

#defining parameters for testing

log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000,max_iter = 200)
print("a")
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)
print("b")

#predicting on the test data
test_pred = log_model.predict(data_feature[len(train_data):])

spl = random.sample(xrange(len(test_pred)), Pos_Test)

print("c")

#Counting the no. of correct predictions for logistic regression model
PosLogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 1:
        PosLogcount = PosLogcount + 1
    

#training and testing SVM on the same positive test data
log_model = svm.SVC()
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)
#print(classification_report(y_test, y_pred))

log_model = svm.SVC()
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)

# get predictions
test_pred = log_model.predict(data_feature[len(train_data):])

# sample some of them
spl = random.sample(xrange(len(test_pred)), Pos_Test)

#counting the number of correct positive prediction
PosSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 1:
        PosSVMcount = PosSVMcount + 1
    

###########################################
#Doing the same operations on the negative dataset
##########################################

test_file = 'Yelp_negative_Test.csv'
train_file = 'Yelp_training_data.csv'



test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df= vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature= df.toarray()
vob = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)#same parameter values as for positive
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)
print(classification_report(y_test, y_pred))

log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000, max_iter = 200)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])

# sample some of them
spl = random.sample(xrange(len(test_pred)), Neg_Test)

# print text and labels
NegLogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 0:
        NegLogcount = NegLogcount + 1
    #print sentiment, text

log_model = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=90)
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)
print 'SVM Cross Validation'
print(classification_report(y_test, y_pred))

log_model = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=90)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)


test_pred = log_model.predict(data_feature[len(train_data):])

spl = random.sample(xrange(len(test_pred)), Neg_Test)


NegSVMcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 0:
        NegSVMcount = NegSVMcount + 1


print 'Positive Logistic Regression count=', PosLogcount
print 'Positive SVM count=', PosSVMcount
print 'Negative Logistic Regression count', NegLogcount
print 'Negative SVM count', NegSVMcount

accuracysvm=(NegSVMcount+PosSVMcount)/float(Pos_Test+Neg_Test)
print 'Support Vector Machine accuracy', accuracysvm
accuracylog=(PosLogcount+NegLogcount)/float(Pos_Test+Neg_Test)
print 'Logistic Regression accuracy', accuracylog

F1_score_log = 2*(PosLogcount+NegLogcount)/float((2*(PosLogcount+NegLogcount))+(Pos_Test - PosLogcount)+(Neg_Test - NegLogcount))#F1 score calculation for Logistic Regression
print 'F1 score for Logistic Regression ', F1_score_log

F1_score_SVM = 2*(PosSVMcount+NegSVMcount)/float((2*(PosSVMcount+NegSVMcount))+(Pos_Test - PosSVMcount)+(Neg_Test - NegSVMcount))#F1 score calculation for Support Vector Machine
print 'F1 score for Support Vector Machine', F1_score_SVM



