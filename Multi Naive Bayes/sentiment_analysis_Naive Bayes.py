
import re
import csv
import nltk
from nltk.corpus import stopwords


def textPreProcessing(text):
    
    cf = re.compile('((//www\.[^\s]+)|(https?://[^\s]+)|(html)|(bit.[^\s]+)|((mailto\:|(news|(ht|f)tp(s?))\://){1}\S+))')
    text = cf.sub('',text)    
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub('@[\S]+','',text)    
    comp = re.compile('[/.,$%&*^!?~()-=+;:"<>'']')
    text = comp.sub(' ',text)    
    text = text.lower()
    text = re.sub('[\s]+', ' ', text)    
    c = re.compile(r"(.)\1{1,}", re.DOTALL)
    text = c.sub(r"\1\1", text)
    
    text = text.strip('\'"?.,')
    return text

StopwordList = set(stopwords.words("english"))

def getFeatureVector(text):
    featureVector = []
    words = text.split()
    for w in words:
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

def extract_features(text):
    word_in_text = set(text)
    features = {}
    for word in flist:
        features['contains(%s)' % word] = (word in word_in_text)
    return features

stopWords = []
sf = open('stopwords.txt', 'r')
line = sf.readline()
while line:
    word = line.strip()
    stopWords.append(word)
    line = sf.readline()
sf.close()

Positive_Train_Set = csv.reader(open('Positive_train_data.csv', 'rb'))
Negative_Train_Set = csv.reader(open('Negative_train_data.csv', 'rb'))

allyelp = []
flist =[]
for row in Positive_Train_Set:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'positive'));
    flist.extend(featureVector)

for row in Negative_Train_Set:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'negative'));
    flist.extend(featureVector)

flist = list(set(flist))
print flist

#classifier training
training_set = nltk.classify.util.apply_features(extract_features, allyelp)    

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

#################################output for positive test############################################

Positive_Test_Set = csv.reader(open('Positive_test_data.csv', 'rb'))
positiveCounter = 0.0
TotalPositive = 0.0

for row in Positive_Test_Set:
    testData = row[0]
    TotalPositive = TotalPositive + 1;    
    processedyelp = textPreProcessing(testData)
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
     
    if (result is 'positive'):
        positiveCounter = positiveCounter + 1

print positiveCounter
print TotalPositive


####################################output for negative sentiment#######################################

Negative_Test_Set = csv.reader(open('Negative_test_data.csv', 'rb'))

negativeCounter = 0.0
TotalNegative = 0.0

for row in Negative_Test_Set:
    testData = row[0]
    TotalNegative = TotalNegative + 1;    
    processedyelp = textPreProcessing(testData)
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
    
    if (result is 'negative'):
        negativeCounter = negativeCounter + 1

print negativeCounter
print TotalNegative

##################################F1 score ##############################################################
print("Overall F1 Score")
Overall_F1_Score = float((100 * float(2*(negativeCounter + positiveCounter)) /(float(2*(negativeCounter + positiveCounter)) + float((TotalNegative + TotalPositive) - (negativeCounter + positiveCounter)))))
print Overall_F1_Score

################################## Accuracy ##############################################################
print("accuracy")
accuracy=float(negativeCounter+positiveCounter)/(TotalNegative + TotalPositive)
print accuracy
