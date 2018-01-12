

import nltk
import re
import csv
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
    word = text.split()
    for w in word:
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

one_star_Train = csv.reader(open('one_star_train_data.csv', 'rb'))
two_star_Train = csv.reader(open('two_star_train_data.csv', 'rb'))
three_star_Train = csv.reader(open('three_star_train_data.csv', 'rb'))
four_star_Train = csv.reader(open('four_star_train_data.csv', 'rb'))
five_star_Train = csv.reader(open('five_star_train_data.csv', 'rb'))



allyelp = []
flist =[]
for row in one_star_Train:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'one_star'));
    flist.extend(featureVector)

for row in two_star_Train:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'two_star'));
    flist.extend(featureVector)
    
for row in three_star_Train:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'three_star'));
    flist.extend(featureVector)
    
for row in four_star_Train:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'four_star'));
    flist.extend(featureVector)

for row in five_star_Train:
    
    text = row[0]
    preProcessedText = textPreProcessing(text)
    featureVector = getFeatureVector(preProcessedText)
    allyelp.append((featureVector, 'five_star'));
    flist.extend(featureVector)

flist = list(set(flist))
print flist

#classifier training
train_classifier = nltk.classify.util.apply_features(extract_features, allyelp)    

NBClassifier = nltk.NaiveBayesClassifier.train(train_classifier)

#output for one star rating
onestarcounter = 0.0
totalonestar = 0.0
one_star_Test_Set = csv.reader(open('one_star_test_data.csv', 'rb'))

for row in one_star_Test_Set:
    testData = row[0]
    totalonestar = totalonestar + 1;    
    processedyelp = textPreProcessing(testData)
     
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
    
    if (result is 'one_star'):
        onestarcounter = onestarcounter + 1

print onestarcounter
print totalonestar

############################output for two star rating######################################
twostarCounter = 0.0
Totaltwostar = 0.0
two_star_Test_Set = csv.reader(open('two_star_test_data.csv', 'rb'))

for row in two_star_Test_Set:
    testData = row[0]
    Totaltwostar = Totaltwostar + 1;    
    processedyelp = textPreProcessing(testData)  
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
    if (result is 'two_star'):
        twostarCounter = twostarCounter + 1

print twostarCounter
print Totaltwostar

############################output for three star rating####################################

threestarcounter = 0
totalthreestar = 0

three_star_test_set = csv.reader(open('three_star_test_data.csv', 'rb'))
for row in three_star_test_set:
    testData = row[0]
    totalthreestar = totalthreestar + 1
    processedyelp = textPreProcessing(testData)
     
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp))) 
    if (result is 'three_star'):
        threestarcounter = threestarcounter + 1

print threestarcounter
print totalthreestar

###############################output for four star #######################################


fourstarcounter = 0.0
totalfourstar = 0.0
four_star_Test_Set = csv.reader(open('four_star_test_data.csv', 'rb'))

for row in four_star_Test_Set:
    testData = row[0]
    totalfourstar = totalfourstar + 1;    
    processedyelp = textPreProcessing(testData)  
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
    
    if (result is 'four_star'):
        fourstarcounter = fourstarcounter + 1

print fourstarcounter
print totalfourstar

###############################output for five star ########################################

fivestarcounter = 0.0
totalfivestar = 0.0

five_star_Test_Set = csv.reader(open('five_star_test_data.csv', 'rb'))

for row in five_star_Test_Set:
    testData = row[0]
    totalfivestar = totalfivestar + 1;    
    processedyelp = textPreProcessing(testData)  
    result = NBClassifier.classify(extract_features(getFeatureVector(processedyelp)))
    
    if (result is 'five_star'):
        fivestarcounter = fivestarcounter + 1

print fivestarcounter
print totalfivestar


print("Overall F1 Score")
Overall_F1_Score = float((100 * float(2*(fourstarcounter + onestarcounter)) /(float(2*(fourstarcounter + onestarcounter)) + float((totalfourstar + totalonestar) - (fourstarcounter + onestarcounter)))))
print Overall_F1_Score

print("accuracy")
accuracy=float(onestarcounter+twostarCounter+threestarcounter+fourstarcounter+fivestarcounter)/(totalonestar + Totaltwostar+totalthreestar+totalfourstar+totalfivestar)
print accuracy

