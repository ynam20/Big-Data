import nltk
nltk.download('words')
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd
import scipy
words = set(nltk.corpus.words.words())
nltk.download('stopwords')
stop_words = stopwords.words('english')
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def cleaning(text):
    import string
    exclude = set(string.punctuation)

    import re
    text = text.lower()
    # remove new line and digits with regular expression
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\d', '', text)
    # remove patterns matching url format
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', text)
    #stopwords
    text = " ".join(filter(lambda word: word.lower() not in stop_words, text.split()))
    # remove non-ascii characters
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    text = ''.join(character for character in text if ord(character) < 128)

    # remove punctuations
    text = ''.join(character for character in text if character not in exclude)
    text = ' '.join(word for word in text.split() if len(word) > 2)

    # standardize white space
    text = re.sub(r'\s+', ' ', text)

    #remove white space
    text = text.strip()
    return text

#import Hillary's tweets into allhilltweets
df = pd.read_csv('/Users/ooganam/Desktop/get_tweets/HillaryClinton_TWEET.csv', encoding='utf-8', usecols = ['text'], squeeze = True)
df = list(df)
allhilltweets = []
allwords = []
bagofwordstrain = []
bagofwordstest = []


hilltermfrequency = Counter()
trumptermfrequency = Counter()
trumpwordcount = 0
hillarywordcount = 0

for tweet in df:

    cleaned = cleaning(tweet)

    if len(cleaned.split(" "))>1:
        vectorizer = CountVectorizer(ngram_range=(2,2))
        X = vectorizer.fit_transform([cleaned])

        for word in cleaned.split(" "):
            if word != "the":
                allwords.append(word)
                hilltermfrequency[word] += 1
                hillarywordcount += 1
        allhilltweets.append(cleaned)

#import Trump's tweets into alltrumptweets
df1 = pd.read_csv('/Users/ooganam/Desktop/get_tweets/realDonaldTrump_TWEET.csv', encoding='utf-8', usecols = ['text'], squeeze = True)
df1 = list(df1)

alltrumptweets = []
for tweet in df1:
    cleaned1 = cleaning(tweet)
    if len(cleaned1.split(" "))>1:
        for word in cleaned1.split(" "):
            if word != "the":
                allwords.append(word)
                trumptermfrequency[word] += 1
                trumpwordcount += 1
        alltrumptweets.append(cleaned1)

allwords = set(allwords)
allwords = list(allwords)


def getindex(unit, allwhat):
    for i, j in enumerate(allwhat):
        if j == unit:
            return i

def countoccurrences(column):
    counter = 0
    for index in column:
        counter += index
    return counter

def deletecolumn(matrix):
    startinglength = len(matrix[0])
    column = 0
    allwords.append("sentiment")
    allwords.append("subjectivity")
    while column < startinglength:

        if (countoccurrences(matrix[:,column]) < 5):

            matrix = np.delete(matrix, column, 1)

            allwords.pop(column)



            startinglength = startinglength - 1
            column = column - 1
        column += 1
    return matrix

def makerow(tweet):
    occurences = np.zeros(len(allwords) + 2)
    length = len(occurences)

    for word in tweet.split(" "):
        if word != "the":
            occurences[getindex(word, allwords)] += 1
    occurences[length-2] = TextBlob(tweet).sentiment[0]
    occurences[length-1] = TextBlob(tweet).sentiment[1]

    return occurences

#combine hillary and trump's first 150 tweets each into one x training set
x_train =  allhilltweets[0:2499] + alltrumptweets[0:2499]
bagofwords = []
counter = 0
for tweet in x_train:
    counter += 1
    bagofwords.append(makerow(tweet))

#0 if hillary, 1 if trump
y_train = [0 if i < 2500 else 1 for i in range(5000)]

print(len(allhilltweets), len(alltrumptweets))

#combine the rest of their tweets into the x test set
x_test = allhilltweets[2500:3000] + alltrumptweets[2500:3000]

for tweet in x_test:
    bagofwords.append(makerow(tweet))
#0 if hillary, 1 if trump
y_test = [0 if i < 500 else 1 for i in range(998)]

bagofwords = np.array(bagofwords)
bagofwords = deletecolumn(bagofwords)
bagofwordstrain = np.array(bagofwords[0:5000])
bagofwordstest = np.array(bagofwords[5000:len(bagofwords)])

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [2,4,6]
}

model = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt', max_depth=4, n_estimators=400, oob_score = True)
model.fit(bagofwordstrain, y_train)
y_preds = model.predict(bagofwordstest)
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(y_preds, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(accuracy_score(y_preds, y_test), "single words")

def plot_confusion_matrix(df_confusion, cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title("Trump (1) vs Hillary (0) Confusion Matrix ")
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()
plot_confusion_matrix(df_conf_norm)


bagofwordstrain = pd.DataFrame(bagofwordstrain, columns = allwords)
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
indices = indices[:10]
# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices],
       color="r", align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()

print(indices[0])
print(indices[1])
print(indices[2])

for i, j in enumerate(allwords):
    print(i, j)

print(allwords[indices[0]])
print(allwords[indices[1]])
print(allwords[indices[2]])

bagofwordstrain = bagofwordstrain.values

hillsentiment= bagofwordstrain[:,len(bagofwordstrain[0]) - 2]
hillsentiment = hillsentiment[0:2499]
trumpsentiment= bagofwordstrain[:,len(bagofwordstrain[0]) - 2]
trumpsentiment = trumpsentiment[2500:5000]

print(np.mean(trumpsentiment), "Trump Mean Sentiment")
print(np.mean(hillsentiment), "Hillary Mean Sentiment")

hillTFIDF = defaultdict(float)
trumpTFIDF = defaultdict(float)
for word in hilltermfrequency:
    hillTFIDF[word] = (hilltermfrequency[word] / hillarywordcount)* np.log((2/1+trumptermfrequency[word]))

for word in trumptermfrequency:
    trumpTFIDF[word] = (trumptermfrequency[word] / trumpwordcount) * np.log((2 / 1 + hilltermfrequency[word]))

import operator
print("HILLARY TFIDF SCORES")
for k,v in sorted(hillTFIDF.items(), key=operator.itemgetter(1), reverse = True)[:5]:
    print (k,v)
print("TRUMP TFIDF SCORES")
for k,v in sorted(trumpTFIDF.items(), key=operator.itemgetter(1), reverse= True)[:5]:
    print (k,v)
t,p = scipy.stats.ttest_ind(trumpsentiment, hillsentiment) #CHANGE FOR FULL DATA
print("T test sentiment score results:           t = %g  p = %g" % (t, p))
