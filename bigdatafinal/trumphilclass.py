import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

words = set(nltk.corpus.words.words())

nltk.download('stopwords')
stop_words = stopwords.words('english')

def cleaning(text):
    import string
    exclude = set(string.punctuation)

    import re

    # remove new line and digits with regular expression
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\d', '', text)
    # remove patterns matching url format
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', text)
    #stopwords
    text = " ".join(filter(lambda word: word not in stop_words, text.split()))

    # remove non-ascii characters
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    text = ''.join(character for character in text if ord(character) < 128)

    # remove punctuations
    text = ''.join(character for character in text if character not in exclude)
    text = ' '.join(word for word in text.split() if len(word) > 2)

    # standardize white space
    text = re.sub(r'\s+', ' ', text)

    # drop capitalization
    text = text.lower()

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

allbigrams = []


for tweet in df:
    if len(cleaning(tweet).split(" "))>1:
        vectorizer = CountVectorizer(ngram_range=(2,2))
        X = vectorizer.fit_transform([cleaning(tweet)])

        for bigram in vectorizer.get_feature_names():

            allbigrams.append(bigram)
        for word in cleaning(tweet).split(" "):
            allwords.append(word)
        allhilltweets.append(cleaning(tweet))

#import Trump's tweets into alltrumptweets
df1 = pd.read_csv('/Users/ooganam/Desktop/get_tweets/realDonaldTrump_TWEET.csv', encoding='utf-8', usecols = ['text'], squeeze = True)
df1 = list(df1)

alltrumptweets = []
for tweet in df1:
    if len(cleaning(tweet).split(" "))>1:
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        X = vectorizer.fit_transform([cleaning(tweet)])
        for bigram in vectorizer.get_feature_names():
            allbigrams.append(bigram)
        for word in cleaning(tweet).split(" "):
            allwords.append(word)
        alltrumptweets.append(cleaning(tweet))

allbigrams = set(allbigrams)

allwords = set(allwords)

def getindex(bigram):
    for i, j in enumerate(allbigrams):
        if j == bigram:
            return i

from textblob import TextBlob

def countoccurrences(column):
    counter = 0
    for index in column:
        counter += index
    return counter

def deletecolumn(matrix):
    startinglength = len(matrix[0])
    column = 0
    while column < startinglength:
        if (countoccurrences(matrix[:,column]) < 5):
            matrix = np.delete(matrix, column, 1)
            startinglength = startinglength - 1
            column = column - 1
        column += 1
    return matrix



def makerow(tweet):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    # The ngram range specifies your ngram configuration.

    X = vectorizer.fit_transform([tweet])

    #occurences = np.zeros(len(allwords) + 2)
    occurences = np.zeros(len(allbigrams) + 2)
    length = len(occurences)

    #for word in tweet.split(" "):
    for bigram in vectorizer.get_feature_names():

        occurences[getindex(bigram)] += 1
    occurences[length - 2] = TextBlob(tweet).sentiment[0]
    occurences[length - 1] = TextBlob(tweet).sentiment[1]
    return occurences
#combine hillary and trump's first 2000 tweets each into one x training set
x_train =  allhilltweets[0:150] + alltrumptweets[0:150]

bagofwords = []

for tweet in x_train:
    bagofwords.append(makerow(tweet))
#0 if hillary, 1 if trump
y_train = [0 if i < 2000 else 1 for i in range(4000)]

#combine the rest of their tweets into the x test set
x_test = allhilltweets[:len(allhilltweets)] + alltrumptweets[150:len(alltrumptweets)]
for tweet in x_test:
    bagofwords.append(makerow(tweet))
#0 if hillary, 1 if trump
y_test = [0 if i < len(allhilltweets[150:len(allhilltweets)]) else 1 for i in range(len(x_test))]

bagofwords = np.array(bagofwords)
bagofwords = deletecolumn(bagofwords)
bagofwordstrain = np.array(bagofwords[0:300])
bagofwordstest = np.array(bagofwords[300:len(bagofwords)])



param_grid = {
    'n_estimators': [200, 400, 600],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [2,4,6]
}

from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
model.fit(bagofwordstrain, y_train)

y_preds = model.predict(bagofwordstest)

import pandas as pd
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(y_preds, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_preds, y_test))



import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()
plot_confusion_matrix(df_conf_norm)

