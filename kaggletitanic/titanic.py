# Silencing warnings from scikit-learn
import warnings
warnings.filterwarnings("ignore")

# Importing libraries
import numpy as np
from statistics import median
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def processdata(dataset, trainortest):
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # Convert to make more common
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    dataset['Family_size'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = np.where(dataset['Family_size'] != 1, 0, 1)
    dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    dataset['Age'].fillna(data.Age.median(), inplace=True)
    dataset['Family'] = [sum(x) for x in zip(dataset['SibSp'], dataset['Parch'])]
    dataset['Cabin'].fillna(0, inplace=True)

    dataset['Cabin'] = [dict[dataset['Cabin'][i][0]] if dataset['Cabin'][i] != 0 else 0 for i in range(len(dataset['Cabin'])) ]

    if trainortest==0:
        x = dataset.drop(['Survived','PassengerId', 'Name', 'Ticket'], axis=1)
    else:
        x = dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    return x


data = pd.read_csv('/Users/ooganam/Desktop/train.csv')
y_tr = data['Survived']
x_tr = processdata(data, 0)



'''parametergridNN = [{'hidden_layer_sizes':[10*i for i in range(6,13)]}]

clfNN = GridSearchCV(MLPClassifier(), parametergridNN, cv = 5)

clfNN.fit(x_tr, y_tr)

testdata = pd.read_csv('/Users/ooganam/Desktop/test.csv')
passengerid = testdata['PassengerId']
x_te = processdata(testdata, 1)

x_te.fillna(x_te.mean(), inplace = True)

ypred = clfNN.predict(x_te)
finaly = np.column_stack((passengerid, ypred))'''
rfc=RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200,300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,6,7,8],
    'criterion' :['gini', 'entropy']
}
rfc1=GridSearchCV(cv=4, error_score='raise',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_estimators': [200, 300], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4, 6, 8], 'criterion': ['gini', 'entropy']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
rfc1.fit(x_tr, y_tr)
testdata = pd.read_csv('/Users/ooganam/Desktop/test.csv')
passengerid = testdata['PassengerId']
x_te = processdata(testdata, 1)

x_te.fillna(x_te.mean(), inplace = True)

ypred = rfc1.predict(x_te)
finaly = np.column_stack((passengerid, ypred))

np.savetxt("titanic.csv", finaly, delimiter=",")
#df.to_csv(index=False)



