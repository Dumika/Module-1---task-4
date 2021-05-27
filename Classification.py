import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# grab the data
data = pd.read_csv('bayesTrain.csv')

# Let's see how many categories we have here
print(f"Total unique categories are: {len(data['Category'].value_counts())}")
print(f"Count of occurance of each category:")
print(data['Category'].value_counts())
X = data['News']
y = data['Category']
print(X.head())

# Split the data into 70-30 i.e. test size of 30% to check the accuracy of the training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)

#Let's check the shape of the splitted data
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)
X_train_cv.shape

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_cv,y_train)

# Transform the test data before predicting
X_test_cv = cv.transform(X_test)

# Form a prediction set
predictions = clf.predict(X_test_cv)



# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))
print(metrics.precision_score(y_test,predictions, average='micro'))
# 'micro'  will return the total ratio of tp/(tp + fp)
print(metrics.recall_score(y_test,predictions, average='micro'))
print(metrics.f1_score(y_test,predictions, average='micro'))

# Report the confusion matrix
print(metrics.confusion_matrix(y_test,predictions))
# Print a classification report
#print(metrics.classification_report(y_test,predictions))

X_test1 = "හරීන් පාර්ලිමේන්තුවේ ජනාධිපතිගෙන් ඉල්ලූ ඉල්ලීම"
print(X_test1)
X_test1 = [X_test1]
X_test1_cv = cv.transform(X_test1)
print(clf.predict(X_test1_cv))