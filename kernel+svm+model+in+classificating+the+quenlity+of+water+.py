
# coding: utf-8

# In[18]:

import numpy as np
import pandas as pd
#import data#

datafile='d:/python/moment.csv'
data=pd.read_csv(datafile,encoding='gbk')
x = data.iloc[:, 2: 10].values
y = data.iloc[:, 0].values


# In[19]:

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[22]:

#expend the character
x_train=x_train*30
x_test=x_test*30
y_train=y_train.astype(int)
y_test=y_test.astype(int)


# In[23]:

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[25]:

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
#cv is the number of fold want to be use 
accuracies.mean()
accuracies.std()


# In[ ]:



