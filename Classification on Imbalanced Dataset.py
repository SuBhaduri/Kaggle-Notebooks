#!/usr/bin/env python
# coding: utf-8

# #### Fraud identification problem is one of the cases where we will have imbalanced dataset i.e the fraud transactions will be very few in numbers. So if we don't apply specific techniques we will not be able to get a proper model.
# 
# #### In this notebook we will start applying standard approach of a classification problem and see why it doesn't work and what will be the alternative approaches.
# 
# #### Import the libraries.

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the data.

# In[42]:


data = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# #### Explore the data.

# In[43]:


data.head()


# In[44]:


data.info()


# #### So 'Class' is the target variable and let's see how it is distributed in the dataset and then we can understand why it is called a imbalanced dataset.

# In[45]:


data['Class'].value_counts()


# In[46]:


sns.countplot(x='Class', data=data)


# #### We can visualize the number of entries with Class=1 (Fraud) is very less compared to Class=0 (Not-Fraud).
# 
# #### Next we will split the data into train_set and validation_set.

# In[47]:


from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(data, test_size=0.25, random_state=42)
print(train_data.shape, val_data.shape)


# #### Since all V* columns are already scaled we will scale only the 'Time' and 'Amount' columns.

# In[48]:


train_label = train_data['Class']
val_label   = val_data['Class']
train_data  = train_data.drop(['Class'], axis=1)
val_data    = val_data.drop(['Class'], axis=1)
print(train_data.shape, val_data.shape, train_label.shape, val_label.shape)


# In[49]:


train_data.head(2)


# In[50]:


val_data.head(2)


# In[51]:


from sklearn.preprocessing import StandardScaler

std_scaler_Time   = StandardScaler()
std_scaler_Amount = StandardScaler()

train_data['Time']   = std_scaler_Time.fit_transform(train_data[['Time']])
train_data['Amount'] = std_scaler_Amount.fit_transform(train_data[['Amount']])

val_data['Time']   = std_scaler_Time.transform(val_data[['Time']])
val_data['Amount'] = std_scaler_Amount.transform(val_data[['Amount']])


# In[52]:


train_data.head(2)


# In[53]:


val_data.head(2)


# #### Create a standard function to display model performance metrices.

# In[54]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, classification_report

def model_def(model, model_name, m_train_data, m_train_label):
    model.fit(m_train_data, m_train_label)
    s = "predict_"
    p = s + model_name
    p = model.predict(m_train_data)
    cm = confusion_matrix(m_train_label, p)
    print("Confusion Matrix: \n", cm)
    cr = classification_report(m_train_label, p, target_names=['Not Fraud', 'Fraud'])
    print("Classification Report: \n", cr)
    precision = np.diag(cm)/np.sum(cm, axis=0)
    recall    = np.diag(cm)/np.sum(cm, axis=1)
    F1 = 2 * np.mean(precision) * np.mean(recall)/(np.mean(precision) + np.mean(recall))
    cv_score = cross_val_score(model, m_train_data, m_train_label, cv=10, scoring='recall')
    print("Mean CV Score     :", cv_score.mean())
    print("Std Dev CV Score  :", cv_score.std())


# #### Let's apply Logistic Regreesion algorithm on the dataset.

# In[55]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg, "logreg", train_data, train_label)


# #### Here you can see the problem. Our overall model accuracy is 1.00 but Recall for Fraud class is just 0.64 which means there is lots of misclassification for Fraud class. As per the confusion matrix out (242 + 137) = 379 Fraud class only 242 are classified properly and this has happened because of imbalanced representation of data.

# In[65]:


val_data_logreg = logreg.predict(val_data)
print("Logistic Regression: \n", confusion_matrix(val_label, val_data_logreg))


# #### We can see that just 68 out of 113 are predicted properly.

# #### Here is the alternative approach - creating synthetic data i.e the data that does not exist in the original dataset and there are 2 main techinques to do this - NearMiss and SMOTE.

# ### **NearMiss Algorithm – Undersampling**
# 
# #### NearMiss is an under-sampling technique. It aims to balance class distribution by randomly eliminating majority class examples. When instances of two different classes are very close to each other, it removes the instances of the majority class to increase the spaces between the two classes. 
# 
# #### The basic intuition about the working of near-neighbor methods is as follows:
# 
# #### Step 1: The method first finds the distances between all instances of the majority class and the instances of the minority class. Here, majority class is to be under-sampled.
# #### Step 2: Then, n instances of the majority class that have the smallest distances to those in the minority class are selected.
# #### Step 3: If there are k instances in the minority class, the nearest method will result in k * n instances of the majority class.
# 
# Reference : https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/

# In[57]:


from imblearn.under_sampling import NearMiss

print("Before Undersampling, counts of label '1': {}".format(sum(train_label == 1))) 
print("Before Undersampling, counts of label '0': {} \n".format(sum(train_label == 0))) 
  
nr = NearMiss() 
  
train_data_miss, train_label_miss = nr.fit_sample(train_data, train_label.ravel()) 
  
print("After Undersampling, counts of label '1': {}".format(sum(train_label_miss == 1))) 
print("After Undersampling, counts of label '0': {}".format(sum(train_label_miss == 0))) 


# #### So we can see count of majority class i.e Class=0 is reduced to the same count of Class=1 and the dataset has become balanced.
#  
# #### Now we will apply Logistic Regression with the same parameters on this undersampled data.

# In[58]:


logreg_miss = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg_miss, "logreg_miss", train_data_miss, train_label_miss)


# In[59]:


val_data_miss   = logreg_miss.predict(val_data)
print("Logistic Regression - Undersampling: \n", confusion_matrix(val_label, val_data_miss))


# #### Here we see the improvement - 104 out of 113 are predicted properly. So the model has become more accurate but at the cost of low precision.

# ### **SMOTE (Synthetic Minority Oversampling Technique) – Oversampling**
# 
# #### SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the imbalance problem.
# #### It aims to balance class distribution by randomly increasing minority class examples by replicating them.
# #### SMOTE synthesises new minority instances between existing minority instances. It generates the virtual training records by linear interpolation for the minority class. These synthetic training records are generated by randomly selecting one or more of the k-nearest neighbors for each example in the minority class.
# 
# Reference - https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/

# In[60]:


from imblearn.over_sampling import SMOTE

print("Before Oversampling, counts of label '1': {}".format(sum(train_label == 1))) 
print("Before Oversampling, counts of label '0': {} \n".format(sum(train_label == 0))) 
  
sm = SMOTE(random_state=42) 
  
train_data_SMOTE, train_label_SMOTE = sm.fit_sample(train_data, train_label.ravel()) 
  
print("After Oversampling, counts of label '1': {}".format(sum(train_label_SMOTE == 1))) 
print("After Oversampling, counts of label '0': {}".format(sum(train_label_SMOTE == 0))) 


# #### So we can see count of minority class i.e Class=1 is increased to the same count of Class=0 and the dataset has become balanced.
#  
# #### Now we will apply Logistic Regression with the same parameters on this oversampled data.

# In[61]:


logreg_SMOTE = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg_SMOTE, "logreg_SMOTE", train_data_SMOTE, train_label_SMOTE)


# In[62]:


val_data_SMOTE  = logreg_SMOTE.predict(val_data)
print("Logistic Regression - Oversampling: \n", confusion_matrix(val_label, val_data_SMOTE))


# #### Here we see the real advantage - 105 out of 113 are predicted properly. So the model has become more accurate and look at the Precision compared to that of NearMiss algorithm. So by far this is the most effective model.

# #### Since the most effective model by far is SMOTE and where the datset has become almost doubled we will fit this dataset to an Artificial Neural Network (ANN) to see how it goes.

# In[72]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[73]:


# Applying Neural Network
def build_classifier():
    classifier = Sequential([Dense(128, activation='relu', input_shape=(train_data_SMOTE.shape[1], )),
                             Dropout(rate=0.1),
                             Dense(64, activation='relu'),
                             Dropout(rate=0.1),
                             Dense(1, activation='sigmoid')])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision', 'Recall'])
    print(classifier.summary())
    return classifier

model = KerasClassifier(build_fn=build_classifier)


# In[74]:


history = model.fit(train_data_SMOTE, train_label_SMOTE,
                    batch_size=30,
                    epochs=10,
                    validation_data=(val_data, val_label))


# In[75]:


val_data_Neural = model.predict(val_data)
print("Artificial Neural Network: \n", confusion_matrix(val_label, val_data_Neural))


# #### So we can see in the ANN model the Precision is highest but the Recall is compromised slightly as compared to SMOTE.

# In[ ]:




