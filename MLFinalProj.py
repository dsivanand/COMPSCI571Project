from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -sf')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np

#Importing Data
data = pd.read_csv('seeds_dataset.csv')
#display(data.head())

#Data Visualization
plt.figure(figsize=(14,28) )
plt.subplot(4,2,1)
plt.plot(data['A'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Area')
plt.subplot(4,2,2)
plt.plot(data['P'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Perimeter')
plt.subplot(4,2,3)
plt.plot(data['C'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Compactness')
plt.subplot(4,2,4)
plt.plot(data['Lk'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Length of kernel')
plt.subplot(4,2,5)
plt.plot(data['Wk'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Width of kernel')
plt.subplot(4,2,6)
plt.plot(data['Ac'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Asymmetric Coefficient')
plt.subplot(4,2,7)
plt.plot(data['Lkg'],data['Result'],'o')
plt.ylabel('Seed')
plt.xlabel('Length of kernel groove')
plt.title('Each parameter V Classification')

#%%

Log_score1 = np.zeros(1001)
SVM_score1 = np.zeros(1001)
RF_score1 = np.zeros(1001)
K_score1 = np.zeros(1001)
NB_score1 = np.zeros(1001)
LDA_score1 = np.zeros(1001)

#Basic controls
fr = [0.25, 0.3, 0.4, 0.5, 0.6, 0.75] # Fraction of total data for TEST DATA
z = 1
for y in fr:
 for x in range(1001):
  #Multi-Class Logistic Regression Fit
  Ltrain, Ltest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  model = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=2000)
  model.fit(Ltrain,Rtrain)
  Log_score1[x] = model.score(Ltest,Rtest)
  Log_pred = model.predict(Ltest)

  #Support Vector Method
  Strain, Stest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  model = SVC(gamma = 'auto')
  model.fit(Strain,Rtrain)
  SVM_score1[x] = model.score(Stest,Rtest)
  SVM_pred = model.predict(Stest)

  #Random forest Method
  RFtrain, RFtest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(RFtrain,Rtrain)
  RF_score1[x] = rf.score(RFtest,Rtest)
  RF_pred = rf.predict(RFtest)

  #K-nearest Neighbours Method
  Ktrain, Ktest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  Kmodel = KNeighborsClassifier()
  Kmodel.fit(Ktrain,Rtrain)
  K_score1[x] = Kmodel.score(Ktest,Rtest)
  K_pred = Kmodel.predict(Ktest)

  #Naive Bayes Algorithm
  NBtrain, NBtest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  NBmodel = MultinomialNB()
  NBmodel.fit(NBtrain,Rtrain)
  NB_score1[x] = Kmodel.score(NBtest,Rtest)
  NB_pred = Kmodel.predict(NBtest)

  #Linear Discriminant Analysis (LDA)
  LDAtrain, LDAtest, Rtrain, Rtest = train_test_split(data.drop('Result',axis=1),data['Result'], test_size=y,random_state = x)
  LDAmodel = LinearDiscriminantAnalysis()
  LDAmodel.fit(LDAtrain,Rtrain)
  LDA_score1[x] = LDAmodel.score(LDAtest,Rtest)
  LDA_pred = LDAmodel.predict(LDAtest)
  print('Progress: (', z , '/',len(fr),') ... ', np.around(x*100/1001,decimals=2),' % iterations complete')
  get_ipython().magic('clear')

 Log_score = np.mean(Log_score1)
 SVM_score = np.mean(SVM_score1)
 RF_score = np.mean(RF_score1)
 K_score = np.mean(K_score1)
 NB_score = np.mean(NB_score1)
 LDA_score = np.mean(LDA_score1)
 

 #Comparing scores for all methods
 Modeltxt = ['Multi-class logistic Regression','Support Vector Method','Random forest Method','K-nearest Neighbours Method','Naive Bayes Algorithm','Linear Discriminant Analysis']
 ModelScore = [Log_score,SVM_score,RF_score,K_score,NB_score,LDA_score]
 n = 3
 rdnd = [np.around(Log_score, decimals=n),np.around(SVM_score,decimals=n),np.around(RF_score, decimals=n),np.around(K_score, decimals=n),np.around(NB_score, decimals=n),np.around(LDA_score, decimals=n)]
 plt.figure() 
 plt.barh(Modeltxt, ModelScore, align='center')
 plt.xlabel('Score')
 plt.ylabel('Classification method')
 plt.xlim([0.8,1])
 plt.xticks([0.8, 0.85, 0.9, 0.95, 1])
 tittle = 'Scores for ' + str((1-y)*100) + ' % of full data as training data'
 plt.title(tittle)
 for i, v in enumerate(rdnd):
    plt.text(v+0.01, i-0.08, str(v), color='black', fontweight='bold')
 z = z+1
 