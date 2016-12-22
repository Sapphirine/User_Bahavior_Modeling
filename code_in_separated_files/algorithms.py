import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation

from sklearn.cluster import KMeans
from sklearn import datasets
import happybase
# remember to ./bin/hbase thrift start -threadpool


####################Naive Bayer Algorithm########################################
#Open the connection and extract the trainning data set from Hbase
connection = happybase.Connection('localhost', autoconnect=False)
connection.open()
table = connection.table('train')
columns = [b'l',b'f']
rows = table.scan(columns=columns)
rowsList = []
#cleanning the data---create a table of column names with feature ID and Label ID and click action value and feature values for each user
datamatrix=np.zeros((len(columns),13346+380+1), dtype=np.float)#380 labels and 13346 features
k=0
for key, data in rows:
    #feature (add the feature values to datamatrix)
    featurevalue=[data[x] for x in dict.keys(data) if  x.find('f')!=-1]
    featureno=[x for x in dict.keys(data) if  x.find('f')!=-1]
    #label(add the click action values to datamatrix)
    labelvalue=[data[x] for x in dict.keys(data) if  x.find('l')!=-1]
    labelno=[x for x in dict.keys(data) if  x.find('l')!=-1]    
    if len(featurevalue)!=0:
        j=0
        for b in featureno:
            col=int(b[2:])+380-1
            datamatrix[k,col]=float(featurevalue[j])
            j=j+1
    if len(labelvalue)!=0:
        i=0       
        for a in labelno:
            col=int(a[2:])-1
            datamatrix[k,col]=int(labelvalue[i])
            i=i+1
    k=k+1
 	 #save the datamatrix
    rowsList.append([key,data])
    
connection.close()
count=0
#to have non negative value on the click action. 
#2 if a user viewed and clicked on an ad
#1 if a user didn't view an ad
#0 if a user viewed but not clicked on an ad
datamatrix2=datamatrix
for i in range(0,len(columns)):
    for x in range(0,379):
        datamatrix2[i,x]=datamatrix[i,x]+1



#naive bayes

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(datamatrix2[:-len(columns),0:378], datamatrix2[:-len(columns),379])
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#5-fold cross validation
#tranning error
np.mean(prediction!=datamatrix2[-len(columns):,379])
results = cross_validation.cross_val_score(clf, datamatrix2[:-len(columns),0:378],datamatrix2[:-len(columns),379],cv=5)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
################Logistic Regression Algorithm####################################
#Open the connection and extract the trainning data set from Hbase
connection = happybase.Connection('localhost', autoconnect=False)
connection.open()
table = connection.table('train')
columns = [b'l',b'f']
rows = table.scan(columns=columns)
rowsList = []
#cleanning the data---create a table with only average feature values, label, and click action
datamatrixfinal=np.zeros((len(columns),3), dtype=np.float)#380 labels and 13346 features
#k as the kth column row for datamatrixfinal table
k=0

for key, data in rows:
    #feature
    featurevalue=[data[x] for x in dict.keys(data) if  x.find('f')!=-1]
    featureno=[x for x in dict.keys(data) if  x.find('f')!=-1]
    #label
    labelvalue=[data[x] for x in dict.keys(data) if  x.find('l')!=-1]
    labelno=[x for x in dict.keys(data) if  x.find('l')!=-1]
    #calculate the average feature value
    if len(featurevalue)!=0:
        j=0
        s=0
        for b in featureno:
            s=float(featurevalue[j])+s
            j=j+1
        featurevalue=k/j
    if len(labelvalue)!=0:
        i=0       
        for a in labelno:
            labelID=int(a[2:])-1
            #assign LabelID
            datamatrixfinal[k,0]=labelID
            #assign click action -1 to 0, thus 0 if the user doesn't click on the ad, 1 otherwise
            if int(labelvalue[i])<0:
                labelvalue[i]=0
            #assign the binary value whether the user click or not(-1,1)
            datamatrixfinal[k,2]=int(labelvalue[i])
            #assign the average feature value 
            datamatrixfinal[k,1]=featurevalue
            i=i+1
            k=k+1
    k=k+1
    #save the datamatrix
    rowsList.append([key,data])

connection.close()

#Plot the graph with label and average feature values
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
X=datamatrixfinal
y=X[:,2]
pos=where(y == 1)
neg = where(y == 0)
scatter(X[pos,0], X[pos, 1], marker='o', c='forestgreen')
scatter(X[neg, 0], X[neg, 1], marker='x', c='red')
plt.xlim(0,381)
plt.ylim(0,max(X[:,1])
xlabel('Label')
ylabel('Feature Value')
legend(['Click', 'Not Click'])
show()

#Implement logistic regression
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

pdmatrix=pd.DataFrame(datamatrixfinal)

#change the label and click action(0,1) to categorical data
pdmatrix[0]=(pdmatrix[0].astype(int)).astype('category')
pdmatrix[2]=(pdmatrix[2].astype(int)).astype('category')
datatype=pdmatrix.dtypes

# Convert label variable to numeric
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()
encoded_label = label_encoder.fit_transform(pdmatrix[0])
encoded_click = label_encoder.fit_transform(pdmatrix[2])
#independent variable
#normalize the average feature value
normavgfeaturevalue = [float(i)/sum(pdmatrix[1]) for i in pdmatrix[1]]

train_features = pd.DataFrame([encoded_label,
                              normavgfeaturevalue
                                          ]).T
                               
# Initialize logistic regression model
log_model = linear_model.LogisticRegression()

# Train the model
log_model.fit(X = train_features,
              y = pdmatrix[2])

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)

#Predict the value with test data set----------
preds = log_model.predict(X= train_features)
#tranning error
np.mean(preds-encoded_click)
results = cross_validation.cross_val_score(log_model,X,y,cv=5)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
# Generate table of predictions vs actual
pd.crosstab(preds,encoded_click)
correctrate=round(float(sum(preds==encoded_click))/float(len(pdmatrix[2])),2)
plt.figure(figsize=(9,9))
plt.plot(preds,   # X-axis rang# Predicted values
         color="red")


