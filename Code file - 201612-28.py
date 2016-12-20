# Data Generation
import csv
from collections import deque
import os

""" clean sample data, training data and test data"""
zeroBasedArrayFlag = 1
dataFilePath = ['./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-sample-v1_0.txt']#,
                #'./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-train-v1_0',
                #'./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-test-v1_0']
# featureFilePath = ['sampleFeaturesLibSVM.txt', 'trainingFeaturesLibSVM.txt', 'testFeaturesLibSVM.txt']
# labelFilePath = ['sampleLabelsLibSVM.txt', 'trainingLabelsLibSVM.txt', 'testLabelsLibSVM.txt']
# for i in range(0,len(dataFilePath)):
#     with open(dataFilePath[i], 'r') as dataFile, \
#             open(featureFilePath[i], 'w') as featureFile, open(labelFilePath[i], 'w') as labelFile:
#         for line in dataFile:
#             featuresAndLabels = line.partition('\t')
#             featureFile.write(featuresAndLabels[0])
#             featureFile.write('\n')
#             featureFile.flush()
#             labelFile.write(featuresAndLabels[2])
#             # labelFile.write('\n') python keeps \n in line so don't need to write('\n') again
#             labelFile.flush()

# featureFilePath = ['sampleFeatures.txt', 'trainingFeatures.txt', 'testFeatures.txt']
# labelFilePath = ['sampleLabels.txt', 'trainingLabels.txt', 'testLabels.txt']
# for i in range(0,len(dataFilePath)):
#     with open(dataFilePath[i], 'r') as dataFile, \
#             open(featureFilePath[i], 'w') as featureFile, open(labelFilePath[i], 'w') as labelFile:
#         for line in dataFile:
#             featuresAndLabels = line.partition('\t')
#             pairs = featuresAndLabels[0].split(' ')
#             index = []
#             val = []
#             for aPair in pairs:
#                 aPair = aPair.split(':')
#                 index.append(int(aPair[0]) - zeroBasedArrayFlag)
#                 val.append(float(aPair[1]))
#             featureFile.write(str(index) + '\t')
#             featureFile.write(str(val) + '\n')
#             featureFile.flush()
#             pairs = featuresAndLabels[2].strip().split(' ')
#             index = []
#             val = []
#             for aPair in pairs:
#                 aPair = aPair.split(':')
#                 index.append(int(aPair[0]) - zeroBasedArrayFlag)
#                 val.append(int(aPair[1]))
#             labelFile.write(str(index) + '\t')
#             labelFile.write(str(val) + '\n')
#             labelFile.flush()

# hiveFilePath = ['sampleHiveHeader.txt', 'trainingHiveHeader.txt', 'testingHiveHeader.txt']
# tableName = ['sample', 'training', 'testing']
# for i in range(0,len(dataFilePath)):
#     with open(dataFilePath[i], 'r') as dataFile, \
#             open(hiveFilePath[i], 'w') as hiveFile:
#         hiveFile.write("create table " + tableName[i])
#         hiveFile.write(" (l1 DOUBLE")
#         for j in range(2, 13347):
#             hiveFile.write(", f" + str(j) + " DOUBLE")
#         for j in range(1, 381):
#             hiveFile.write(", l" + str(j) + " INT")
#         hiveFile.write(")")
#         hiveFile.flush()
#
# denseDataFilePath = ['sampleDenseData.txt', 'trainingDenseData.txt', 'testingDenseData.txt']
# for i in range(0,len(dataFilePath)):
#     with open(dataFilePath[i], 'r') as dataFile, \
#             open(denseDataFilePath[i], 'w') as denseDataFile:
#         for line in dataFile:
#             colCounter=1
#             featuresAndLabels = line.partition('\t')
#             pairs = featuresAndLabels[0].split(' ')
#             for aPair in pairs:
#                 aPair = aPair.split(':')
#                 while colCounter < int(aPair[0]):
#                     denseDataFile.write("0,")
#                     colCounter += 1
#                 denseDataFile.write(aPair[1] + ",")
#                 colCounter += 1
#             while colCounter <= 13346:
#                 denseDataFile.write("0,")
#                 colCounter += 1
#             denseDataFile.flush()
#             pairs = featuresAndLabels[2].strip().split(' ')
#             colCounter = 1
#             for aPair in pairs:
#                 aPair = aPair.split(':')
#                 while colCounter < int(aPair[0]):
#                     denseDataFile.write("0,")
#                     colCounter += 1
#                 denseDataFile.write(aPair[1] + ",")
#                 colCounter += 1
#             while colCounter <= 380:
#                 denseDataFile.write("0,")
#                 colCounter += 1
#             positionNow = denseDataFile.tell()
#             denseDataFile.seek(positionNow-1, os.SEEK_SET)
#             denseDataFile.write("\n")
#             denseDataFile.flush()

hBaseFilePath = ['sampleHBase.txt', 'trainingHBase.txt', 'testingHBase.txt']
tableName = ['sample', 'train', 'test']
for i in range(0,len(dataFilePath)):
    with open(dataFilePath[i], 'r') as dataFile, \
            open(hBaseFilePath[i], 'w') as hBaseFile:
        hBaseFile.write("create '" + tableName[i] + "'")
        hBaseFile.write(", {NAME=>'f', VERSIONS=>2}, {NAME=>'l', VERSIONS=>2}")
        hBaseFile.write('\n')
        hBaseFile.flush()
        table = tableName[i]
        lineCounter = 0
        for line in dataFile:
            lineCounter += 1
            featuresAndLabels = line.partition('\t')
            pairs = featuresAndLabels[0].split(' ')
            for aPair in pairs:
                hBaseFile.write("put '" + table + "', " + str(lineCounter))
                aPair = aPair.split(':')
                hBaseFile.write(", 'f:" + aPair[0] + "', " + aPair[1] + ", 1\n")
            hBaseFile.flush()
            pairs = featuresAndLabels[2].strip().split(' ')
            for aPair in pairs:
                hBaseFile.write("put '" + table + "', " + str(lineCounter))
                aPair = aPair.split(':')
                hBaseFile.write(", 'l:" + aPair[0] + "', " + aPair[1] + ", 1\n")
            hBaseFile.flush()
        hBaseFile.write('exit\n')
        hBaseFile.flush()
exit()


""" cleaning taxonomy tree """
class tree:
    def __init__(self, nodeID):
        self.nodeID = nodeID
        self.children = []

parentToChildrenDict = {}
with open('./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-taxonomy-v1_0.txt', newline='') as csvfile:
    nodeReader = csv.reader(csvfile, delimiter=':')
    for row in nodeReader:
        parentID = int(row[1])
        if parentID in parentToChildrenDict:
            parentToChildrenDict[parentID].append(int(row[0]))
        else:
            parentToChildrenDict[parentID] = [int(row[0])]

taxonomyTree = tree(-1)
stack = [taxonomyTree]
leafList = []
nodeList = []
while len(stack) != 0:
    node = stack.pop()
    if node.nodeID not in parentToChildrenDict:
        leafList.append(node.nodeID)
        continue
    nodeList.append(node.nodeID)
    children = parentToChildrenDict[node.nodeID]
    children.sort()
    for childID in children:
        newNode = tree(childID)
        node.children.append(newNode)
        stack.append(newNode)

leafList.sort()
nodeList.sort()
with open('./taxonomyLeaves.txt', 'w') as leavesFile, open('./taxonomyNodes.txt', 'w') as nodesFile:
    leavesFile.write(str(leafList))
    nodesFile.write(str(nodeList))
# this representation is not useful
# f = open('./taxonomyTree.txt', 'w')
# f.write('[-1]\n')
# f.flush()
# queue = deque([taxonomyTree])
# while len(queue) != 0:
#     currentSize = len(queue)
#     while currentSize > 0:
#         node = queue.popleft()
#         if node.children != []:
#             f.write('[')
#             for i in node.children:
#                 f.write(str(i.nodeID) + ' ')
#             f.write(']\t')
#         queue.extend(node.children)
#         currentSize -= 1
#     f.write('\n')
#     f.flush()
#
# f.close()


def output_tree_in_newick_format(node, f):
    if not node.children:
        f.write('(')
        for i in node.children:
            output_tree_in_newick_format(i, f)
            if i != node.children[len(node.children)-1]:
                f.write(',')
        f.write(')')
    f.write(str(node.nodeID))
    f.flush()
    return

f = open('./newickFormat.txt', 'w')
output_tree_in_newick_format(taxonomyTree, f)
f.write(';')
f.close()

# Feature selection
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import ast
import time
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import happybase

appName = 'PCASVM'
master = 'local[4]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
spark = SparkSession \
        .builder \
        .appName("PCASVM") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

def labelsToSparseVec(line):
    pair = line.split('\t')
    return (Vectors.sparse(380, ast.literal_eval(pair[0]), ast.literal_eval(pair[1])),)
def featuresToSparseVec(line):
    pair = line.split('\t')
    return (Vectors.sparse(13346, ast.literal_eval(pair[0]), ast.literal_eval(pair[1])),)
def featuresToSparseVecFromLine(line):
    indexList = []
    valList = []
    k = 100
    for i in range(1, fyTime):
        if 'f:'+str(i) in line:
            indexList.append(i)
            valList.append(lineDict['f:'+str(i)])
    return (Vectors.sparse(fyTime, indexList, valList),)

with open("taxonomyNodes.txt") as nodeFile:
    for line in nodeFile:
        nodes = ast.literal_eval(line)

for col in nodes:
    connection = happybase.Connection('localhost', autoconnect=False)
    connection.open()
    table = connection.table('train')
    columns = ['l:'+str(col), 'f']
    rows = table.scan(columns=columns, filter="SingleColumnValueFilter ('l','" + str(col) + "',!=,'regexstring:0', true, true)")
    lines = []
    for key, data in rows:
        lines.append(data)
    connection.close()
    features = []
    for line in lines:
        indexList = []
        valList = []
        k = 100
        for i in range(1, fyTime):
            if 'f:'+str(i) in line:
                indexList.append(i)
                valList.append(lineDict['f:'+str(i)])
        features.append((Vectors.sparse(fyTime, indexList, valList),))
    features = sc.parallelize(features)
    #sclines = sc.parallelize(lines)
    #features = sclines.map(featuresToSparseVecFromLine)
    featureDataFrame = spark.createDataFrame(features, ["features"])
    pca = PCA(k=100, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(featureDataFrame)
    pcaresult = model.transform(featureDataFrame).select("pcaFeatures").collect()
    lp = []
    c = 0
    for com in pcaresult:
        lp.append(LabeledPoint(lines[c]['l:' + str(col)], com))
        c += 1
    lp = sc.parallelize(lp)
    model = SVMWithSGD.train(lp, iterations=1000)
    labelsAndPreds = lp.map(lambda p: (p.label, model.predict(p.features)))
    err = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
    print("err at node" + str(col) + " = " + str(err))

sc.stop()

# Algorithms and Analysis
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