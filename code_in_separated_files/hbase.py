import csv
from collections import deque
import os
import happybase

""" clean sample data, training data and test data"""
dataFilePath = ['./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-sample-v1_0.txt',
                './ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-train-v1_0']#,
                #'./ydata-ytargeting-user-modeling-v1_0/ydata-ytargeting-test-v1_0']

connection = happybase.Connection('localhost', autoconnect=False)
connection.open()

tableName = ['sample', 'train', 'test']
for i in range(1,len(dataFilePath)):
    table = connection.table(tableName[i])
    with open(dataFilePath[i], 'r') as dataFile:
        lineCounter = 0
        with table.batch(batch_size=1000, timestamp=1) as b:
            for line in dataFile:
                lineCounter += 1
                if lineCounter % 1000 == 0:
                    print(lineCounter)
                argDict = {}
                featuresAndLabels = line.partition('\t')
                pairs = featuresAndLabels[0].split(' ')
                for aPair in pairs:
                    aPair = aPair.split(':')
                    argDict['f:' + aPair[0]] = aPair[1]
                pairs = featuresAndLabels[2].strip().split(' ')
                for aPair in pairs:
                    aPair = aPair.split(':')
                    argDict['l:' + aPair[0]] = aPair[1]
                b.put(str(lineCounter), argDict)
                
connection.close()
exit()
