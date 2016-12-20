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

