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
