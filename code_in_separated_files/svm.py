from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import ast
import time
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import Vectors as mllibVectors
from pyspark.ml.feature import PCA
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import happybase

appName = 'SVM'
master = 'local[4]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
spark = SparkSession \
        .builder \
        .appName("SVM") \
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
    for i in range(1, k):
        if 'f:'+str(i) in line:
            indexList.append(i)
            valList.append(line['f:'+str(i)])
    for i in range(1, 380):
        if 'l:'+str(i) in line:
            label = int(line['l:'+str(i)])
            if label == -1:
                label = 0
    return (Vectors.sparse(k, indexList, valList),label)

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
        for i in range(1, k):
            if 'f:'+str(i) in line:
                indexList.append(i)
                valList.append(line['f:'+str(i)])
        label = int(line['l:'+str(col)])
        if label == -1:
            label = 0
        features.append((Vectors.sparse(k, indexList, valList),label))
    features = sc.parallelize(features)
    #sclines = sc.parallelize(lines)
    #features = sclines.map(featuresToSparseVecFromLine)
    featureDataFrame = spark.createDataFrame(features, ["features", "label"])
    pca = PCA(k=100, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(featureDataFrame)
    #pcaresult = model.transform(featureDataFrame).select("pcaFeatures").collect()
    #lp = []
    #c = 0
    #for com in pcaresult:
    #    lp.append(LabeledPoint(lines[c]['l:' + str(col)], mllibVectors.fromML(com.pcaFeatures)))
    #    c += 1
    #lp = sc.parallelize(lp)
    pcaresult = model.transform(featureDataFrame).rdd
    lp = pcaresult.map(lambda r: LabeledPoint(r.label, mllibVectors.fromML(r.pcaFeatures)))
    model = SVMWithSGD.train(lp)
    model.save(sc, "svm/SVM" + str(col))
    labelsAndPreds = lp.map(lambda p: (p.label, model.predict(p.features)))
    err = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
    print("err at node " + str(col) + " = " + str(err))

sc.stop()
