from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import numpy as np
import time
import itertools

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

def toCSVLine(data):
  return ','.join(str(d) for d in data)

def train_sha():
    
    dataFileTex = "hdfs://columbus-oh.cs.colostate.edu:30148/data/hog_sha.data"
    data = sc.textFile(dataFileTex)
    
    
    parsedData = data.map(parsePoint)

    iterations = 1000
    step = 0.5
    regParam = 0.1
    miniBatchFraction = 1.0
    initialWeights = None
    regType = None
    #True never does better than random
    intercept = False
    validateData = True
    convergenceTol = 0.0001
    
    #Finally, train a new model with the best params on the entire data set
    finalModel = SVMWithSGD.train(parsedData,iterations,step,regParam, miniBatchFraction,initialWeights,regType,intercept,validateData,convergenceTol)          
    w = np.append(finalModel.weights.values, finalModel.intercept)
    weightsRDD = sc.parallelize(("w", ','.join(['%.16f' % num for num in w])))
    
    timestamp = int(time.time())
    weightsRDD.saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/model/weights_sha" + str(timestamp) + "_FINAL" + ".model")
                
    predsAndLabels = parsedData.map(lambda p: (finalModel.predict(p.features), p.label))
    trainError = predsAndLabels.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("Training Error = " + str(trainError) + "----------------------------------------------")
                    
    # Instantiate metrics object
    #metric = BinaryClassificationMetrics(predsAndLabels)
    # Area under precision-recall curve
    #print("Area under PRC = %s" % metric.areaUnderPR)
    # Area under ROC curve
    #print("Area under ROC = %s" % metric.areaUnderROC)
    
def train_tex():
    
    dataFileTex = "hdfs://columbus-oh.cs.colostate.edu:30148/data/hog_tex.data"
    data = sc.textFile(dataFileTex)
    
    
    parsedData = data.map(parsePoint)

    iterations = 1000
    step = 0.01
    regParam = 0.1
    miniBatchFraction = 1.0
    initialWeights = None
    regType = None
    #True never does better than random
    intercept = False
    validateData = True
    convergenceTol = 0.0001
    
    #Finally, train a new model with the best params on the entire data set
    finalModel = SVMWithSGD.train(parsedData,iterations,step,regParam,miniBatchFraction,initialWeights,regType,intercept,validateData,convergenceTol)          
    w = np.append(finalModel.weights.values, finalModel.intercept)
    weightsRDD = sc.parallelize(("w", ','.join(['%.16f' % num for num in w])))
    
    timestamp = int(time.time())
    weightsRDD.saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/model/weights_tex" + str(timestamp) + "_FINAL" + ".model")
                
    predsAndLabels = parsedData.map(lambda p: (finalModel.predict(p.features), p.label))
    trainError = predsAndLabels.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("Training Error = " + str(trainError) + "----------------------------------------------")
    # Instantiate metrics object
    #metric = BinaryClassificationMetrics(predsAndLabels)
    # Area under precision-recall curve
    #print("Area under PRC = %s" % metric.areaUnderPR)
    # Area under ROC curve
    #print("Area under ROC = %s" % metric.areaUnderROC)
              
    
if __name__ == '__main__':
    conf = SparkConf().setAppName('SVM_TRAINING')
    sc = SparkContext(conf=conf)

    #train_sha()
    train_tex()
    sc.stop()
    
#$SPARK_HOME/bin/spark-submit --deploy-mode cluster --master yarn --supervise pyspark_train.py