from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.conf import SparkConf

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

def toCSVLine(data):
  return ','.join(str(d) for d in data)

def train():
    data = sc.textFile("hdfs://columbus-oh.cs.colostate.edu:30148/data/hog.data")
    parsedData = data.map(parsePoint)
    
    trainingRDD, validationRDD, testRDD = parsedData.randomSplit([6, 2, 2], seed=0)
    
    # Build the model
    model = SVMWithSGD.train(parsedData, iterations=100, regType=None)
    
    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("-----------------------------------------Training Error = " + str(trainErr) + "----------------------------------------------")
    
    # Save and load model
    #model.save(sc, "hdfs://columbus-oh.cs.colostate.edu:30148/model/model")
    #this doesn't work
    #model.toPMML(sc, "hdfs://columbus-oh.cs.colostate.edu:30148/pmml/model.xml")
    #sameModel = SVMModel.load(sc, "hdfs://columbus-oh.cs.colostate.edu:30148/model/model")
    #array = model.weights().values()
    
    print("intercept: ", model.intercept)
    print("weights: ", model.weights.values.shape, model.weights.values)
    w = numpy.append(model.weights.values, model.intercept)
    weightsRDD = sc.parallelize(w)
    weights.saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/model/weights.data")
    
    
if __name__ == '__main__':
    conf = SparkConf().setAppName('SVM_TRAINING')
    sc = SparkContext(conf=conf)
    #sc.setLogLevel('WARN')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    train()
    sc.stop()
    
#$SPARK_HOME/bin/spark-submit --deploy-mode cluster --master yarn --supervise pyspark_train.py