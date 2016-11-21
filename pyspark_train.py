from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.conf import SparkConf

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

def train():
    data = sc.textFile("hdfs://columbus-oh.cs.colostate.edu:30148/data/hog.data")
    parsedData = data.map(parsePoint)
    
    # Build the model
    model = SVMWithSGD.train(parsedData, iterations=100)
    
    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))
    
    # Save and load model
    model.save(sc, "hdfs://columbus-oh.cs.colostate.edu:30148/model/cat_svm.model")
    #sameModel = SVMModel.load(sc, "hdfs://columbus-oh.cs.colostate.edu:30148/model/model")
    
    
if __name__ == '__main__':
    conf = SparkConf().setAppName('SVM_TRAINING')
    sc = SparkContext(conf=conf)
    #sc.setLogLevel('WARN')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    train()
    sc.stop()
    sc.close()
    
#$SPARK_HOME/bin/spark-submit --deploy-mode cluster --master yarn --supervise pyspark_train.py