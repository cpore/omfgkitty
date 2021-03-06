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

def train():
    
    dataFileTex = "hdfs://columbus-oh.cs.colostate.edu:30148/data/hog_tex.data"
    iterations = 0
    finalParams = []
        
    tag = "_tex"
    
    data = sc.textFile(dataFileTex)
    parsedData = data.map(parsePoint)

    params = make_params()
    
    bestParamSet = None
    bestTrainError = 2.0
    bestTestError = 2.0
    bestModel = None
    
    for parms in params:
        trainValidateRDD, testingRDD = parsedData.randomSplit([0.9, 0.1])
        validationErrors = []
        bestParmSet = None
        for fold in range(10):
            iterations += 1
            
            trainingRDD, validationRDD = trainValidateRDD.randomSplit([0.8, 0.2])               
            
            # Build the model
            model = SVMWithSGD.train(trainingRDD, parms[0], parms[1], parms[2],parms[3],parms[4],parms[5],parms[6],parms[7],parms[8])
            # Evaluating the model on training data
            predsAndLabels = validationRDD.map(lambda p: (model.predict(p.features), p.label))
            validationError = predsAndLabels.filter(lambda lp: lp[0] != lp[1]).count() / float(validationRDD.count())
            
            print('Finished Running iteration:', str(iterations), 'of', len(params) * 10, 'Running fold', str(fold), '\nwith params:', parms, '\nwith error:', validationError)
            validationErrors.append(validationError)      
            
        # Calculate the mean of these errors.
        meanError = np.mean(np.array(validationErrors))
        
        # If this error is less than the previously best error (plus a 10% delta for wiggle room)
        # for parmSet, update best parameter values and best error
        if meanError < bestTrainError + 0.01:
            bestTrainError = meanError
            bestParmSet = parms
                
            #Train a new model with the best params and check it against the test hold-out set
            model = SVMWithSGD.train(trainValidateRDD,bestParmSet[0], bestParmSet[1], bestParmSet[2],bestParmSet[3],bestParmSet[4],bestParmSet[5],bestParmSet[6],bestParmSet[7],bestParmSet[8])
            predsAndLabels = testingRDD.map(lambda p: (model.predict(p.features), p.label))
            trainError = predsAndLabels.filter(lambda lp: lp[0] != lp[1]).count() / float(testingRDD.count())
            #If its better than the best training error then these params make the best general model
            if trainError < bestTestError:
                print("New Best Training Error = " + str(trainError) + "----------------------------------------------")
                print('New Best Params:', bestParmSet)
                bestTestError = trainError
                bestModel = model
                bestParamSet = bestParmSet
                finalParams.append(bestParamSet)
                
                w = np.append(bestModel.weights.values, bestModel.intercept)
                weightsRDD = sc.parallelize(("w", ','.join(['%.16f' % num for num in w])))
                timestamp = int(time.time())
                weightsRDD.saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/model/weights_" + str(iterations) + "_" + tag + "_" + str(timestamp) + "_" + str(bestTestError) + ".model")
      
    #Finally, train a new model with the best params on the entire data set
    finalModel = SVMWithSGD.train(parsedData,bestParamSet[0], bestParamSet[1], bestParamSet[2],bestParamSet[3],bestParamSet[4],bestParamSet[5],bestParamSet[6],bestParamSet[7],bestParamSet[8])          
    w = np.append(finalModel.weights.values, finalModel.intercept)
    weightsRDD = sc.parallelize(("w", ','.join(['%.16f' % num for num in w])))
    
    timestamp = int(time.time())
    weightsRDD.saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/model/weights_" + str(iterations) + "_" + tag + "_" + str(timestamp) + "_" + str(bestTestError) + "_FINAL" + ".model")
                
    # Instantiate metrics object
    #metrics.append(BinaryClassificationMetrics(predictionAndLabels))
    
#     for i in range(len(metrics)):
#         tag = "_sha" if i==0 else "_tex"
#         # Area under precision-recall curve
#         print(tag, "Area under PR = %s" % metric.areaUnderPR)
#         # Area under ROC curve
#         print(tag, "Area under ROC = %s" % metric.areaUnderROC)
#         print(tag, 'best params:', finalParams[i])
            
def make_params():
    iterations_params = [1000]
    step_params = [0.01, 0.1, 0.5]
    regParam_params = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    miniBatchFraction_params = [1.0]
    initialWeights_params = [None]
    regType_params = ["l2", None]
    #True never does better than random
    intercept_params = [False]
    validateData_params = [True]
    convergenceTol_params = [0.0001]
    listOParams = [iterations_params,step_params,regParam_params,miniBatchFraction_params,initialWeights_params,regType_params,intercept_params,validateData_params,convergenceTol_params]
    
    params = list(itertools.product(*listOParams))
    return params  
    
    
if __name__ == '__main__':
    conf = SparkConf().setAppName('SVM_TRAINING')
    sc = SparkContext(conf=conf)
    #sc.setLogLevel('WARN')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    train()
    sc.stop()
    
#$SPARK_HOME/bin/spark-submit --deploy-mode cluster --master yarn --supervise pyspark_train.py