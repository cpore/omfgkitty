import pyspark
import sys
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pydoop import hdfs


if __name__ == '__main__':
    conf = SparkConf().setAppName('jpeg_data')
    sc = SparkContext(conf=conf)
    imgs = sc.binaryFiles("hdfs://columbus-oh.cs.colostate.edu:30148/CAT_DATASET")
    
    fs = hdfs.hdfs('columbus-oh.cs.colostate.edu', 30148)
    
    if fs.exists('/output/imgrdd'):
        fs.delete('/output/imgrdd', True)
        
    imgs.keys().saveAsTextFile("hdfs://columbus-oh.cs.colostate.edu:30148/output/imgrdd")






#$SPARK_HOME/bin/spark-submit --py-files test.py --deploy-mode cluster --master yarn --supervise 
    
    