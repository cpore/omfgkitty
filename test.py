import pyspark
import socket
from time import sleep
import sys
import cv2


def print_hostname(LOGGER):
    for i in range(30):
        print("-----------------------------------test info---------------------------------")
        print('iteration:' + str(i))
        print('host:' + socket.gethostname())
        print('interpreter version:' + sys.version)
        print('openCV version:' + cv2.__version__)
        sleep(10)
        
if __name__ == '__main__':
    sc = pyspark.SparkContext()
    sc.setLogLevel('DEBUG')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger('py4j')
    LOGGER.info("pyspark script logger initialized")
    print_hostname(LOGGER)
    
    
#$SPARK_HOME/bin/spark-submit --py-files test.py --deploy-mode cluster --master yarn --supervise 