import pyspark
import socket
from time import sleep
import sys
import cv2


def print_hostname(LOGGER):
    for i in range(30):
        LOGGER.info("-----------------------------------test info---------------------------------")
        LOGGER.info('iteration:' + str(i))
        LOGGER.info('host:' + socket.gethostname())
        LOGGER.info('interpreter version:' + sys.version)
        LOGGER.info('openCV version:' + cv2.__version__)
        sleep(10)
        
if __name__ == '__main__':
    sc = pyspark.SparkContext()
    #sc.setLogLevel('WARN')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")
    print_hostname(LOGGER)
    
    
#$SPARK_HOME/bin/spark-submit --py-files test.py --deploy-mode cluster --master yarn --supervise 