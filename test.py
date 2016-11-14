import pyspark
import socket
from time import sleep


def print_hostname(LOGGER):
    i = 0
    while True:
        LOGGER.info('iteration:', i, socket.gethostname())
        sleep(10)
        i = i + 1
        
if __name__ == '__main__':
    sc = pyspark.SparkContext()
    #sc.setLogLevel('WARN')
    
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")