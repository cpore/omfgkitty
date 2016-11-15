#! /bin/bash

$SPARK_HOME/bin/spark-submit --py-files ~/workspace/omfgkitty/test.py --deploy-mode cluster --master yarn --supervise
