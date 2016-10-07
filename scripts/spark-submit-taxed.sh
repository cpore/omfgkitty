#! /bin/bash

$SPARK_HOME/bin/spark-submit --class cs535.PageRankTaxed --deploy-mode cluster --master yarn --supervise ~/workspace/cs535a1/build/jar/pageranktaxed.jar
