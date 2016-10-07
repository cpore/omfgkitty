#! /bin/bash

$SPARK_HOME/bin/spark-submit --class cs535.PageRank --deploy-mode cluster --master yarn --supervise ~/workspace/cs535a1/build/jar/pagerank.jar
