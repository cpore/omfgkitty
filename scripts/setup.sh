#!/bin/bash

DFS='$HADOOP_HOME/bin/hdfs dfs'
STARTDFS='$HADOOP_HOME/sbin/start-dfs.sh'

#set up directories and copy data
ssh cpore@columbus-oh.cs.colostate.edu -t "$DFS -mkdir /CAT_DATASET; \
              $DFS -mkdir /output; \
              $DFS -put ~/workspace/omfgkitty/CAT_DATASET/* /CAT_DATASET; \
              exit"

