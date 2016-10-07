#!/bin/bash

DFS='$HADOOP_HOME/bin/hdfs dfs'
STARTDFS='$HADOOP_HOME/sbin/start-dfs.sh'

#set up directories and copy data
ssh cpore@columbus-oh.cs.colostate.edu -t "$DFS -mkdir /usr; \
              $DFS -mkdir /usr/cpore; \
              $DFS -mkdir /usr/cpore/data; \
              $DFS -mkdir /output; \
              $DFS -put ~/workspace/cs535a1/data/* /usr/cpore/data; \
              exit"

