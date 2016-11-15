#!/bin/bash

DFS='$HADOOP_HOME/bin/hdfs dfs'

#delete output
ssh cpore@columbus-oh.cs.colostate.edu -t "$DFS -rm -r /output*; \
              exit"

