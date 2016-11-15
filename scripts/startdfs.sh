#!/bin/bash

STARTDFS='$HADOOP_HOME/sbin/start-dfs.sh'

ssh cpore@columbus-oh.cs.colostate.edu -t "$STARTDFS;"
