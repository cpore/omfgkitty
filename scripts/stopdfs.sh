#!/bin/bash

STOPDFS='$HADOOP_HOME/sbin/stop-dfs.sh'

ssh cpore@columbus-oh.cs.colostate.edu -t "$STOPDFS;"
