#!/bin/bash

STOPYARN='$HADOOP_HOME/sbin/stop-yarn.sh'

ssh cpore@denver.cs.colostate.edu -t "$STOPYARN;"
