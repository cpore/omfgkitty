#!/bin/bash

STARTYARN='$HADOOP_HOME/sbin/start-yarn.sh'

ssh cpore@denver.cs.colostate.edu -t "$STARTYARN;"
