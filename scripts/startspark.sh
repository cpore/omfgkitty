#!/bin/bash

STARTSPARK='$SPARK_HOME/sbin/start-all.sh'

ssh cpore@columbus-oh.cs.colostate.edu -t "$STARTSPARK;"
