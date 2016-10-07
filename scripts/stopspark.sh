#!/bin/bash

STOPSPARK='$SPARK_HOME/sbin/stop-all.sh'

ssh cpore@columbus-oh.cs.colostate.edu -t "$STOPSPARK;"
