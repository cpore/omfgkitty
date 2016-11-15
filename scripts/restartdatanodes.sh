#!/bin/bash

#delete data from datanodes
for i in $(cat ~/hadoopConf/slaves); do
  echo $i
  #login to remote machine
  ssh $i -t "$HADOOP_HOME/sbin/yarn-daemon.sh stop nodemanager; $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode; $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode; $HADOOP_HOME/sbin/yarn-daemon.sh start nodemanager; exit"
done

