#!/bin/bash



#delete data from datanodes
for i in $(cat ~/hadoopConf/slaves); do
  echo $i
  #login to remote machine
  ssh $i -t "rm -r /tmp/hadoop-cpore/cpore/dfs/data/; exit"
done

DFS='$HADOOP_HOME/bin/hdfs dfs'
STARTDFS='$HADOOP_HOME/sbin/start-dfs.sh'

#reformat namenode and set up directories
ssh cpore@columbus-oh.cs.colostate.edu -t "$HADOOP_HOME/bin/hdfs namenode -format; \
              $STARTDFS; \
              exit"

