#!/bin/bash

#delete data from datanodes
for i in $(cat ~/hadoopConf/slaves); do
  echo $i
  #login to remote machine
  ssh $i -t "rm -r /tmp/hadoop-cpore/; rm -r /tmp/hadoop-cpore-datanode.pid; rm -r /s/$i/a/tmp/hadoop-cpore/; exit"
done

