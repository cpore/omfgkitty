##OMFGKITTY!

It detects cats in images.

[http://research.microsoft.com/pubs/80582/ECCV_CAT_PROC.pdf](http://research.microsoft.com/pubs/80582/ECCV_CAT_PROC.pdf)

[https://users.soe.ucsc.edu/~rcsumner/papers/RossSumner.pdf](https://users.soe.ucsc.edu/~rcsumner/papers/RossSumner.pdf)

[http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

[http://www.cs.colostate.edu/~draper/papers/draper_ssspr02.pdf](http://www.cs.colostate.edu/~draper/papers/draper_ssspr02.pdf)

[https://github.com/harthur/kittydar](https://github.com/harthur/kittydar)

[http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)

[http://scikit-image.org/docs/dev/auto_examples/plot_hog.html](http://scikit-image.org/docs/dev/auto_examples/plot_hog.html)

[http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/](http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)

[http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)

[https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.vhc7kab6x](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.vhc7kab6x)

[https://github.com/mobolic/facebook-sdk](https://github.com/mobolic/facebook-sdk)

[PySpark Carpentry: How to Launch a PySpark Job with Yarn-cluster](http://tech.magnetic.com/2016/03/pyspark-carpentry-how-to-launch-a-pyspark-job-with-yarn-cluster.html)

[Spark Internals](https://cwiki.apache.org/confluence/display/SPARK/Spark+Internals?src=breadcrumbs-parent)

#Environment Setup

###Install Anaconda Python 3

`wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh`

###Install OpenCV3

`$ conda install -c menpo opencv3=3.1.0`

`$ conda install -c asmeurer pango`

### Setup spark-env.sh

Add `export PYSPARK_PYTHON=python3` to use Python 3

###Make sure to add cv2 to "Forced builtins" for Python code completion in Eclipse:

[http://stackoverflow.com/questions/9085643/how-to-use-code-completion-into-eclipse-with-opencv](http://stackoverflow.com/questions/9085643/how-to-use-code-completion-into-eclipse-with-opencv)