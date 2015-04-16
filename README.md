jmlt
====

java machine learning toolkit

A java implementation of svm struct

refer codes:<br>
  http://www.cs.cornell.edu/People/tj/svm_light/svm_struct.html

refer papers:<br>
   Support Vector Machine Learning for Interdependent and Structured Output Spaces . Ioannis Tsochantaridis , Thomas Hofmann et. al. 2004 <br>
   Making Large-Scale SVM Learning Practical. Thorsten Joachims. 1998 <br>
   
Usage:
   build a jar file: ant <br>
   train: java  -Xmx3000m -cp jmlt.jar org.jmlp.classify.svm_struct.source.svm_struct_main -c 5000 example/train.txt example/model<br>
   test:   java  -Xmx1000m -cp jmlt.jar org.jmlp.classify.svm_struct.source.svm_struct_classify example/test.txt example/model example/prediction<br>

jdk 1.70以上   
