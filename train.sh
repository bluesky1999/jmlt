java -Xmx1000m -cp out/jmlt.jar org.click.classify.svmstruct.model.MainStruct -c 5000 example/train.txt example/model

java -Xmx1000m -cp out/jmlt.jar org.click.classify.svmstruct.model.ClassifyStruct example/test.txt example/model example/prediction
