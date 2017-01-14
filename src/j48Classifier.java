import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.Evaluation;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class j48Classifier {

	Instances train;
	Instances test;
	J48 j48;
	
	j48Classifier(String trainPath, String testPath) throws IOException
	{
		ArffLoader aLoad = new ArffLoader();			 
		 aLoad.setFile(new File(trainPath));
		 train = aLoad.getDataSet();
		
		 ArffLoader aLoad_Test = new ArffLoader();			 
		 aLoad_Test.setFile(new File(testPath));
		 test = aLoad_Test.getDataSet();
		 
		 train.setClassIndex(0);
		 test.setClassIndex(0);
		 
		 j48 = new J48();
		 j48.setUnpruned(false);
		 j48.setReducedErrorPruning(true);
	}
	
	public Double classify(int objs,int folds) throws Exception
	{
		 
		 j48.setMinNumObj(objs);
		 j48.setNumFolds(folds);
		 
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setClassifier(j48);
		 
		 Evaluation eval = new Evaluation(train);
		 eval.crossValidateModel(fc, train, 10, new Random(1));
		 fc.buildClassifier(train);

		 return (eval.pctIncorrect());
	}
	
	public void classifyTestData(int objs,int folds) throws Exception
	{
		 j48.setMinNumObj(objs);
		 j48.setNumFolds(folds);
		 
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setClassifier(j48);
		 
		 Evaluation eval = new Evaluation(train);
		 eval.crossValidateModel(fc, train, 10, new Random(1));
		 fc.buildClassifier(train);
		 
		 System.out.println("Predicting on Test DataSet using the best model : ");
		 for (int i = 0; i < test.numInstances(); i++) {
			   double pred = fc.classifyInstance(test.instance(i));
			   System.out.print("Instance No.: " + i);
			   System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
			   System.out.println(", predicted: " + test.classAttribute().value((int) pred));
			 }
		 System.out.println();
			System.out.println("*******************************************************************************");
		System.out.println();
		 eval.evaluateModel(fc,test);
		 System.out.println("ErrorS : " + eval.pctIncorrect()); 
	}
}
