import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.classifiers.Evaluation;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.bayes.*;
import weka.core.converters.ArffLoader;

public class naiveBayesClassifier {
	
	Instances train;
	Instances test;
	NaiveBayesSimple nbs;
	
	naiveBayesClassifier(String trainPath, String testPath) throws IOException
	{
		ArffLoader aLoad = new ArffLoader();			 
		 aLoad.setFile(new File(trainPath));
		 train = aLoad.getDataSet();
		
		 ArffLoader aLoad_Test = new ArffLoader();			 
		 aLoad_Test.setFile(new File(testPath));
		 test = aLoad_Test.getDataSet();
		 
		 train.setClassIndex(0);
		 test.setClassIndex(0);
		 
		 nbs = new NaiveBayesSimple();
	}
	
	public Double classify() throws Exception
	{
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setClassifier(nbs);
		 
		 Evaluation eval = new Evaluation(train);
		 eval.crossValidateModel(fc, train, 10, new Random(1));
		 fc.buildClassifier(train);
		 return (eval.pctIncorrect());
	}
	
	public void classifyTestData() throws Exception
	{
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setClassifier(nbs);
		 
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
