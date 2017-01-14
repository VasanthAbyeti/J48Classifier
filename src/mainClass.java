
public class mainClass {

	public static void main(String args[]) throws Exception
	{
				double minErr = 100;
				double err;
				j48Classifier jClass = new j48Classifier(args[0], args[1]);
				
				int p = 4;
				int obj,fold;
				int minObj,minFold;
				minObj = minFold = 0;
				
				System.out.println();
				System.out.println("*******************************************************************************");
				System.out.println();
				
				
				while(p < args.length)
				{
					String par[] = args[p].split(",");
					obj = Integer.parseInt(par[0]);
					fold = Integer.parseInt(par[1]);
					
					err = jClass.classify(obj, fold);
					
					if(minErr > err)
					{
						minErr = err;
						minObj = obj;
						minFold = fold;
					}
					p++;
					System.out.println("Avg. Error of 10folds CV J48 of minNumOfObjs : " + obj + " numOfFolds : " + fold + " is :"  + err);
				}
				System.out.println();
				System.out.println("*******************************************************************************");
				System.out.println();
				naiveBayesClassifier nBc = new naiveBayesClassifier(args[0], args[1]);
				double nBcError = nBc.classify();
				System.out.println("Avg. Error of 10folds CV NaiveBayesSimple : " + nBcError); 
				
				if(minErr > nBcError)
				{
					minErr = nBcError;
					minObj = -1;
					minFold = -1;
					
					System.out.println();
					System.out.println("*******************************************************************************");
					System.out.println();
					System.out.println("Best Model was NaiveBayesSimple with Error : " + minErr);
					System.out.println();
					System.out.println("*******************************************************************************");
					System.out.println();
					jClass.classifyTestData(minObj, minFold);
					
				}
				else
				{
					System.out.println();
					System.out.println("*******************************************************************************");
					System.out.println();
					System.out.println("Best Model's  Settings are; minObjs : " + minObj + " folds : " + minFold + " Error : " + minErr);
					System.out.println();
					System.out.println("*******************************************************************************");
					System.out.println();
					jClass.classifyTestData(minObj, minFold);
				}
	}	
}