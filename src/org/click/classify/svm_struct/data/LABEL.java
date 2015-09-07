package org.click.classify.svm_struct.data;
/**
 * this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence.
 * @author lq
 *
 */
public class LABEL {

	public int class_index;
	public int num_classes;
	public double[] scores;
	
	public int[] extra_indexs;
	public int[] extra_classes;
	public int num_level=1;
	
	//label的类型，一层、二层等
	public int label_type=0;
	public String toString()
	{
		if(label_type==0)
		{
		   return class_index+"";
		}
		else if(label_type==1)
		{
			return "";
		}
		return "";
	}
	
	
}
