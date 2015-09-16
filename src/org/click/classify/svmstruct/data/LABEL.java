package org.click.classify.svmstruct.data;
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
	public double[] class_indexs;
	public double dou_index;
	
	public int label_type=0;
	public int totdoc=0;
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
