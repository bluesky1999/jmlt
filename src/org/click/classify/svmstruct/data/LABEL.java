package org.click.classify.svmstruct.data;

/**
 * this defines the y-part (the label) of a training example, e.g. the parse
 * tree of the corresponding sentence.
 * 
 * @author lq
 *
 */
public class LABEL {

	public int class_index;
	public int num_classes;
	public double[] scores;

	//label的类型，一层、二层等
	public double[] class_indexs;
	public double dou_index;

	public int totdoc = 0;

	public String toString() {
		return class_index + "";
	}

}
