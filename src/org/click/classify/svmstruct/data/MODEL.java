package org.click.classify.svmstruct.data;



public class MODEL {
	  public int sv_num;
	  public int at_upper_bound;
	  public double b;
	  public DOC[] supvec;
	  public double[] alpha;
	  
	  //index from docnum to position in model 
	  public int[] index;
	  
	  //number of features
	  public int totwords;
	  
	  //number of training documents
	  public int totdoc;
	  
	  public KERNEL_PARM kernel_parm;
	  
	  //weights for linear case using folding
	  public double[] lin_weights;
	  
	  //precision, up to which this  model is accurate
	  public double maxdiff;
	
}
