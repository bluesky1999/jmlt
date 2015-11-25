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
	  
	  public void init(int totdoc,int totwords,KERNEL_PARM kernel_parm)
	  {
			this.supvec = new DOC[totdoc + 1];
			this.alpha = new double[totdoc + 1];
			this.index = new int[totdoc + 1];
			this.at_upper_bound = 0;
			this.b = 0;
			this.supvec[0] = null;
			this.alpha[0] = 0;
			this.lin_weights = null;
			this.totwords = totwords;
			this.totdoc = totdoc;
			this.kernel_parm = kernel_parm;
			this.sv_num = 1;
	  }
	
}
