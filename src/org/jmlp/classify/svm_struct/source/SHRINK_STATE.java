package org.jmlp.classify.svm_struct.source;

public class SHRINK_STATE {
	  int[] active;
	  int[] inactive_since;
	  int deactnum;
	  double[][] a_history;
	  int maxhistory;
	  double[] last_a;
	  double[] last_lin;
}
