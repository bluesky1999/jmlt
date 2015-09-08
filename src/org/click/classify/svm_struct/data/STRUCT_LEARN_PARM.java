package org.click.classify.svm_struct.data;

public class STRUCT_LEARN_PARM {

	/**
	 * precision for which to solve quadratic program
	 */
	public double epsilon;

	/**
	 * number of new constraints to accumulate before recomputing the QP
	 * solution
	 */
	public double newconstretrain;

	/**
	 * maximum number of constraints to cache for each example (used in w=4
	 * algorithm)
	 */
	public int ccache_size;

	/**
	 * size of the mini batches in percent of training set size (used in w=4
	 * algorithm)
	 */
	public double batch_size;

	/**
	 * trade-off between margin and loss
	 */
	public double C;

	public String[] custom_argv;

	public int custom_argc;

	public int slack_norm;;

	public int loss_type;;

	public int loss_function;

	public int num_classes;

	public int num_features;

	// should test example vectors be
	// truncated to the number of features
	// seen in the training data?
	public int truncate_fvec;
	public double bias; // value for artificial bias feature
	public int bias_featurenum; // id number of bias feature
	public double prec_rec_k_frac; // fraction of training set size to use as value of
							// k for Prec@k and Rec@k
	public int sparse_kernel_type; // Same as -t option
	public int sparse_kernel_size; // Number of basis functions to select from the set
								// of basis functions for training with
								// approximate kernel expansion.
	String sparse_kernel_file;// File that contains set of basis functions for
								// training with approximate kernel expansion.
	public  int sparse_kernel_method; // method for selecting sparse kerne subspace (1
								// random sampling, 2 incomplete cholesky)
	public int shrinking; // Selects whether shrinking heuristic is used in the custom
					// algorithm for minimizing error rate.
	public double rset_precision; // (only used internally) minimum improvement in
							// euclidian distance so that the preimage is added
							// to the reduced set expansion
	public int preimage_method; // method to use for finding preimages
	public int recompute_rset; // selects whether the reduced set is recomputed and the
						// method restarted again, after it has converged for
						// the first time
	public int classify_dense; // uses a dense vector representation when classifying
						// new examples in svm_perf_classify. This uses more
						// memory, but is faster if the support vectors in the
						// model are dense.

}
