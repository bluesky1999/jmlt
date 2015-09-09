package org.click.classify.svm_struct.data;

/**
 * 
 * @author lq
 *
 */
public class STRUCTMODEL {

	public double[] w;
	public MODEL svm_model;
	public int sizePsi;
	public double walpha;

	/*
	 * other information that is needed for the stuctural model can be added
	 * here, e.g. the grammar rules for NLP parsing
	 */
	public int sparse_kernel_type; /*
							 * Selects the kernel type for sparse kernel
							 * approximation. The values are the same as for the
							 * -t option. LINEAR (i.e. 0) means that sparse
							 * kernel approximation is not used.
							 */
	public int expansion_size; /*
						 * Number of vectors in sparse kernel expansion
						 */
	public DOC[] expansion; /* Vectors in sparse kernel expansion */
	public long reducedset_size; /*
						 * Number of vectors in reduced set expansion
						 */
	public SVECTOR[] reducedset; /* Vectors in reduced set expansion */
	public double[] reducedset_kernel; /*
								 * Kernel values between reduced set expansion
								 * and training examples
								 */
	public MATRIX reducedset_gram; /* Gram matrix of reduced set expansion */
	public MATRIX reducedset_cholgram; /*
								 * Cholesky decomposition of Gram matrix of
								 * reduced set expansion
								 */
	public MATRIX invL; /*
				 * Inverse of Cholesky decomposition of Gram matrix over the
				 * vectors in the sparse kernel expansion
				 */
}
