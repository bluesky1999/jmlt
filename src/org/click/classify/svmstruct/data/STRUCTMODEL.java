package org.click.classify.svmstruct.data;

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

	// other information that is needed for the stuctural model can be added
	// here, e.g. the grammar rules for NLP parsing

	// Selects the kernel type for sparse kernel
	// approximation. The values are the same as for the
	// -t option. LINEAR (i.e. 0) means that sparse
	// kernel approximation is not used.
	public int sparse_kernel_type;

	// Number of vectors in sparse kernel expansion
	public int expansion_size;

	// Vectors in sparse kernel expansion 
	public DOC[] expansion;

	// Number of vectors in reduced set expansion
	public long reducedset_size;

	// Vectors in reduced set expansion
	public SVECTOR[] reducedset;

	// Kernel values between reduced set expansion
	// and training examples
	public double[] reducedset_kernel;

	// Gram matrix of reduced set expansion 
	public MATRIX reducedset_gram;

	// Cholesky decomposition of Gram matrix of
	// reduced set expansion
	public MATRIX reducedset_cholgram;

	// Inverse of Cholesky decomposition of Gram matrix over the
	// vectors in the sparse kernel expansion
	public MATRIX invL;
}
