package org.click.classify.svmstruct.data;

public class KERNEL_PARM {

	//0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom, 5=matrix
	public short kernel_type;

	public int poly_degree;
	public double rbf_gamma;
	public double coef_lin;
	public double coef_const;
	public String custom;

	//here one can directly supply the kernel matrix. The matrix is accessed if kernel_type=5 is selected.
	public MATRIX gram_matrix;

	public KERNEL_PARM() {
		this.kernel_type = 0;
		this.poly_degree = 3;
		this.rbf_gamma = 1.0;
		this.coef_lin = 1;
		this.coef_const = 1;
		this.custom = "empty";
	}

}
