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
	  
	  public KERNEL_PARM copyKERNEL_PARM()
	  {
		  KERNEL_PARM nkp=new KERNEL_PARM();
		  nkp.kernel_type=kernel_type;
		  nkp.poly_degree=poly_degree;
		  nkp.rbf_gamma=rbf_gamma;
		  nkp.coef_lin=coef_lin;
		  nkp.coef_const=coef_const;
		  nkp.custom=custom;
		  
		  return nkp;
	  }
	  
}
