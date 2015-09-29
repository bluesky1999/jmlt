package org.click.classify.svmstruct.data;

public class LEARN_PARM {
	 /**
	  * 选择是回归还是 分类
	  */
	 public short type;
	 
	 /**
	  * alphas的上界C
	  */
	 public double svm_c;
	 
	 /**
	  * 回归 epsilon (对于分类是1)
	  */
	 public double eps;
	 
	 /**
	  * C对于正样本的乘因子
	  */
	 public double svm_costratio;
	
	 /**
	  * 如果非0，则使用超平面 w*x+b=0,否则为w*x=0
	  */
	 public short biased_hyperplane;
	 
	 
	 /**
	  * 如果非0，则使用shared slack variable,要求每个训练样本都设置slackid
	  */
	 public short sharedslack;
	 
	 /**
	  * working set 的大小
	  */
	 public int svm_maxqpsize;
	 
	 
	 public int svm_newvarsinqp;
	 
	 
	 // tolerable error for distances used 
	 // in stopping criterion 
	 public double epsilon_crit;  
	 
	// how much a multiplier should be above 
	 // zero for shrinking 
	 public double epsilon_shrink;     
    
     // number of iterations after which the
	 // optimizer terminates, if there was
	 // no progress in maxdiff 
	 public long   maxiter;        
	 
	 // exclude examples with alpha at C and  retrain 
	 public long   remove_inconsistent;  
     
	 
	 // file for predicitions on unlabeled examples
	 // in transduction 
	 public  String predfile;         
	 
	 // file to store optimal alphas in. use  empty string if alphas should not  output 
	 public   String alphafile;        
       

	  // tolerable error on alphas at bounds 
	 public double epsilon_a;          
    
   
	// individual upper bounds for each var
	 public  double[] svm_cost;           
    
	 // number of features 
	 public  int   totwords;            
}
