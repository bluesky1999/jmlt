package org.click.classify.svmstruct.data;

/** 
  The following specifies a quadratic problem of the following form <br>
  <br>
  minimize   g0 * x + 1/2 x' * G * x <br>
  &nbsp;&nbsp; subject to ce*x - ce0 = 0 <br>
  &nbsp; &nbsp;&nbsp;     l <= x <= u <br>
*/
public class QP {
	
	 public int opt_n;
	 public int opt_m;
	 public double[] opt_ce;
	 public double[] opt_ce0;
	 	 
	 public double[] opt_g;
	 public double[] opt_g0;
	 public double[] opt_xinit;
	 public double[] opt_low,opt_up;
	 

	 public String toString2()
	 {
		 String str="";
		 
		  str+=("opt_n:"+opt_n+"\n");
		  str+=("opt_m:"+opt_m+"\n");
		  
		  str+=("opt_ce.len:"+opt_ce.length+"\n");
		  for(int i=0;i<opt_ce.length;i++)
		  {
			  str+=(opt_ce[i]+" ");	
		  }		  
		  str+="\n";
		 
		  str+="opt_ce0.len:"+opt_ce0.length+"\n";  
		  for(int i=0;i<opt_ce0.length;i++)
		  {
			  str+=(opt_ce0[i]+" ");		  
		  }		  
		  str+="\n";  
		  
		  str+=("opt_g.len:"+opt_g.length+"\n");  
		  for(int i=0;i<opt_g.length;i++)
		  {
			  if(opt_n>0&&i%opt_n==0)
			  {
				  str+=("\n");
			  }
			  str+=(opt_g[i]+" ");		  
		  }		  
		  str+="\n";    
		  
		  str+=("opt_g0.len:"+opt_g0.length+"\n");  
		  for(int i=0;i<opt_g0.length;i++)
		  {
			  str+=(opt_g0[i]+" ");		  
		  }		  
		  str+="\n"; 
		  
		  str+=("opt_xinit.len:"+opt_xinit.length+"\n");  
		  for(int i=0;i<opt_xinit.length;i++)
		  {
			  str+=(opt_xinit[i]+" ");		  
		  }		  
		  str+="\n";
		  
		  str+=("opt_low.len:"+opt_low.length+"\n");  
		  for(int i=0;i<opt_low.length;i++)
		  {
			  str+=(opt_low[i]+" ");		  
		  }		  
		  str+="\n";
		  
		  str+=("opt_up.len:"+opt_up.length+"\n");  
		  for(int i=0;i<opt_up.length;i++)
		  {
			  str+=(opt_up[i]+" ");		  
		  }		  
		  str+="\n";
		  
		 return str;
	 }
}
