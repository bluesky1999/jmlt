package  org.click.lib.math;

/**
 * 简单的数学函数
 * @author lq
 *
 */
public class SimFunc<T extends Comparable<T>> {

	public T max(T  a, T  b)
	{
	  if(a.compareTo(b)>0)
	    return(a);
	  else
	    return(b);
	}
	
	
	public static double entropy(double x)
	{
		double y=0;
		y=-x*Math.log(x)-(1-x)*Math.log(1-x);
		return y;
	}
	
	public static void main(String[] args)
	{
		SimFunc<Double> sf=new SimFunc<Double>();
		
		double a=1.0,b=15.0;
		
		System.err.println("max:"+sf.max(a, b));
		
	}
	
	
	
}
