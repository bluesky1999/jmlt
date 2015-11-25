package org.click.lib.math;

public class Normalize {

	
	public static void normalize(double[] arr)
	{
		double sum=0.0;
		
		for(int i=0;i<arr.length;i++)
		{
			sum+=Math.abs(arr[i]);
		}
		
		sum+=1;
		
		for(int i=0;i<arr.length;i++)
		{
			arr[i]/=sum;
		}
		
	}
	

}
