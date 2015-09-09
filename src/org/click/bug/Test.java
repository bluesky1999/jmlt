package org.click.bug;

public class Test {

	public void init(double[] a)
	{
		a=new double[10];
		for(int i=0;i<a.length;i++)
		{
			a[i]=i;
		}
	}
	
	public void test()
	{
		TT t=new TT();
		init(t.a);
		
		System.out.println("a.len:"+t.a.length);
	}
	
	public static void main(String[] args)
	{
		Test t=new Test();
		t.test();
	}
	
}
