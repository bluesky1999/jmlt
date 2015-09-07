package org.click.classify.svm_struct.data;

public class CCACHEELEM {
	public SVECTOR fydelta;
	public double rhs;
	public double viol;
	public CCACHEELEM next;
	
	public CCACHEELEM copyCCACHEELEM()
	{
		CCACHEELEM nc=new CCACHEELEM();
		nc.fydelta=fydelta.copySVECTOR();
		nc.rhs=rhs;
		nc.viol=viol;
		if(next!=null)
		{
		 nc.next=next.copyCCACHEELEM();
		}
		else
		{
		 nc.next=null;
		}
		return nc;
	}
	

}
