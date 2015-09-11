package org.click.classify.svm_struct.data;

public class STRUCT_ID_SCORE implements Comparable {
	public int id;
	public double score;
	public double tiebreak;

	public STRUCT_ID_SCORE()
	{
		
	}
	
	public STRUCT_ID_SCORE(int id, double score, double tiebreak) {
		this.id = id;
		this.score = score;
		this.tiebreak = tiebreak;
	}

	@Override
	public int compareTo(Object o) {

		STRUCT_ID_SCORE s = (STRUCT_ID_SCORE) o;
		double va, vb;
		va = s.score;
		vb = this.score;
		if (va == vb) {
			va = s.tiebreak;
			vb = this.tiebreak;
		}

		int m = 0, l = 0;

		if (va > vb) {
			m = 1;
		}

		if (va < vb) {
			l = 1;
		}
		return m - l;
	}
	
	@Override
	public String toString()
	{	
		return "id:"+this.id+" score:"+this.score+" tiebreak:"+this.tiebreak;
	}
}
