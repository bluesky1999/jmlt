package org.click.bug;

import java.util.ArrayList;
import java.util.Collections;

import org.click.classify.svmstruct.data.STRUCT_ID_SCORE;

public class T {

	public static void main(String[] args)
	{
		ArrayList<STRUCT_ID_SCORE> l=new ArrayList<STRUCT_ID_SCORE>();
		l.add(new STRUCT_ID_SCORE(1,0.3,0));
		l.add(new STRUCT_ID_SCORE(2,0.1,0));
		l.add(new STRUCT_ID_SCORE(3,0.7,0));
		l.add(new STRUCT_ID_SCORE(4,0.4,0));
		l.add(new STRUCT_ID_SCORE(5,0.2,0));
		
		for(int i=0;i<l.size();i++)
		{
			System.err.println("i:"+l.get(i));
		}
		
		Collections.sort(l);
		
		System.err.println("after sort");
		
		for(int i=0;i<l.size();i++)
		{
			System.err.println("i:"+l.get(i));
		}
	}
}
