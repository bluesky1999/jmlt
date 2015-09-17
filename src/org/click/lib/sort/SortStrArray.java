package org.click.lib.sort;


import java.util.ArrayList;
import java.util.Collections;

import org.click.lib.string.SSO;


public class SortStrArray {

	
	public static class SortElement implements Comparable{
		
	   int key_index=0;
	   String[] fields=null;
	   
	    /**str:string
	     * int:integer
	     * dou:double
	     */
	     String key_type="";
		
		public SortElement(int key_index,String[] fields,String key_type)
		{
		 this.key_index=key_index;
		 this.fields=fields;	 
		 this.key_type=key_type;
		}
			
		public int compareTo(Object o) {
			SortElement s = (SortElement)o;
			if(this.key_type.equals("str"))
			{
			  return (fields[key_index].compareTo(s.fields[key_index])<0) ? 1 : ((fields[key_index].compareTo(s.fields[key_index])==0) ? 0 : -1);
			}
			else if (this.key_type.equals("int"))
			{
				return Integer.parseInt(fields[key_index]) < Integer.parseInt(s.fields[key_index]) ? 1 : (Integer.parseInt(fields[key_index]) ==Integer.parseInt(s.fields[key_index])  ? 0 : -1);
			}
			else if (this.key_type.equals("dou"))
			{
				return Double.parseDouble(fields[key_index]) < Double.parseDouble(s.fields[key_index]) ? 1 : (Double.parseDouble(fields[key_index]) ==Double.parseDouble(s.fields[key_index])  ? 0 : -1);
			}
			return 0;
			
		};
		
		public String toString(){
			String info="";
			for(int i=0;i<fields.length;i++)
			{
				info=info+fields[i]+"\001";
			}
					
			info=info.trim();	
		    return  info ;
		}
	}
	
	/**
	 * input is an array composed of strings
	 * 
	 *   each string's format is :
	 *     line .=. <field1><sep><field2><sep>.....<fieldN>
	 *     
	 *   key_index is the index of sort key in the line ,start with zero
	 *   key_type  is the type of sort key: 
	 *             support three types
	 *               	    str:string
	 *                      int:integer
	 *                      dou:double
	 *   field_num is the number of fields in each line  
	 *   separator is the separator of each line
	 *   
	 *   e.g.
	 *     	String[] sortLabels  = SortStrArray.sort_array(labels, 1, "dou", 2, ":");
	 * @param arr : input array
	 * @param key_index: the  index of sort key  
	 * @param key_type
	 * @return sorted array
	 */
	public static String[] sort_array(String[] arr,int key_index,String key_type,int field_num,String separator)
	{
		ArrayList<SortElement> al = new ArrayList<SortElement>();
		
	 	SortElement sorele=null;
	 	
	 	String[] seg_arr=null;
	 	String row="";
	 	
		for(int i=0;i<arr.length;i++)
		{
			row=arr[i].trim();
			if(!(SSO.tnoe(row)))
			{
				continue;
			}
			seg_arr=row.split(separator);
			if(seg_arr.length!=field_num)
			{
				continue;
			}
			
			sorele=new SortElement(key_index,seg_arr,key_type);
			al.add(sorele);
		}
		Collections.sort(al);
		String[] narr=new String[al.size()];
		
		
		for(int i=0;i<narr.length;i++)
		{
			sorele=al.get(i);
			seg_arr=sorele.fields;
			row=SSO.implode(seg_arr, separator);
			narr[i]=row;
		}
		
		return narr;
	}
	
	/**
	 * input is an arrayList composed of strings
	 * 
	 *   each string's format is :
	 *     line .=. <field1><sep><field2><sep>.....<fieldN>
	 *     
	 *   key_index is the index of sort key in the line ,start with zero
	 *   key_type  is the type of sort key: 
	 *             support three types
	 *               	    str:string
	 *                      int:integer
	 *                      dou:double
	 *   field_num is the number of fields in each line  
	 *   separator is the separator of each line
	 *   
	 *   e.g.
	 *     	String[] sortLabels  = SortStrArray.sort_array(labelList, 1, "dou", 2, ":");
	 * @param arr : input array
	 * @param key_index: the  index of sort key  
	 * @param key_type
	 * @return sorted array
	 */
	public static String[] sort_List(ArrayList<String> arr,int key_index,String key_type,int field_num,String separator)
	{
		ArrayList<SortElement> al = new ArrayList<SortElement>();
		
	 	SortElement sorele=null;
	 	
	 	String[] seg_arr=null;
	 	String row="";
	 	
		for(int i=0;i<arr.size();i++)
		{
			row=(arr.get(i)).trim();
			if(!(SSO.tnoe(row)))
			{
				continue;
			}
			seg_arr=row.split(separator);
			if(seg_arr.length!=field_num)
			{
				continue;
			}
			
			sorele=new SortElement(key_index,seg_arr,key_type);
			al.add(sorele);
		}
		Collections.sort(al);
		String[] narr=new String[al.size()];
		
		
		for(int i=0;i<narr.length;i++)
		{
			sorele=al.get(i);
			seg_arr=sorele.fields;
			row=SSO.implode(seg_arr, separator);
			narr[i]=row;
		}
		
		return narr;
	}
	
	public static void main(String[] args)
	{
		String[] arr={"3\001hello\0012010-10-18 05:02:03","5\001the\0012010-10-18 03:00:02","2\001word\0012010-10-18 07:00:01","4\001!\0012010-10-18 09:01:00"};
		
		String[] narr=SortStrArray.sort_array(arr, 0, "int", 3,"\001");
		
		for(int i=0;i<narr.length;i++)
		{
			System.out.println(narr[i]);
		}
		
		
		
	}
	
	
	
}
