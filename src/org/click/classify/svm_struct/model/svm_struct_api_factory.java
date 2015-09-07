package org.click.classify.svm_struct.model;

import java.util.Properties;

public class svm_struct_api_factory {

	public static int api_type=0;
	
	
	public svm_struct_api_factory()
	{
		this.api_type=0;
	}
	
	public svm_struct_api_factory(int api_type)
	{
		this.api_type=api_type;
	}
	
	public svm_struct_api_factory(Properties prop)
	{
	   	String api_type_conf=prop.getProperty("api_type");
	   	if(api_type_conf!=null)
	   	{
	   		this.api_type=Integer.parseInt(api_type_conf);
	   	}	
	   	
	}
	
	public static svm_struct_api get_svm_struct_api()
	{
		if(api_type==0)//多分类
			return new svm_struct_api_multiclass();
		else if(api_type==1)
			return null;
		
		return null;
	}
	
}
