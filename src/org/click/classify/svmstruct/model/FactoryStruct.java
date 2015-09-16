package org.click.classify.svmstruct.model;

import java.util.Properties;

public class FactoryStruct {

	public static int api_type=0;
	
	
	public FactoryStruct()
	{
		this.api_type=0;
	}
	
	public FactoryStruct(int api_type)
	{
		this.api_type=api_type;
	}
	
	public FactoryStruct(Properties prop)
	{
	   	String api_type_conf=prop.getProperty("api_type");
	   	if(api_type_conf!=null)
	   	{
	   		this.api_type=Integer.parseInt(api_type_conf);
	   	}	
	   	
	}
	
	public static Struct get_svm_struct_api()
	{
		if(api_type==0)//多分类
			return new Multiclass();
		else if(api_type==1)
			return null;
		else if(api_type==2)
			return new Perf();

		return null;
	}
	
}
