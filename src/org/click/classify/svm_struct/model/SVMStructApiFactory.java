package org.click.classify.svm_struct.model;

import java.util.Properties;

public class SVMStructApiFactory {

	public static int api_type=0;
	
	
	public SVMStructApiFactory()
	{
		this.api_type=0;
	}
	
	public SVMStructApiFactory(int api_type)
	{
		this.api_type=api_type;
	}
	
	public SVMStructApiFactory(Properties prop)
	{
	   	String api_type_conf=prop.getProperty("api_type");
	   	if(api_type_conf!=null)
	   	{
	   		this.api_type=Integer.parseInt(api_type_conf);
	   	}	
	   	
	}
	
	public static SVMStructApi get_svm_struct_api()
	{
		if(api_type==0)//多分类
			return new SVMStructApiMulticlass();
		else if(api_type==1)
			return null;
		else if(api_type==2)
			return new SVMStructApiPerf();

		return null;
	}
	
}
