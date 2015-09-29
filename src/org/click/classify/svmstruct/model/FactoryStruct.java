package org.click.classify.svmstruct.model;

public class FactoryStruct {

	public static int api_type=0;
	
	
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
