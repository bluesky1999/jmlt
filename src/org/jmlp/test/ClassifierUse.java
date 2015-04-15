package org.jmlp.test;

import org.jmlp.classify.svm_struct.source.svm_struct_api_factory;
import org.jmlp.classify.svm_struct.source.svm_struct_classify;

/**
 * 调用分类模型
 * 输入：identified format samples
 * 输出：identified label
 * @author zkyz
 */
public class ClassifierUse {

	
	public static void main(String[] args) {
		
		svm_struct_classify ssc=new svm_struct_classify();
		if (args.length < 1) {
			System.out
					.println("Usage:ClassifierUse [<api_type>] [<model>] \n"
							+ " api_type: svm struct api type for example:multiclass, \n"
							+ " model: model save path \n");
			System.exit(1);
		}
		
		if(args.length==0)
		{
			
		}
		else if (args.length == 1) {// default: multiclass
			
			//选用何种分类体系
			svm_struct_api_factory ssaf = new svm_struct_api_factory(0);			
			ssc.classify_from_stream(args[0]);

		} else if (args.length == 2) {

			//选用何种分类体系
			svm_struct_api_factory ssaf = new svm_struct_api_factory(Integer.parseInt(args[0]));
			ssc.classify_from_stream(args[1]);
			
		}
		
	}
	
	
}
