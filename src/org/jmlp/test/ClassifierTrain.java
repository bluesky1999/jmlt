package org.jmlp.test;

import org.jmlp.classify.svm_struct.source.svm_struct_api_factory;
import org.jmlp.classify.svm_struct.source.svm_struct_main;

/**
 * 训练分类模型
 * 输入：format samples
 * 输出: classifier model
 * @author zkyz
 */
public class ClassifierTrain {

	public static void main(String[] args) {

		svm_struct_main svm_struct = new svm_struct_main();

		if (args.length < 1) {
			System.out
					.println("Usage:ClassifierModel [<api_type>] <model> \n"
							+ " api_type: svm struct api type for example:multiclass, \n"
							+ " model: model save path \n");
			System.exit(1);
		}

		if (args.length == 1) {// default: multiclass
			double c = 5000.0;
			
			//选用何种分类体系
			svm_struct_api_factory ssaf = new svm_struct_api_factory(0);
			
			svm_struct.train_from_stream(c, args[0]);

		} else if (args.length == 2) {
			double c = 5000.0;
			svm_struct_api_factory ssaf = new svm_struct_api_factory(
					Integer.parseInt(args[0]));
			svm_struct.train_from_stream(c, args[1]);

			
			
			
		}

	}

}
