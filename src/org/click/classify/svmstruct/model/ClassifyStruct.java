package org.click.classify.svmstruct.model;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;

import org.click.classify.svmstruct.data.LABEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.SAMPLE;
import org.click.classify.svmstruct.data.STRUCTMODEL;
import org.click.classify.svmstruct.data.STRUCT_LEARN_PARM;
import org.click.classify.svmstruct.data.STRUCT_TEST_STATS;
import org.click.lib.string.SSO;

public class ClassifyStruct {

	public static String testfile = "";
	public static String modelfile = "";
	public static String predictionsfile = "";
	private STRUCTMODEL model;
	private STRUCT_LEARN_PARM sparm;
	public Struct ssa = null;

	private Common com = null;

	public void init_svm_struct(String model_file) {
		com = new Common();
		String[] args = { "no.txt", model_file, "no.txt" };
		int correct = 0, incorrect = 0, no_accuracy = 0;
		int i;
		double t1, runtime = 0;
		double avgloss = 0, l = 0;
		PrintWriter predfl;
		sparm = new STRUCT_LEARN_PARM();
		STRUCT_TEST_STATS teststats = null;
		SAMPLE testsample;
		LABEL y = new LABEL();

		ssa = FactoryStruct.get_svm_struct_api();
		ssa.svmStructClassifyApiInit(args.length + 1, args);

		readInputParameters(args.length + 1, args, sparm, CommonStruct.verbosity, CommonStruct.struct_verbosity);


		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}
		ssa = FactoryStruct.get_svm_struct_api();
	}

	/**
	 * the format of sample: <br>
	 * &nbsp; line .=. [(wordIndex:wordValue)]* <br>
	 * &nbsp; e.g. <br>
	 * &nbsp;&nbsp;&nbsp; 2:0.24666666666666667 3:0.38 14:1.0 16:0.33 17:1.0
	 * 56:0.25 90:0.13 94:0.63
	 * 
	 * @param sample
	 * @return
	 */
	public LABEL classifyWordString(String sample) {
		// return ssa.classifyStructExample(ssa.sample2pattern(sample), model,
		// sparm);
		if (FactoryStruct.api_type == 2)
			return ssa.classifyStructDoc(ssa.sample2doc(sample), model, sparm);
		else if (FactoryStruct.api_type == 0)
			return ssa.classifyStructExample(ssa.sample2pattern(sample), model, sparm);

		return null;
	}

	/**
	 * 从arraylist读取未分类样本， 结果写到标准输出 输入格式: identified format samples
	 * 输出格式：identified label
	 */
	public void classifyFromArraylist(ArrayList<String> input_list, String model_file) {
		String[] args = { "no.txt", model_file, "no.txt" };
		int correct = 0, incorrect = 0, no_accuracy = 0;
		int i;
		double t1, runtime = 0;
		double avgloss = 0, l = 0;
		PrintWriter predfl;
		STRUCTMODEL model;
		STRUCT_LEARN_PARM sparm = new STRUCT_LEARN_PARM();
		STRUCT_TEST_STATS teststats = null;
		SAMPLE testsample;
		LABEL y = new LABEL();

		ssa = FactoryStruct.get_svm_struct_api();
		ssa.svmStructClassifyApiInit(args.length + 1, args);

		readInputParameters(args.length + 1, args, sparm,CommonStruct.verbosity, CommonStruct.struct_verbosity);

		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;

		}

		Struct ssa = FactoryStruct.get_svm_struct_api();
		ArrayList<String> sample_list = new ArrayList<String>();
		ArrayList<String> identifier_list = new ArrayList<String>();
		String linestd = "";

		String[] tokens = null;
		String docid = "";
		String doclabel = "5";
		String docwords = "";

		try {

			for (int k = 0; k < input_list.size(); k++) {
				linestd = input_list.get(k);
				if (SSO.tioe(linestd)) {
					continue;
				}
	
				tokens = linestd.split("\\s+");
				if (tokens.length < 2) {
					continue;
				}

				docid = tokens[0];

				docwords = "";
				for (int j = 1; j < tokens.length; j++) {
					docwords += (tokens[j] + " ");
				}
				docwords = docwords.trim();
				identifier_list.add(docid);
				sample_list.add(doclabel + " " + docwords);

			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		testsample = ssa.readStructExamplesFromArraylist(sample_list, sparm);

		for (i = 0; i < testsample.n; i++) {

			t1 = com.getRuntime();
			y = ssa.classifyStructExample(testsample.examples[i].x, model, sparm);
			if (y == null) {
				continue;
			}

			runtime += (com.getRuntime() - t1);
			
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm, teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}


		}

		avgloss /= testsample.n;


		System.err.println("Average loss on test set:" + (float) avgloss);
		System.err.println("Zero/one-error on test set " + (float) 100.0 * incorrect / testsample.n + "(" + correct + " correct, " + incorrect + " incorrect," + testsample.n + ", total");

		ssa.printStructTestingStats(testsample, model, sparm, teststats);
	}

	/**
	 * 从标准输入读取未分类样本， 结果写到标准输出 输入格式: identified format samples 输出格式：identified
	 * label
	 */
	public void classifyFromStream(String model_file) {

		String[] args = { "no.txt", model_file, "no.txt" };
		int correct = 0, incorrect = 0, no_accuracy = 0;
		int i;
		double t1, runtime = 0;
		double avgloss = 0, l = 0;
		PrintWriter predfl;
		STRUCTMODEL model;
		STRUCT_LEARN_PARM sparm = new STRUCT_LEARN_PARM();
		STRUCT_TEST_STATS teststats = null;
		SAMPLE testsample;
		LABEL y = new LABEL();

		ssa = FactoryStruct.get_svm_struct_api();
		ssa.svmStructClassifyApiInit(args.length + 1, args);

		readInputParameters(args.length + 1, args, sparm, CommonStruct.verbosity, CommonStruct.struct_verbosity);


		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}


		Struct ssa = FactoryStruct.get_svm_struct_api();

		InputStreamReader isrstd = new InputStreamReader(System.in);
		BufferedReader brstd = new BufferedReader(isrstd);
		ArrayList<String> sample_list = new ArrayList<String>();
		ArrayList<String> identifier_list = new ArrayList<String>();
		String linestd = "";

		String[] tokens = null;
		String docid = "";
		String doclabel = "5";
		String docwords = "";
		try {
			while ((linestd = brstd.readLine()) != null) {

				linestd = linestd.trim();
				tokens = linestd.split("\\s+");
				if (tokens.length < 2) {
					continue;
				}

				docid = tokens[0];

				docwords = "";
				for (int j = 1; j < tokens.length; j++) {
					docwords += (tokens[j] + " ");
				}
				docwords = docwords.trim();
				identifier_list.add(docid);
				sample_list.add(doclabel + " " + docwords);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		testsample = ssa.readStructExamplesFromArraylist(sample_list, sparm);

		for (i = 0; i < testsample.n; i++) {

			t1 = com.getRuntime();
			y = ssa.classifyStructExample(testsample.examples[i].x, model, sparm);
			if (y == null) {
				continue;
			}
			runtime += (com.getRuntime() - t1);
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm, teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}


		}

		avgloss /= testsample.n;


		ssa.printStructTestingStats(testsample, model, sparm, teststats);

	}

	public void readInputParameters(int argc, String[] argv, STRUCT_LEARN_PARM struct_parm, int verbosity, int struct_verbosity) {
		int i;
		modelfile = "svm_model";
		predictionsfile = "svm_predictions";
		verbosity = 0;
		struct_verbosity = 1;
		struct_parm.custom_argc = 0;

		for (i = 1; (i < argc) && ((argv[i].charAt(0)) == '-'); i++) {
			switch ((argv[i].charAt(1))) {
			case 'h':
				printHelp();
				System.exit(0);
			case '?':
				printHelp();
				System.exit(0);
			case '-':
				struct_parm.custom_argv[struct_parm.custom_argc++] = argv[i];
				i++;
				struct_parm.custom_argv[struct_parm.custom_argc++] = argv[i];
				break;
			case 'v':
				i++;
				struct_verbosity = Integer.parseInt(argv[i]);
				break;
			case 'y':
				i++;
				verbosity = Integer.parseInt(argv[i]);
				break;
			default:
				System.out.println("\nUnrecognized option " + argv[i] + "!\n\n");
				printHelp();
				System.exit(0);
			}
		}

		if ((i + 1) >= argc) {
			printHelp();
			System.exit(0);
		}

		testfile = argv[0];
		modelfile = argv[1];

		if ((i + 2) < argc) {
			predictionsfile = argv[2];
		}
		Struct ssa = FactoryStruct.get_svm_struct_api();
		ssa.parseStructParametersClassify(struct_parm);
	}

	public void printHelp() {
		System.out.println("\nSVM-struct classification module: " + CommonStruct.INST_NAME + ", " + CommonStruct.INST_VERSION + ", " + CommonStruct.INST_VERSION_DATE + "\n");
		System.out.println("   includes SVM-struct " + CommonStruct.STRUCT_VERSION + " for learning complex outputs, " + CommonStruct.STRUCT_VERSION_DATE + "\n");
		System.out.println("   includes SVM-light " + ModelConstant.VERSION + " quadratic optimizer, " + ModelConstant.VERSION_DATE + "\n");
		com.copyright_notice();
		System.out.println("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
		System.out.println("options: -h         -> this help\n");
		System.out.println("         -v [0..3]  -> verbosity level (default 2)\n\n");
		Struct ssa = FactoryStruct.get_svm_struct_api();
		ssa.printStructHelpClassify();
	}

	public static void main(String[] args) throws Exception {

		ClassifyStruct cs = new ClassifyStruct();
		cs.com = new Common();
		int correct = 0, incorrect = 0, no_accuracy = 0;
		int i;
		double t1, runtime = 0;
		double avgloss = 0, l = 0;
		// PrintWriter predfl;
		STRUCTMODEL model;
		STRUCT_LEARN_PARM sparm = new STRUCT_LEARN_PARM();
		STRUCT_TEST_STATS teststats = null;
		SAMPLE testsample;
		LABEL y = new LABEL();
		FactoryStruct.api_type = 0;
		Struct ssa = FactoryStruct.get_svm_struct_api();
		ssa.svmStructClassifyApiInit(args.length + 1, args);

		cs.readInputParameters(args.length + 1, args, sparm, CommonStruct.verbosity, CommonStruct.struct_verbosity);


		PrintWriter pw = new PrintWriter(predictionsfile);
		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			cs.com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		testsample = ssa.readStructExamples(testfile, sparm);

		for (i = 0; i < testsample.n; i++) {

			t1 = cs.com.getRuntime();
			y = ssa.classifyStructExample(testsample.examples[i].x, model, sparm);
			if (y == null) {
				continue;

			}

			if (FactoryStruct.api_type == 2) {
				ssa.writeLabel(pw, y, testsample.examples[i].y);
			}

			if (FactoryStruct.api_type != 2) {
				pw.println(testsample.examples[i].y.class_index + " " + y.class_index);
			}
			runtime += (cs.com.getRuntime() - t1);

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm, teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}


		}

		avgloss /= testsample.n;

		pw.close();

		System.out.println("Average loss on test set:" + (float) avgloss);
		System.out.println("Zero/one-error on test set " + (float) 100.0 * incorrect / testsample.n + "(" + correct + " correct, " + incorrect + " incorrect," + testsample.n + ", total");

		ssa.printStructTestingStats(testsample, model, sparm, teststats);

	}

}
