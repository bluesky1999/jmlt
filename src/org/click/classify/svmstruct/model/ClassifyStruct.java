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
		// System.err.println("in init_svm_struct");
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

		readInputParameters(args.length + 1, args, sparm, Common.verbosity,
				CommonStruct.struct_verbosity);

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading model ...");
		}

		// logger.info("testfile:" + testfile);
		// logger.info("modelfile:" + modelfile);
		// logger.info("predictionsfile:" + predictionsfile);
		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			// logger.info("begin add_weight_vector_to_linear_model");
			com.addWeightVectorToLinearModel(model.svm_model);
			// logger.info("after add_weight_vector_to_linear_model");
			model.w = model.svm_model.lin_weights;
		}
		ssa = FactoryStruct.get_svm_struct_api();
	}

	public LABEL classifyWordString(String sample) {
		// return ssa.classifyStructExample(ssa.sample2pattern(sample), model,
		// sparm);
		if (FactoryStruct.api_type == 2)
			return ssa.classifyStructDoc(ssa.sample2doc(sample), model, sparm);
		else if (FactoryStruct.api_type == 0)
			return ssa.classifyStructExample(ssa.sample2pattern(sample), model,
					sparm);

		return null;
	}

	/**
	 * 从arraylist读取未分类样本， 结果写到标准输出 输入格式: identified format samples
	 * 输出格式：identified label
	 */
	public void classifyFromArraylist(ArrayList<String> input_list,
			String model_file) {
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

		readInputParameters(args.length + 1, args, sparm, Common.verbosity,
				CommonStruct.struct_verbosity);

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading model ...");
		}

		// logger.info("testfile:" + testfile);
		// logger.info("modelfile:" + modelfile);
		// logger.info("predictionsfile:" + predictionsfile);
		// logger.info("begin read model:");
		model = ssa.readStructModel(modelfile, sparm);
		// logger.info("after read model:");

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			// logger.info("begin add_weight_vector_to_linear_model");
			com.addWeightVectorToLinearModel(model.svm_model);
			// logger.info("after add_weight_vector_to_linear_model");
			model.w = model.svm_model.lin_weights;

		}

		Struct ssa = FactoryStruct.get_svm_struct_api();
		// testsample=ssa.read_struct_examples(testfile, sparm);
		// logger.info("after get svm struct api");
		ArrayList<String> sample_list = new ArrayList<String>();
		ArrayList<String> identifier_list = new ArrayList<String>();
		String linestd = "";

		String[] tokens = null;
		String docid = "";
		String doclabel = "5";
		String docwords = "";
		// logger.info("reading samples to arraylist");
		try {

			for (int k = 0; k < input_list.size(); k++) {
				linestd = input_list.get(k);
				if (SSO.tioe(linestd)) {
					continue;
				}
				// //logger.info("k=" + k + " " + linestd);

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
				// logger.info("docid:" + docid + " sample:" + doclabel + " "
				// + docwords);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		// logger.info("begin read_struct_examples_from_arraylist");
		testsample = ssa.readStructExamplesFromArraylist(sample_list, sparm);
		// logger.info("end read_struct_examples_from_arraylist");
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done.");
		}

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

		// logger.info("predict result ===============================");
		for (i = 0; i < testsample.n; i++) {

			t1 = com.getRuntime();
			// //logger.info("doc [" + i + "] "
			// + testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classifyStructExample(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;
			}
			// //logger.info("y:" + y.class_index + "  testsample.examples[" + i
			// + "].y:" + testsample.examples[i].y.class_index);
			// logger.info(testsample.examples[i].y.class_index + " "
			// + y.class_index);
			runtime += (com.getRuntime() - t1);
			// svm_struct_api.write_label(predfl, y);
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm,
					teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (CommonStruct.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					// logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done");
			// logger.info("Runtime (without IO) in cpu-seconds:"
			// + (float) (runtime / 100.0));
		}

		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		// logger.info("Average loss on test set:" + (float) avgloss);
		// logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
		// / testsample.n + "(" + correct + " correct, " + incorrect
		// + " incorrect," + testsample.n + ", total");
		System.err.println("Average loss on test set:" + (float) avgloss);
		System.err.println("Zero/one-error on test set " + (float) 100.0
				* incorrect / testsample.n + "(" + correct + " correct, "
				+ incorrect + " incorrect," + testsample.n + ", total");
		// }

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

		readInputParameters(args.length + 1, args, sparm, Common.verbosity,
				CommonStruct.struct_verbosity);

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading model ...");
		}

		// logger.info("testfile:" + testfile);
		// logger.info("modelfile:" + modelfile);
		// logger.info("predictionsfile:" + predictionsfile);

		model = ssa.readStructModel(modelfile, sparm);
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done");
		}

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading test examples ...");
			// System.out.println("Reading test examples ...");
		}

		Struct ssa = FactoryStruct.get_svm_struct_api();
		// testsample=ssa.read_struct_examples(testfile, sparm);

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
				// logger.info("docid:" + docid + " sample:" + doclabel + " "
				// + docwords);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		// logger.info("begin read_struct_examples_from_arraylist");
		testsample = ssa.readStructExamplesFromArraylist(sample_list, sparm);
		// logger.info("end read_struct_examples_from_arraylist");
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done.");
		}

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

		for (i = 0; i < testsample.n; i++) {

			t1 = com.getRuntime();
			// logger.info("doc [" + i + "] "
			// + testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classifyStructExample(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;
			}
			// logger.info("y:" + y.class_index + "  testsample.examples[" + i
			// + "].y:" + testsample.examples[i].y.class_index);
			runtime += (com.getRuntime() - t1);
			// svm_struct_api.write_label(predfl, y);
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm,
					teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (CommonStruct.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					// logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done");
			// logger.info("Runtime (without IO) in cpu-seconds:"
			// + (float) (runtime / 100.0));
		}

		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		// logger.info("Average loss on test set:" + (float) avgloss);
		// logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
		// / testsample.n + "(" + correct + " correct, " + incorrect
		// + " incorrect," + testsample.n + ", total");
		// }

		ssa.printStructTestingStats(testsample, model, sparm, teststats);

	}

	public void readInputParameters(int argc, String[] argv,
			STRUCT_LEARN_PARM struct_parm, int verbosity, int struct_verbosity) {
		// System.err.println("in read_input_parameters");
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
				System.out
						.println("\nUnrecognized option " + argv[i] + "!\n\n");
				printHelp();
				System.exit(0);
			}
		}

		if ((i + 1) >= argc) {
			// System.out.println("Not enough input parameters!");
			printHelp();
			System.exit(0);
		}

		testfile = argv[0];
		modelfile = argv[1];

		// System.out.println("testfile:" + testfile);
		// System.out.println("modelfile:" + modelfile);

		if ((i + 2) < argc) {
			predictionsfile = argv[2];
		}
		Struct ssa = FactoryStruct.get_svm_struct_api();
		ssa.parseStructParametersClassify(struct_parm);
	}

	public void printHelp() {
		System.out.println("\nSVM-struct classification module: "
				+ CommonStruct.INST_NAME + ", " + CommonStruct.INST_VERSION
				+ ", " + CommonStruct.INST_VERSION_DATE + "\n");
		System.out.println("   includes SVM-struct "
				+ CommonStruct.STRUCT_VERSION
				+ " for learning complex outputs, "
				+ CommonStruct.STRUCT_VERSION_DATE + "\n");
		System.out.println("   includes SVM-light " + ModelConstant.VERSION
				+ " quadratic optimizer, " + ModelConstant.VERSION_DATE + "\n");
		com.copyright_notice();
		System.out
				.println("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
		System.out.println("options: -h         -> this help\n");
		System.out
				.println("         -v [0..3]  -> verbosity level (default 2)\n\n");
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

		cs.readInputParameters(args.length + 1, args, sparm, Common.verbosity,
				CommonStruct.struct_verbosity);

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading model ...");
		}

		// logger.info("testfile:" + testfile);
		// logger.info("modelfile:" + modelfile);
		// logger.info("predictionsfile:" + predictionsfile);

		PrintWriter pw = new PrintWriter(predictionsfile);
		model = ssa.readStructModel(modelfile, sparm);
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done");
		}

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			cs.com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Reading test examples ...");
			// System.out.println("Reading test examples ...");
		}

		// svm_struct_api ssa = svm_struct_api_factory.get_svm_struct_api();

		testsample = ssa.readStructExamples(testfile, sparm);

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done.");
		}

		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

		for (i = 0; i < testsample.n; i++) {

			t1 = cs.com.getRuntime();
			// //logger.info("doc [" + i + "] "
			// + testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classifyStructExample(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;

			}

			if (FactoryStruct.api_type == 2) {
				ssa.writeLabel(pw, y, testsample.examples[i].y);
			}
			// //logger.info("y:" + y.class_index + "  testsample.examples[" + i
			// + "].y:" + testsample.examples[i].y.class_index);
			// logger.info(testsample.examples[i].y.class_index + " "
			// + y.class_index);
			if (FactoryStruct.api_type != 2) {
				pw.println(testsample.examples[i].y.class_index + " "
						+ y.class_index);
			}
			runtime += (cs.com.getRuntime() - t1);
			// svm_struct_api.write_label(predfl, y);

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			ssa.evalPrediction(i, testsample.examples[i], y, model, sparm,
					teststats);

			if (ssa.emptyLabel(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (CommonStruct.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					// logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (CommonStruct.struct_verbosity >= 1) {
			// logger.info("done");
			// logger.info("Runtime (without IO) in cpu-seconds:"
			// + (float) (runtime / 100.0));
		}
		pw.close();
		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		// logger.info("Average loss on test set:" + (float) avgloss);
		// logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
		// / testsample.n + "(" + correct + " correct, " + incorrect
		// + " incorrect," + testsample.n + ", total");
		System.out.println("Average loss on test set:" + (float) avgloss);
		System.out.println("Zero/one-error on test set " + (float) 100.0
				* incorrect / testsample.n + "(" + correct + " correct, "
				+ incorrect + " incorrect," + testsample.n + ", total");
		// }

		ssa.printStructTestingStats(testsample, model, sparm, teststats);

	}

}
