package org.click.classify.svmstruct.model;


import java.io.PrintWriter;
import org.click.classify.svmstruct.data.LABEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.SAMPLE;
import org.click.classify.svmstruct.data.STRUCTMODEL;
import org.click.classify.svmstruct.data.STRUCT_LEARN_PARM;

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
		sparm = new STRUCT_LEARN_PARM();
		ssa = FactoryStruct.get_svm_struct_api();
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
	}

	public void printHelp() {
		System.out.println("\nSVM-struct classification module: " + CommonStruct.INST_NAME + ", " + CommonStruct.INST_VERSION + ", " + CommonStruct.INST_VERSION_DATE + "\n");
		System.out.println("   includes SVM-struct " + CommonStruct.STRUCT_VERSION + " for learning complex outputs, " + CommonStruct.STRUCT_VERSION_DATE + "\n");
		System.out.println("   includes SVM-light " + ModelConstant.VERSION + " quadratic optimizer, " + ModelConstant.VERSION_DATE + "\n");
		com.copyright_notice();
		System.out.println("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
		System.out.println("options: -h         -> this help\n");
		System.out.println("         -v [0..3]  -> verbosity level (default 2)\n\n");
	}

	public static void main(String[] args) throws Exception {

		ClassifyStruct cs = new ClassifyStruct();
		cs.com = new Common();
		int correct = 0, incorrect = 0;
		int i;
		double avgloss = 0, l = 0;
		STRUCTMODEL model;
		STRUCT_LEARN_PARM sparm = new STRUCT_LEARN_PARM();

		SAMPLE testsample;
		LABEL y = new LABEL();
		FactoryStruct.api_type = 0;
		Struct ssa = FactoryStruct.get_svm_struct_api();
		cs.readInputParameters(args.length + 1, args, sparm, CommonStruct.verbosity, CommonStruct.struct_verbosity);


		PrintWriter pw = new PrintWriter(predictionsfile);
		model = ssa.readStructModel(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			cs.com.addWeightVectorToLinearModel(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		testsample = ssa.readStructExamples(testfile, sparm);

		for (i = 0; i < testsample.n; i++) {
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
			l = ssa.loss(testsample.examples[i].y, y, sparm);
			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

		}

		avgloss /= testsample.n;

		pw.close();

		System.out.println("Average loss on test set:" + (float) avgloss);
		System.out.println("Zero/one-error on test set " + (float) 100.0 * incorrect / testsample.n + "(" + correct + " correct, " + incorrect + " incorrect," + testsample.n + ", total");

	}

}
