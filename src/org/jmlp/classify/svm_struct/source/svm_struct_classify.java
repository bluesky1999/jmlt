package org.jmlp.classify.svm_struct.source;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;

import org.apache.log4j.Logger;
import org.jmlp.file.utils.FileWriterUtil;
import org.jmlp.str.basic.SSO;

public class svm_struct_classify {

	public static String testfile = "";
	public static String modelfile = "";
	public static String predictionsfile = "";
	private STRUCTMODEL model;
	private STRUCT_LEARN_PARM sparm;
	svm_struct_api ssa = null;
	private static Logger logger = Logger.getLogger(svm_struct_classify.class);

	public void init_svm_struct(String model_file) {
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

		svm_struct_api.svm_struct_classify_api_init(args.length + 1, args);

		read_input_parameters(args.length + 1, args, sparm,
				svm_common.verbosity, svm_struct_common.struct_verbosity);

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading model ...");
		}

		logger.info("testfile:" + testfile);
		logger.info("modelfile:" + modelfile);
		logger.info("predictionsfile:" + predictionsfile);
		model = svm_struct_api.read_struct_model(modelfile, sparm);

		if (model.svm_model.kernel_parm.kernel_type == svm_common.LINEAR) {
			logger.info("begin add_weight_vector_to_linear_model");
			svm_common.add_weight_vector_to_linear_model(model.svm_model);
			logger.info("after add_weight_vector_to_linear_model");
			model.w = model.svm_model.lin_weights;
		}
		ssa = svm_struct_api_factory.get_svm_struct_api();
	}

	public LABEL classifyWordString(String sample)
	{	
		return ssa.classify_struct_example(ssa.sample2pattern(sample), model, sparm);
	}
	
	/**
	 * 从arraylist读取未分类样本， 结果写到标准输出 输入格式: identified format samples
	 * 输出格式：identified label
	 */
	public void classify_from_arraylist(ArrayList<String> input_list,
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

		svm_struct_api.svm_struct_classify_api_init(args.length + 1, args);

		read_input_parameters(args.length + 1, args, sparm,
				svm_common.verbosity, svm_struct_common.struct_verbosity);

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading model ...");
		}

		logger.info("testfile:" + testfile);
		logger.info("modelfile:" + modelfile);
		logger.info("predictionsfile:" + predictionsfile);
		logger.info("begin read model:");
		model = svm_struct_api.read_struct_model(modelfile, sparm);
		logger.info("after read model:");

		if (model.svm_model.kernel_parm.kernel_type == svm_common.LINEAR) {
			logger.info("begin add_weight_vector_to_linear_model");
			svm_common.add_weight_vector_to_linear_model(model.svm_model);
			logger.info("after add_weight_vector_to_linear_model");
			model.w = model.svm_model.lin_weights;

		}

		svm_struct_api ssa = svm_struct_api_factory.get_svm_struct_api();
		// testsample=ssa.read_struct_examples(testfile, sparm);
		logger.info("after get svm struct api");
		ArrayList<String> sample_list = new ArrayList<String>();
		ArrayList<String> identifier_list = new ArrayList<String>();
		String linestd = "";

		String[] tokens = null;
		String docid = "";
		String doclabel = "5";
		String docwords = "";
		logger.info("reading samples to arraylist");
		try {
			// while ((linestd = brstd.readLine())!=null) {
			for (int k = 0; k < input_list.size(); k++) {
				linestd = input_list.get(k);
				if (SSO.tioe(linestd)) {
					continue;
				}
				logger.info("k=" + k + " " + linestd);

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
				logger.info("docid:" + docid + " sample:" + doclabel + " "
						+ docwords);
			}
			// }

		} catch (Exception e) {
			e.printStackTrace();
		}

		logger.info("begin read_struct_examples_from_arraylist");
		testsample = ssa
				.read_struct_examples_from_arraylist(sample_list, sparm);
		logger.info("end read_struct_examples_from_arraylist");
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done.");
		}

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

                logger.info("predict result ===============================");
		for (i = 0; i < testsample.n; i++) {

			t1 = svm_common.get_runtime();
			//logger.info("doc [" + i + "] "
			//		+ testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classify_struct_example(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;
			}
			//logger.info("y:" + y.class_index + "  testsample.examples[" + i
			//		+ "].y:" + testsample.examples[i].y.class_index);
			logger.info(testsample.examples[i].y.class_index+" "+y.class_index);
			runtime += (svm_common.get_runtime() - t1);
			// svm_struct_api.write_label(predfl, y);
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			svm_struct_api.eval_prediction(i, testsample.examples[i], y, model,
					sparm, teststats);

			if (ssa.empty_label(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (svm_struct_common.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done");
			logger.info("Runtime (without IO) in cpu-seconds:"
					+ (float) (runtime / 100.0));
		}

		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		logger.info("Average loss on test set:" + (float) avgloss);
		logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
				/ testsample.n + "(" + correct + " correct, " + incorrect
				+ " incorrect," + testsample.n + ", total");
		System.err.println("Average loss on test set:" + (float) avgloss);
		System.err.println("Zero/one-error on test set " + (float) 100.0 * incorrect
				/ testsample.n + "(" + correct + " correct, " + incorrect
				+ " incorrect," + testsample.n + ", total");
		// }

		svm_struct_api.print_struct_testing_stats(testsample, model, sparm,
				teststats);
	}

	/**
	 * 从标准输入读取未分类样本， 结果写到标准输出 输入格式: identified format samples 输出格式：identified
	 * label
	 */
	public void classify_from_stream(String model_file) {
		
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

		svm_struct_api.svm_struct_classify_api_init(args.length + 1, args);

		read_input_parameters(args.length + 1, args, sparm,
				svm_common.verbosity, svm_struct_common.struct_verbosity);

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading model ...");
		}

		logger.info("testfile:" + testfile);
		logger.info("modelfile:" + modelfile);
		logger.info("predictionsfile:" + predictionsfile);

		model = svm_struct_api.read_struct_model(modelfile, sparm);
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done");
		}

		if (model.svm_model.kernel_parm.kernel_type == svm_common.LINEAR) {
			svm_common.add_weight_vector_to_linear_model(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading test examples ...");
			// System.out.println("Reading test examples ...");
		}

		svm_struct_api ssa = svm_struct_api_factory.get_svm_struct_api();
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
				logger.info("docid:" + docid + " sample:" + doclabel + " "
						+ docwords);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		logger.info("begin read_struct_examples_from_arraylist");
		testsample = ssa
				.read_struct_examples_from_arraylist(sample_list, sparm);
		logger.info("end read_struct_examples_from_arraylist");
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done.");
		}

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

		for (i = 0; i < testsample.n; i++) {

			t1 = svm_common.get_runtime();
			logger.info("doc [" + i + "] "
					+ testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classify_struct_example(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;
			}
			logger.info("y:" + y.class_index + "  testsample.examples[" + i
					+ "].y:" + testsample.examples[i].y.class_index);
			runtime += (svm_common.get_runtime() - t1);
			// svm_struct_api.write_label(predfl, y);
			System.out.println(identifier_list.get(i) + " " + y.toString());

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			svm_struct_api.eval_prediction(i, testsample.examples[i], y, model,
					sparm, teststats);

			if (ssa.empty_label(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (svm_struct_common.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done");
			logger.info("Runtime (without IO) in cpu-seconds:"
					+ (float) (runtime / 100.0));
		}

		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		logger.info("Average loss on test set:" + (float) avgloss);
		logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
				/ testsample.n + "(" + correct + " correct, " + incorrect
				+ " incorrect," + testsample.n + ", total");
		// }

		svm_struct_api.print_struct_testing_stats(testsample, model, sparm,
				teststats);

	}
	

	public static void main(String[] args) throws Exception {
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

		svm_struct_api.svm_struct_classify_api_init(args.length + 1, args);

		read_input_parameters(args.length + 1, args, sparm,
				svm_common.verbosity, svm_struct_common.struct_verbosity);

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading model ...");
		}

		logger.info("testfile:" + testfile);
		logger.info("modelfile:" + modelfile);
		logger.info("predictionsfile:" + predictionsfile);

                PrintWriter pw=new PrintWriter(predictionsfile);
		model = svm_struct_api.read_struct_model(modelfile, sparm);
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done");
		}

		if (model.svm_model.kernel_parm.kernel_type == svm_common.LINEAR) {
			svm_common.add_weight_vector_to_linear_model(model.svm_model);
			model.w = model.svm_model.lin_weights;
		}

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Reading test examples ...");
			// System.out.println("Reading test examples ...");
		}

		svm_struct_api ssa = svm_struct_api_factory.get_svm_struct_api();

		testsample = ssa.read_struct_examples(testfile, sparm);

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done.");
		}

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Classifying test examples ...");
		}

		// predfl = FileWriterUtil.getPW(predictionsfile);

		for (i = 0; i < testsample.n; i++) {

			t1 = svm_common.get_runtime();
			//logger.info("doc [" + i + "] "
			//		+ testsample.examples[i].x.doc.fvec.toString());
			y = ssa.classify_struct_example(testsample.examples[i].x, model,
					sparm);
			if (y == null) {
				continue;
			}
			//logger.info("y:" + y.class_index + "  testsample.examples[" + i
			//		+ "].y:" + testsample.examples[i].y.class_index);
		        logger.info(testsample.examples[i].y.class_index+" "+y.class_index);
                        pw.println(testsample.examples[i].y.class_index+" "+y.class_index);
			runtime += (svm_common.get_runtime() - t1);
			// svm_struct_api.write_label(predfl, y);

			l = ssa.loss(testsample.examples[i].y, y, sparm);

			avgloss += l;
			if (l == 0) {
				correct++;
			} else {
				incorrect++;
			}

			svm_struct_api.eval_prediction(i, testsample.examples[i], y, model,
					sparm, teststats);

			if (ssa.empty_label(testsample.examples[i].y)) {
				no_accuracy = 1;
			}

			if (svm_struct_common.struct_verbosity >= 2) {
				if ((i + 1) % 100 == 0) {
					logger.info(i + 1);
				}
			}

		}

		avgloss /= testsample.n;
		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("done");
			logger.info("Runtime (without IO) in cpu-seconds:"
					+ (float) (runtime / 100.0));
		}
                pw.close();
		// if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1))
		// {
		logger.info("Average loss on test set:" + (float) avgloss);
		logger.info("Zero/one-error on test set " + (float) 100.0 * incorrect
				/ testsample.n + "(" + correct + " correct, " + incorrect
				+ " incorrect," + testsample.n + ", total");
		System.out.println("Average loss on test set:" + (float) avgloss);
		System.out.println("Zero/one-error on test set " + (float) 100.0 * incorrect
				/ testsample.n + "(" + correct + " correct, " + incorrect
				+ " incorrect," + testsample.n + ", total");
		// }

		svm_struct_api.print_struct_testing_stats(testsample, model, sparm,
				teststats);

	}

	/*
	 * public void classfiy(String[] args,STRUCTMODEL model,STRUCT_LEARN_PARM
	 * sparm,MODEL modelt,SAMPLE sample) { int
	 * correct=0,incorrect=0,no_accuracy=0; int i; double t1,runtime=0; double
	 * avgloss=0,l=0; PrintWriter predfl; STRUCT_TEST_STATS teststats=null;
	 * //SAMPLE testsample; LABEL y; testfile=args[0]; modelfile=args[1];
	 * predictionsfile=args[2];
	 * svm_struct_api.svm_struct_classify_api_init(args.length+1,args);
	 * 
	 * read_input_parameters(args.length+1,args,sparm,
	 * svm_common.verbosity,svm_struct_common.struct_verbosity);
	 * 
	 * if(svm_struct_common.struct_verbosity>=1) {
	 * logger.info("Reading model ..."); }
	 * 
	 * //model=svm_struct_api.read_struct_model(modelfile, sparm);
	 * if(svm_struct_common.struct_verbosity>=1) { logger.info("done"); }
	 * 
	 * 
	 * if(model.svm_model.kernel_parm.kernel_type==svm_common.LINEAR) {
	 * //svm_common.add_weight_vector_to_linear_model(model.svm_model);
	 * //model.w=model.svm_model.lin_weights; }
	 * 
	 * if(svm_struct_common.struct_verbosity>=1) {
	 * logger.info("Reading test examples ..."); }
	 * 
	 * //testsample=svm_struct_api.read_struct_examples(testfile, sparm);
	 * 
	 * if(svm_struct_common.struct_verbosity>=1) { logger.info("done."); }
	 * 
	 * 
	 * if(svm_struct_common.struct_verbosity>=1) {
	 * logger.info("Classifying test examples ..."); }
	 * 
	 * 
	 * predfl=FileWriterUtil.getPW(predictionsfile);
	 * 
	 * for(i=0;i<sample.n;i++) {
	 * //logger.info("test i:"+i+" "+testsample.examples
	 * [i].x.doc.fvec.toString()); t1=svm_common.get_runtime();
	 * y=svm_struct_api.classify_struct_example(sample.examples[i].x, model,
	 * sparm);
	 * logger.info("y:"+y.class_index+"  testsample.examples[i].y:"+sample
	 * .examples[i].y.class_index); runtime+=(svm_common.get_runtime()-t1);
	 * svm_struct_api.write_label(predfl, y);
	 * 
	 * l=svm_struct_api.loss(sample.examples[i].y, y, sparm);
	 * 
	 * avgloss+=l; if(l==0) { correct++; } else { incorrect++; }
	 * 
	 * svm_struct_api.eval_prediction(i,sample.examples[i],y, model, sparm,
	 * teststats);
	 * 
	 * if(svm_struct_api.empty_label(sample.examples[i].y)) { no_accuracy=1; }
	 * 
	 * if(svm_struct_common.struct_verbosity>=2) { if((i+1)%100==0) {
	 * logger.info(i+1); } }
	 * 
	 * }
	 * 
	 * avgloss/=sample.n; if(svm_struct_common.struct_verbosity>=1) {
	 * logger.info("done");
	 * logger.info("Runtime (without IO) in cpu-seconds:"+(float
	 * )(runtime/100.0)); }
	 * 
	 * if((no_accuracy==0)&&(svm_struct_common.struct_verbosity>=1)) {
	 * logger.info("Average loss on test set:"+(float)avgloss);
	 * logger.info("Zero/one-error on test set "
	 * +(float)100.0*incorrect/sample.n+
	 * "("+correct+" correct, "+incorrect+" incorrect,"+sample.n+", total"); }
	 * 
	 * svm_struct_api.print_struct_testing_stats(sample, model, sparm,
	 * teststats); }
	 */
	public static void read_input_parameters(int argc, String[] argv,
			STRUCT_LEARN_PARM struct_parm, int verbosity, int struct_verbosity) {
		int i;
		modelfile = "svm_model";
		predictionsfile = "svm_predictions";
		verbosity = 0;
		struct_verbosity = 1;
		struct_parm.custom_argc = 0;

		for (i = 1; (i < argc) && ((argv[i].charAt(0)) == '-'); i++) {
			switch ((argv[i].charAt(1))) {
			case 'h':
				print_help();
				System.exit(0);
			case '?':
				print_help();
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
				print_help();
				System.exit(0);
			}
		}

		if ((i + 1) >= argc) {
			// System.out.println("Not enough input parameters!");
			print_help();
			System.exit(0);
		}

		testfile = argv[0];
		modelfile = argv[1];

		// System.out.println("testfile:" + testfile);
		// System.out.println("modelfile:" + modelfile);

		if ((i + 2) < argc) {
			predictionsfile = argv[2];
		}

		svm_struct_api.parse_struct_parameters_classify(struct_parm);
	}

	public static void print_help() {
		System.out.println("\nSVM-struct classification module: "
				+ svm_struct_common.INST_NAME + ", "
				+ svm_struct_common.INST_VERSION + ", "
				+ svm_struct_common.INST_VERSION_DATE + "\n");
		System.out.println("   includes SVM-struct "
				+ svm_struct_common.STRUCT_VERSION
				+ " for learning complex outputs, "
				+ svm_struct_common.STRUCT_VERSION_DATE + "\n");
		System.out.println("   includes SVM-light " + svm_common.VERSION
				+ " quadratic optimizer, " + svm_common.VERSION_DATE + "\n");
		svm_common.copyright_notice();
		System.out
				.println("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
		System.out.println("options: -h         -> this help\n");
		System.out
				.println("         -v [0..3]  -> verbosity level (default 2)\n\n");

		svm_struct_api.print_struct_help_classify();
	}

}
