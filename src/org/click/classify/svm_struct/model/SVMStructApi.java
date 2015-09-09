package org.click.classify.svm_struct.model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.regex.Pattern;

import org.click.classify.svm_struct.data.CONSTSET;
import org.click.classify.svm_struct.data.DOC;
import org.click.classify.svm_struct.data.EXAMPLE;
import org.click.classify.svm_struct.data.KERNEL_PARM;
import org.click.classify.svm_struct.data.LABEL;
import org.click.classify.svm_struct.data.LEARN_PARM;
import org.click.classify.svm_struct.data.MODEL;
import org.click.classify.svm_struct.data.ModelConstant;
import org.click.classify.svm_struct.data.PATTERN;
import org.click.classify.svm_struct.data.ReadStruct;
import org.click.classify.svm_struct.data.ReadSummary;
import org.click.classify.svm_struct.data.SAMPLE;
import org.click.classify.svm_struct.data.STRUCTMODEL;
import org.click.classify.svm_struct.data.STRUCT_LEARN_PARM;
import org.click.classify.svm_struct.data.STRUCT_TEST_STATS;
import org.click.classify.svm_struct.data.SVECTOR;
import org.click.classify.svm_struct.data.WORD;
import org.jmlp.str.basic.SSO;

/**
 * svm struct api 的抽象类 不同的分类模型，如多类、多层分类实现各自的svm_struct_api 但是都要继承该基类
 * 
 * @author zkyz
 */

public abstract class SVMStructApi {

	/**
	 * 初始化 svm struct model
	 * 
	 * @param sample
	 * @param sm
	 * @param sparm
	 * @param lparm
	 * @param kparm
	 */
	public abstract void init_struct_model(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm);

	/**
	 * 初始化 struct constraints
	 * 
	 * @param sample
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public CONSTSET init_struct_constraints(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {

		CONSTSET c = new CONSTSET();
		int sizePsi = sm.sizePsi;
		int i;
		WORD[] words = new WORD[2];

		if (true) { /* normal case: start with empty set of constraints */
			c.lhs = null;
			c.rhs = null;
			c.m = 0;
		}

		return (c);
	}

	/**
	 * <word, cate_info> 映射为特征
	 * 
	 * @param x
	 * @param y
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm);

	/**
	 * 在主循环中判段是否终止迭代
	 * 
	 * @param ceps
	 * @param cached_constraint
	 * @param sample
	 * @param sm
	 * @param cset
	 * @param alpha
	 * @param sparm
	 * @return
	 */
	public boolean finalize_iteration(double ceps, int cached_constraint,
			SAMPLE sample, STRUCTMODEL sm, CONSTSET cset, double[] alpha,
			STRUCT_LEARN_PARM sparm) {

		return false;
	}

	/**
	 * 找出样本<x,y> 损失最大的 y'。即loss(<x,y>,<x,y'>)最大 ，损失函数类型是slackrescaling
	 * 
	 * @param x
	 * @param y
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract LABEL find_most_violated_constraint_slackrescaling(
			PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	/**
	 * 找出样本<x,y> 损失最大的 y'。即loss(<x,y>,<x,y'>)最大 ，损失函数类型是marginrescaling
	 * 
	 * @param x
	 * @param y
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract LABEL find_most_violated_constraint_marginrescaling(
			PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	/**
	 * 定义y与y'的损失
	 * 
	 * @param y
	 * @param ybar
	 * @param sparm
	 * @return
	 */
	public abstract double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM sparm);

	/**
	 * 判断label是否为空
	 * 
	 * @param y
	 * @return
	 */
	public abstract boolean empty_label(LABEL y);

	/**
	 * 压缩支持向量
	 * 
	 * @param sample
	 * @param sm
	 * @param cset
	 * @param alpha
	 * @param sparm
	 */
	public void print_struct_learning_stats(SAMPLE sample, STRUCTMODEL sm,
			CONSTSET cset, double[] alpha, STRUCT_LEARN_PARM sparm) {

		/* Replace SV with single weight vector */
		/*******************
		 * MODEL model=sm.svm_model.copyMODEL();
		 ******************/

		MODEL model = sm.svm_model;
		if (model.kernel_parm.kernel_type == ModelConstant.LINEAR) {

			sm.svm_model = SVMCommon.compact_linear_model(model);
			sm.w = sm.svm_model.lin_weights; /* short cut to weight vector */

		}
	}

	/**
	 * 初始化svm_struct
	 * 
	 * @param args
	 */
	public void svm_struct_learn_api_init(String[] args) {

	}

	public void print_struct_help() {

	}

	/**
	 * Parses the command line parameters that start with -- for the
	 * classification module
	 * 
	 * @param sparm
	 */
	public void parse_struct_parameters_classify(STRUCT_LEARN_PARM sparm) {
		int i;

		for (i = 0; (i < sparm.custom_argc)
				&& ((sparm.custom_argv[i]).charAt(0) == '-'); i++) {
			switch ((sparm.custom_argv[i]).charAt(2)) {
			/* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
			default:
				System.out.print("\nUnrecognized option "
						+ sparm.custom_argv[i] + "!\n\n");
				System.exit(0);
			}
		}
	}

	/**
	 * Parses the command line parameters that start with --
	 * 
	 * @param sparm
	 */
	public void parse_struct_parameters(STRUCT_LEARN_PARM sparm) {
		int i;

		for (i = 0; (i < sparm.custom_argc)
				&& ((sparm.custom_argv[i]).charAt(0) == '-'); i++) {
			switch ((sparm.custom_argv[i]).charAt(2)) {
			case 'a':
				i++; /* strcpy(learn_parm->alphafile,argv[i]); */
				break;
			case 'e':
				i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */
				break;
			case 'k':
				i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */
				break;
			}
		}
	}

	/**
	 * 读取svm struct 样本
	 * 
	 * @param file
	 * @param sparm
	 * @return
	 */
	public abstract SAMPLE read_struct_examples(String file,
			STRUCT_LEARN_PARM sparm);

	/*
	 * public static SAMPLE read_struct_examples(String file, STRUCT_LEARN_PARM
	 * sparm) {
	 * 
	 * SAMPLE sample = new SAMPLE(); EXAMPLE[] examples; int n; DOC[] docs;
	 * double[] target = null; int totwords, i, num_classes = 0;
	 * 
	 * docs = svm_common.read_documents(file, target);
	 * 
	 * target = svm_common.read_target; totwords = svm_common.read_totwords; n =
	 * svm_common.read_totdocs;
	 * 
	 * for (int k = 0; k < docs.length; k++) { if (docs[k] == null ||
	 * docs[k].fvec == null) { continue; } } examples = new EXAMPLE[n]; for (int
	 * k = 0; k < n; k++) { examples[k] = new EXAMPLE(); examples[k].x = new
	 * PATTERN(); examples[k].y = new LABEL(); }
	 * 
	 * for (i = 0; i < n; i++) if (num_classes < (target[i] + 0.1)) num_classes
	 * = (int) (target[i] + 0.1); for (i = 0; i < n; i++) if (target[i] < 1) {
	 * System.exit(1); }
	 * 
	 * for (i = 0; i < n; i++) { examples[i].x.doc = docs[i];
	 * examples[i].y.class_index = (int) (target[i] + 0.1); examples[i].y.scores
	 * = null; examples[i].y.num_classes = num_classes; }
	 * 
	 * sample.n = n; sample.examples = examples;
	 * 
	 * return (sample); }
	 */

	/**
	 * 从标准输入读取svm struct样本
	 * 
	 * @param is
	 * @param sparm
	 * @return
	 */
	public abstract SAMPLE read_struct_examples_from_stream(InputStream is,
			STRUCT_LEARN_PARM sparm);

	/*
	 * public static SAMPLE read_struct_examples_from_stream(InputStream is,
	 * STRUCT_LEARN_PARM sparm) {
	 * 
	 * SAMPLE sample = new SAMPLE(); EXAMPLE[] examples; int n; DOC[] docs;
	 * double[] target = null; int totwords, i, num_classes = 0;
	 * 
	 * docs = svm_common.read_documents_from_stream(is, target);
	 * 
	 * target = svm_common.read_target; totwords = svm_common.read_totwords; n =
	 * svm_common.read_totdocs; for (int k = 0; k < docs.length; k++) { if
	 * (docs[k] == null || docs[k].fvec == null) { continue; } } examples = new
	 * EXAMPLE[n]; for (int k = 0; k < n; k++) { examples[k] = new EXAMPLE();
	 * examples[k].x = new PATTERN(); examples[k].y = new LABEL(); }
	 * 
	 * for (i = 0; i < n; i++) if (num_classes < (target[i] + 0.1)) num_classes
	 * = (int) (target[i] + 0.1); for (i = 0; i < n; i++) if (target[i] < 1) {
	 * System.exit(1); }
	 * 
	 * for (i = 0; i < n; i++) { examples[i].x.doc = docs[i];
	 * examples[i].y.class_index = (int) (target[i] + 0.1); examples[i].y.scores
	 * = null; examples[i].y.num_classes = num_classes; }
	 * 
	 * sample.n = n; sample.examples = examples;
	 * 
	 * return (sample); }
	 */

	/**
	 * 从arraylist读取svm struct样本
	 * 
	 * @param is
	 * @param sparm
	 * @return
	 */
	public abstract SAMPLE read_struct_examples_from_arraylist(
			ArrayList<String> list, STRUCT_LEARN_PARM sparm);

	/**
	 * 写svm struct模型到文件
	 * 
	 * @param file
	 * @param sm
	 * @param sparm
	 */
	public void write_struct_model(String file, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		try {
			/* Writes structural model sm to file file. */
			FileWriter fw = new FileWriter(new File(file));
			PrintWriter modelfl = new PrintWriter(fw);
			int j, i, sv_num;
			MODEL model = sm.svm_model.copyMODEL();
			SVECTOR v;

			modelfl.print("SVM-multiclass Version "
					+ SVMStructCommon.INST_VERSION + "\n");
			modelfl.print(sparm.num_classes + "# number of classes\n");
			modelfl.print(sparm.num_features + "# number of base features\n");
			modelfl.print(sparm.loss_function + " # loss function\n");
			modelfl.print(model.kernel_parm.kernel_type + " # kernel type\n");
			modelfl.print(model.kernel_parm.poly_degree
					+ " # kernel parameter -d \n");
			modelfl.print(model.kernel_parm.rbf_gamma
					+ " # kernel parameter -g \n");
			modelfl.print(model.kernel_parm.coef_lin
					+ " # kernel parameter -s \n");
			modelfl.print(model.kernel_parm.coef_const
					+ " # kernel parameter -r \n");
			modelfl.print(model.kernel_parm.custom
					+ " # kernel parameter -u \n");
			modelfl.print(model.totwords + " # highest feature index \n");
			modelfl.print(model.totdoc + " # number of training documents \n");

			sv_num = 1;
			for (i = 1; i < model.sv_num; i++) {
				for (v = model.supvec[i].fvec; v != null; v = v.next)
					sv_num++;
			}
			modelfl.print(sv_num + " # number of support vectors plus 1 \n");
			modelfl.print(model.b
					+ " # threshold b, each following line is a SV (starting with alpha*y)\n");

			for (i = 1; i < model.sv_num; i++) {
				for (v = model.supvec[i].fvec; v != null; v = v.next) {

					modelfl.print((model.alpha[i] * v.factor) + " ");
					modelfl.print("qid:" + v.kernel_id + " ");
					// logger.info("i="+i+" v.length:"+v.words.length);
					for (j = 0; j < v.words.length; j++) {
						modelfl.print((v.words[j]).wnum + ":"
								+ (double) (v.words[j]).weight + " ");
					}
					if (v.userdefined != null)
						modelfl.print("#" + v.userdefined + "\n");
					else
						modelfl.print("#\n");
				}
			}
			modelfl.close();

		} catch (Exception e) {
		}
	}

	public void svm_struct_classify_api_init(int argc, String[] args) {

	}

	public void print_struct_help_classify() {

	}

	/**
	 * 读取svm struct 模型
	 * 
	 * @param file
	 * @param sparm
	 * @return
	 */
	public STRUCTMODEL read_struct_model(String file, STRUCT_LEARN_PARM sparm) {

		File modelfl;
		STRUCTMODEL sm = new STRUCTMODEL();
		int i, queryid, slackid;
		double costfactor;
		int max_sv, max_words, ll, wpos;
		String line, comment;
		WORD[] words;
		String version_buffer;
		MODEL model;
		SVMCommon sc = new SVMCommon();

		ReadSummary summary = null;
		summary = sc.nol_ll(file); /* scan size of model file */
		max_sv = summary.read_max_docs;
		max_words = summary.read_max_words_doc;
		max_words += 2;

		words = new WORD[max_words + 10];
		line = "";
		model = new MODEL();
		model.kernel_parm = new KERNEL_PARM();

		FileReader fr = null;
		BufferedReader br = null;
		try {
			try {
				modelfl = new File(file);
				fr = new FileReader(modelfl);
				br = new BufferedReader(fr);
			} catch (FileNotFoundException e2) {
				InputStream model_is = SVMCommon.class.getResourceAsStream("/"
						+ file);
				InputStreamReader model_isr = new InputStreamReader(model_is);
				br = new BufferedReader(model_isr);
			}

			line = br.readLine();
			// logger.info("line:"+line);
			version_buffer = SSO.afterStr(line, "SVM-multiclass Version")
					.trim();

			if (!(version_buffer.equals(SVMStructCommon.INST_VERSION))) {
				System.err
						.println("Version of model-file does not match version of svm_struct_classify!");

			}
			line = br.readLine();
			// System.err.println("model line:"+line);
			sparm.num_classes = Integer.parseInt(SSO.beforeStr(line, "#")
					.trim());
			line = br.readLine();
			// System.err.println("model line:"+line);
			sparm.num_features = Integer.parseInt(SSO.beforeStr(line, "#")
					.trim());
			line = br.readLine();
			// System.err.println("model line:"+line);
			// System.out.println("line:"+line);
			sparm.loss_function = Integer.parseInt(SSO.beforeStr(line, "#")
					.trim());

			line = br.readLine();
			// System.err.println("model line:"+line);
			// System.out.println("line:"+line);
			model.kernel_parm.kernel_type = Short.parseShort(SSO.beforeStr(
					line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			// System.out.println("line:"+line);
			model.kernel_parm.poly_degree = Integer.parseInt(SSO.beforeStr(
					line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.rbf_gamma = Double.parseDouble(SSO.beforeStr(
					line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.coef_lin = Double.parseDouble(SSO.beforeStr(line,
					"#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.coef_const = Double.parseDouble(SSO.beforeStr(
					line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.custom = SSO.beforeStr(line, "#");

			line = br.readLine();
			// System.err.println("line:"+line);
			model.totwords = Integer.parseInt(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.totdoc = Integer.parseInt(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.sv_num = Integer.parseInt(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.b = Double.parseDouble(SSO.beforeStr(line, "#").trim());

			model.supvec = new DOC[model.sv_num];
			model.alpha = new double[model.sv_num];
			model.index = null;
			model.lin_weights = null;

			WORD[] read_words = null;
			for (i = 1; i < model.sv_num; i++) {
				// System.err.println("sv:"+i);
				line = br.readLine();
				ReadStruct rs = new ReadStruct();
				read_words = sc.parse_document(line, max_words, rs);
				model.alpha[i] = rs.read_doc_label;
				queryid = rs.read_queryid;
				slackid = rs.read_slackid;
				costfactor = rs.read_costfactor;
				wpos = rs.read_wpos;
				comment = rs.read_comment;
				// words = sc.read_words;
				words = read_words;
				// System.out.println("words:" + words.length);
				// System.out.println("queryid:" + queryid);
				model.supvec[i] = sc.create_example(-1, 0, 0, 0.0,
						sc.create_svector(words, comment, 1.0));
				model.supvec[i].fvec.kernel_id = queryid;
				// System.err.println("read supvec["+i+"]:"+model.supvec[i].fvec.toString());
			}
			System.err.println("read model done:");
			if (fr != null) {
				fr.close();
			}
			br.close();

			// if (svm_common.verbosity >= 1) {
			// System.out.println(" (" + (model.sv_num - 1)
			// + " support vectors read) ");
			// }
			// logger.info("kernel type here:"+model.kernel_parm.kernel_type);
			sm.svm_model = model;
			sm.sizePsi = model.totwords;
			sm.w = null;
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
		}
		// System.out.println("reading done");
		return (sm);
	}

	/**
	 * svm struct分类
	 * 
	 * @param x
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract LABEL classify_struct_example(PATTERN x, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm);

	public void write_label(PrintWriter fp, LABEL y) {
		int i;
		fp.print(y.class_index + " ");
		if (y.scores != null) {
			for (i = 1; i < y.num_classes; i++) {
				fp.print(y.scores[i] + " ");
			}
		}
		fp.println();
	}

	public abstract void write_label(PrintWriter fp, LABEL y, LABEL ybar);

	public void eval_prediction(int exnum, EXAMPLE ex, LABEL ypred,
			STRUCTMODEL sm, STRUCT_LEARN_PARM sparm, STRUCT_TEST_STATS teststats) {
		if (exnum == 0) {
			// this is the first time the function is called. So
			// initialize the teststats

		}
	}

	public void print_struct_testing_stats(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, STRUCT_TEST_STATS teststats) {

	}

	public void svm_struct_learn_api_exit() {

	}

	public abstract PATTERN sample2pattern(String words);

	public WORD[] string2words(String sample) {
		WORD[] words = null;

		String[] sample_arr = sample.split("\\s+");
		words = new WORD[sample_arr.length];
		for (int i = 0; i < words.length; i++) {
			words[i] = new WORD();
		}

		String temp_token = "";
		int temp_index = 0;
		double temp_weight = 0.0;

		for (int i = 0; i < sample_arr.length; i++) {
			// System.out.println(i+" "+sample_arr[i]);
			temp_token = sample_arr[i];
			if (Pattern.matches("\\d+:[\\d\\.]+", temp_token)) {
				temp_index = Integer.parseInt(temp_token.substring(0,
						temp_token.indexOf(":")));
				temp_weight = Double.parseDouble(temp_token.substring(
						temp_token.indexOf(":") + 1, temp_token.length()));
				words[i].wnum = temp_index;
				words[i].weight = temp_weight;
			}
		}

		return words;
	}

}
