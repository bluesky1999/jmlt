package org.click.classify.svmstruct.model;

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

import org.click.classify.svmstruct.data.CONSTSET;
import org.click.classify.svmstruct.data.DOC;
import org.click.classify.svmstruct.data.EXAMPLE;
import org.click.classify.svmstruct.data.KERNEL_PARM;
import org.click.classify.svmstruct.data.LABEL;
import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.MODEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.PATTERN;
import org.click.classify.svmstruct.data.ReadStruct;
import org.click.classify.svmstruct.data.ReadSummary;
import org.click.classify.svmstruct.data.SAMPLE;
import org.click.classify.svmstruct.data.STRUCTMODEL;
import org.click.classify.svmstruct.data.STRUCT_LEARN_PARM;
import org.click.classify.svmstruct.data.SVECTOR;
import org.click.classify.svmstruct.data.WORD;
import org.click.lib.string.SSO;

/**
 * svm struct api 的抽象类 不同的分类模型，如多类、多层分类实现各自的svm_struct_api 但是都要继承该基类
 * 
 * @author zkyz
 */

public abstract class Struct {

	public Common com = null;

	public Struct() {
		com = new Common();
	}

	/**
	 * 初始化 svm struct model
	 * 
	 * @param sample
	 * @param sm
	 * @param sparm
	 * @param lparm
	 * @param kparm
	 */
	public abstract void initStructModel(SAMPLE sample, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm);

	/**
	 * 初始化 struct constraints
	 * 
	 * @param sample
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public CONSTSET initStructConstraints(SAMPLE sample, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {

		CONSTSET c = new CONSTSET();
		if (true) { // normal case: start with empty set of constraints 
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
	public abstract SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	/**
	 * 找出样本<x,y> 损失最大的 y'。即loss(<x,y>,<x,y'>)最大 ，损失函数类型是slackrescaling
	 * 
	 * @param x
	 * @param y
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract LABEL findMostViolatedConstraintSlackrescaling(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	/**
	 * 找出样本<x,y> 损失最大的 y'。即loss(<x,y>,<x,y'>)最大 ，损失函数类型是marginrescaling
	 * 
	 * @param x
	 * @param y
	 * @param sm
	 * @param sparm
	 * @return
	 */
	public abstract LABEL findMostViolatedConstraintMarginrescaling(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

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
	public abstract boolean emptyLabel(LABEL y);

	/**
	 * 压缩支持向量
	 * 
	 * @param sample
	 * @param sm
	 * @param cset
	 * @param alpha
	 * @param sparm
	 */
	public void printStructLearningStats(SAMPLE sample, STRUCTMODEL sm, CONSTSET cset, double[] alpha, STRUCT_LEARN_PARM sparm) {

		// Replace SV with single weight vector 
		MODEL model = sm.svm_model;
		if (model.kernel_parm.kernel_type == ModelConstant.LINEAR) {

			sm.svm_model = com.compactLinearModel(model);
			sm.w = sm.svm_model.lin_weights; // short cut to weight vector 
		}
	}

	
	/**
	 * 读取svm struct 样本
	 * 
	 * @param file
	 * @param sparm
	 * @return
	 */
	public abstract SAMPLE readStructExamples(String file, STRUCT_LEARN_PARM sparm);

	/**
	 * 写svm struct模型到文件
	 * 
	 * @param file
	 * @param sm
	 * @param sparm
	 */
	public void writeStructModel(String file, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		try {
			// Writes structural model sm to file file.
			FileWriter fw = new FileWriter(new File(file));
			PrintWriter modelfl = new PrintWriter(fw);
			int j, i, sv_num;
			///MODEL model = sm.svm_model.copyMODEL();
			MODEL model = sm.svm_model;
			SVECTOR v;

			modelfl.print("SVM-multiclass Version " + CommonStruct.INST_VERSION + "\n");
			modelfl.print(sparm.num_classes + "# number of classes\n");
			modelfl.print(sparm.num_features + "# number of base features\n");
			modelfl.print(sparm.loss_function + " # loss function\n");
			modelfl.print(model.kernel_parm.kernel_type + " # kernel type\n");
			modelfl.print(model.kernel_parm.poly_degree + " # kernel parameter -d \n");
			modelfl.print(model.kernel_parm.rbf_gamma + " # kernel parameter -g \n");
			modelfl.print(model.kernel_parm.coef_lin + " # kernel parameter -s \n");
			modelfl.print(model.kernel_parm.coef_const + " # kernel parameter -r \n");
			modelfl.print(model.kernel_parm.custom + " # kernel parameter -u \n");
			modelfl.print(model.totwords + " # highest feature index \n");
			modelfl.print(model.totdoc + " # number of training documents \n");

			sv_num = 1;
			for (i = 1; i < model.sv_num; i++) {
				for (v = model.supvec[i].fvec; v != null; v = v.next)
					sv_num++;
			}
			modelfl.print(sv_num + " # number of support vectors plus 1 \n");
			modelfl.print(model.b + " # threshold b, each following line is a SV (starting with alpha*y)\n");

			for (i = 1; i < model.sv_num; i++) {
				for (v = model.supvec[i].fvec; v != null; v = v.next) {

					modelfl.print((model.alpha[i] * v.factor) + " ");
					modelfl.print("qid:" + v.kernel_id + " ");
					for (j = 0; j < v.words.length; j++) {
						modelfl.print((v.words[j]).wnum + ":" + (double) (v.words[j]).weight + " ");
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


	/**
	 * 读取svm struct 模型
	 * 
	 * @param file
	 * @param sparm
	 * @return
	 */
	public STRUCTMODEL readStructModel(String file, STRUCT_LEARN_PARM sparm) {

		File modelfl;
		STRUCTMODEL sm = new STRUCTMODEL();
		int i, queryid;
		int max_words;
		String line, comment;
		WORD[] words;
		String version_buffer;
		MODEL model;
		Common sc = new Common();

		ReadSummary summary = null;
		summary = sc.nol_ll(file); /* scan size of model file */
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
				InputStream model_is = Common.class.getResourceAsStream("/" + file);
				InputStreamReader model_isr = new InputStreamReader(model_is);
				br = new BufferedReader(model_isr);
			}

			line = br.readLine();
			// logger.info("line:"+line);
			version_buffer = SSO.afterStr(line, "SVM-multiclass Version").trim();

			if (!(version_buffer.equals(CommonStruct.INST_VERSION))) {
				System.err.println("Version of model-file does not match version of svm_struct_classify!");

			}
			line = br.readLine();
			// System.err.println("model line:"+line);
			sparm.num_classes = Integer.parseInt(SSO.beforeStr(line, "#").trim());
			line = br.readLine();
			// System.err.println("model line:"+line);
			sparm.num_features = Integer.parseInt(SSO.beforeStr(line, "#").trim());
			line = br.readLine();
			// System.err.println("model line:"+line);
			// System.out.println("line:"+line);
			sparm.loss_function = Integer.parseInt(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("model line:"+line);
			// System.out.println("line:"+line);
			model.kernel_parm.kernel_type = Short.parseShort(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			// System.out.println("line:"+line);
			model.kernel_parm.poly_degree = Integer.parseInt(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.rbf_gamma = Double.parseDouble(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.coef_lin = Double.parseDouble(SSO.beforeStr(line, "#").trim());

			line = br.readLine();
			// System.err.println("line:"+line);
			model.kernel_parm.coef_const = Double.parseDouble(SSO.beforeStr(line, "#").trim());

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
				read_words = sc.parseDocument(line, max_words, rs);
				model.alpha[i] = rs.read_doc_label;
				queryid = rs.read_queryid;
				comment = rs.read_comment;
				// words = sc.read_words;
				words = read_words;
				model.supvec[i] = sc.createExample(-1, 0, 0, 0.0, sc.createSvector(words, comment, 1.0));
				model.supvec[i].fvec.kernel_id = queryid;
				// System.err.println("read supvec["+i+"]:"+model.supvec[i].fvec.toString());
			}
			System.err.println("read model done:");
			if (fr != null) {
				fr.close();
			}
			br.close();

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
	public abstract LABEL classifyStructExample(PATTERN x, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	public abstract LABEL classifyStructDoc(DOC d, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm);

	public void writeLabel(PrintWriter fp, LABEL y) {
		int i;
		fp.print(y.class_index + " ");
		if (y.scores != null) {
			for (i = 1; i < y.num_classes; i++) {
				fp.print(y.scores[i] + " ");
			}
		}
		fp.println();
	}

	public abstract void writeLabel(PrintWriter fp, LABEL y, LABEL ybar);

	public abstract PATTERN sample2pattern(String words);

	public abstract DOC sample2doc(String words);

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
			temp_token = sample_arr[i];
			if (Pattern.matches("\\d+:[\\d\\.]+", temp_token)) {
				temp_index = Integer.parseInt(temp_token.substring(0, temp_token.indexOf(":")));
				temp_weight = Double.parseDouble(temp_token.substring(temp_token.indexOf(":") + 1, temp_token.length()));
				words[i].wnum = temp_index;
				words[i].weight = temp_weight;
			}
		}

		return words;
	}

}
