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
import org.click.classify.svmstruct.data.KERNEL_PARM;
import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.MATRIX;
import org.click.classify.svmstruct.data.MODEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.ReadStruct;
import org.click.classify.svmstruct.data.ReadSummary;
import org.click.classify.svmstruct.data.SVECTOR;
import org.click.classify.svmstruct.data.SortWordArr;
import org.click.classify.svmstruct.data.WORD;
import org.click.lib.string.SSO;
import org.click.lib.time.TimeOpera;

public class Common {

	public SVECTOR createSvector(WORD[] words, String userdefined, double factor) {
		SVECTOR vec;
		int i;

		vec = new SVECTOR();
		vec.words = new WORD[words.length];

		for (i = 0; i < words.length; i++) {
			vec.words[i] = words[i];
		}

		vec.twonorm_sq = -1;

		if (userdefined != null) {
			vec.userdefined = userdefined;
		} else {
			vec.userdefined = null;
		}

		vec.kernel_id = 0;
		vec.next = null;
		vec.factor = factor;
		return vec;
	}

	public DOC createExample(int docnum, int queryid, int slackid, double costfactor, SVECTOR fvec) {

		DOC example = new DOC();

		example.docnum = docnum;
		example.kernelid = docnum;
		example.queryid = queryid;
		example.slackid = slackid;
		example.costfactor = costfactor;

		example.fvec = fvec;

		return example;
	}

	public double kernel(KERNEL_PARM kernel_parm, DOC a, DOC b) {
		double sum = 0;
		SVECTOR fa, fb;
		if (kernel_parm.kernel_type == ModelConstant.GRAM) {
			if ((a.kernelid >= 0) && (b.kernelid >= 0)) {

				return kernel_parm.gram_matrix.element[Math.max(a.kernelid, b.kernelid)][Math.min(a.kernelid, b.kernelid)];
			} else {
				return 0;
			}
		}
		for (fa = a.fvec; fa != null; fa = fa.next) {
			for (fb = b.fvec; fb != null; fb = fb.next) {

				if (fa.kernel_id == fb.kernel_id) {
					sum += fa.factor * fb.factor * singleKernel(kernel_parm, fa, fb);
				}
			}
		}

		return sum;
	}

	public double singleKernel(KERNEL_PARM kernel_parm, SVECTOR a, SVECTOR b) {

		switch (kernel_parm.kernel_type) {
		case ModelConstant.LINEAR:
			return sprodSs(a, b);
		case ModelConstant.POLY:
			return Math.pow(kernel_parm.coef_lin * sprodSs(a, b) + kernel_parm.coef_const, kernel_parm.poly_degree);
		case ModelConstant.RBF:
			if (a.twonorm_sq < 0) {
				a.twonorm_sq = sprodSs(a, a);
			} else if (b.twonorm_sq < 0) {
				b.twonorm_sq = sprodSs(b, b);
			}
			return Math.exp(-kernel_parm.rbf_gamma * (a.twonorm_sq - 2 * sprodSs(a, b) + b.twonorm_sq));
		case ModelConstant.SIGMOID:
			return Math.tanh(kernel_parm.coef_lin * sprodSs(a, b) + kernel_parm.coef_const);
		case ModelConstant.CUSTOM:
			// return kernel.custom_kernel(kernel_parm, a, b);
		default:
			System.out.println("Error: Unknown kernel function");
			System.exit(1);

		}

		return 0;
	}

	public double sprodSs(SVECTOR a, SVECTOR b) {
		double sum = 0;
		WORD[] ai, bj;
		ai = a.words;
		bj = b.words;

		int i = 0;
		int j = 0;

		while ((i < ai.length) && (j < bj.length)) {
			if (ai[i] == null || bj[j] == null) {
				break;
			}
			if (ai[i].wnum > bj[j].wnum) {
				j++;
			} else if (ai[i].wnum < bj[j].wnum) {
				i++;
			} else {

				sum += ai[i].weight * bj[j].weight;
				i++;
				j++;
			}
		}

		return sum;
	}

	public void clearNvector(double[] vec, int n) {
		int i;
		for (i = 0; i <= n; i++) {
			vec[i] = 0;
		}
	}

	public double[] createNvector(int n) {
		double[] vector;
		vector = new double[n + 1];
		return vector;
	}

	public void addVectorNs(double[] vec_n, SVECTOR vec_s, double faktor) {
		WORD[] ai;
		ai = vec_s.words;

		//System.err.println("vec_n.len:"+vec_n.length);
		for (int i = 0; i < ai.length; i++) {
			if (ai[i] != null) {
				vec_n[ai[i].wnum] += (faktor * ai[i].weight);
			} else {
				continue;
			}
		}
	}

	public double sprodNs(double[] vec_n, SVECTOR vec_s) {
		double sum = 0;
		WORD[] ai;
		ai = vec_s.words;

		//System.err.println("vec_n.length:"+vec_n.length);
		for (int i = 0; i < ai.length; i++) {
			if (ai[i] != null && ai[i].wnum >= 0) {

				sum += (vec_n[ai[i].wnum] * ai[i].weight);
			} else {
				continue;
			}
		}

		return sum;
	}

	public void multVectorNs(double[] vec_n, SVECTOR vec_s, double faktor) {
		WORD[] ai;
		ai = vec_s.words;
		for (int i = 0; i < ai.length; i++) {
			if (ai[i] == null) {
				continue;
			}
			vec_n[ai[i].wnum] *= (faktor * ai[i].weight);
		}

	}

	public double getRuntime() {
		int c = (int) TimeOpera.getCurrentTimeLong();
		double hc = 0;
		hc = ((double) c) * 10;
		return hc;
	}

	public WORD copyWord(WORD w) {
		WORD n = new WORD();
		n.weight = w.weight;
		n.wnum = w.wnum;

		return n;
	}

	public void setLearningDefaults(LEARN_PARM learn_parm, KERNEL_PARM kernel_parm) {
		learn_parm.type = ModelConstant.CLASSIFICATION;
		learn_parm.predfile = "trans_predictions";
		learn_parm.alphafile = "";
		learn_parm.biased_hyperplane = 1;
		learn_parm.sharedslack = 0;
		learn_parm.remove_inconsistent = 0;
		learn_parm.svm_maxqpsize = 10;
		// learn_parm.svm_maxqpsize = 100;
		learn_parm.svm_newvarsinqp = 0;
		// learn_parm.maxiter = 100000;
		learn_parm.maxiter = 100000;

		// learn_parm.kernel_cache_size = 400;
		learn_parm.svm_c = 0.0;
		learn_parm.eps = 0.1;
		learn_parm.svm_costratio = 1.0;
		learn_parm.epsilon_crit = 0.001;
		learn_parm.epsilon_a = 1E-15;
		kernel_parm.kernel_type = ModelConstant.LINEAR;
		kernel_parm.poly_degree = 3;
		kernel_parm.rbf_gamma = 1.0;
		kernel_parm.coef_lin = 1;
		kernel_parm.coef_const = 1;
		kernel_parm.custom = "empty";
	}

	public DOC[] readDocuments(String docfile, ReadStruct struct) {

		String line;
		DOC[] docs;
		int dnum = 0;
		FileReader fr = null;
		BufferedReader br = null;

		if (CommonStruct.verbosity >= 1) {
			System.out.println("Scanning examples...");
		}

		ReadSummary summary = nol_ll(docfile); // scan size of input file
		struct.read_max_words_doc = summary.read_max_words_doc + 2;
		struct.read_max_docs = summary.read_max_docs + 2;

		try {
			fr = new FileReader(new File(docfile));
			br = new BufferedReader(fr);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		docs = new DOC[struct.read_max_docs];

		WORD[] words;

		double[] label = new double[struct.read_max_docs]; // target values

		words = new WORD[struct.read_max_words_doc + 10];
		for (int j = 0; j < words.length; j++) {
			words[j] = new WORD();
			words[j].wnum = 0;
			words[j].weight = 0;
		}

		if (CommonStruct.verbosity >= 1) {
			System.out.println("Reading examples into memory...");
		}

		dnum = 0;
		struct.read_totwords = 0;
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.charAt(0) == '#')
					continue; // line contains comments

				ReadStruct rs = new ReadStruct();
				if ((words = parseDocument(line, struct.read_max_words_doc, rs)) == null) {
					System.out.println("\nParsing error in line " + dnum + "!\n" + line);
					// System.exit(1);
					continue;
				}
				label[dnum] = rs.read_doc_label;

				if ((rs.read_wpos > 1) && ((words[rs.read_wpos - 2]).wnum > rs.read_totwords))
					struct.read_totwords = words[rs.read_wpos - 2].wnum;

				docs[dnum] = createExample(dnum, rs.read_queryid, rs.read_slackid, rs.read_costfactor, createSvector(words, rs.read_comment, 1.0));
				dnum++;
			}

			fr.close();
			br.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		if (CommonStruct.verbosity >= 1) {
			System.out.println("OK. (" + dnum + " examples read)\n");
		}
		struct.read_totdocs = dnum;
		struct.read_target = label;
		return docs;
	}

	public WORD[] parseDocument(String line, int max_words_doc, ReadStruct struct) {
		int wpos = 0;

		if (SSO.tioe(line)) {
			return null;
		}

		ArrayList<WORD> wlist = new ArrayList<WORD>();

		struct.read_queryid = 0;
		struct.read_slackid = 0;
		struct.read_costfactor = 1;
		struct.read_comment = "";
		String dline = "";

		if (line.indexOf("#") > 0) {
			struct.read_comment = line.substring(line.indexOf("#") + 1, line.length());
			dline = line.substring(0, line.indexOf("#"));
		} else {
			dline = line;
		}

		dline = dline.trim();
		wpos = 0;
		String[] seg_arr = dline.split(" ");
		if ((seg_arr.length < 1) || (seg_arr[0].indexOf("#") > -1)) {
			return null;
		}
		struct.read_doc_label = Double.parseDouble(seg_arr[0]);

		String wstr = "";
		String pstr = "";
		String sstr = "";
		for (int i = 1; i < seg_arr.length; i++) {

			wstr = seg_arr[i].trim();
			if (wstr.indexOf(":") < 0) {
				continue;
			}

			pstr = wstr.substring(0, wstr.indexOf(":"));
			sstr = wstr.substring(wstr.indexOf(":") + 1, wstr.length());
			pstr = pstr.trim();
			sstr = sstr.trim();
			if (pstr.equals("qid")) {
				struct.read_queryid = Integer.parseInt(sstr);
			} else if (pstr.equals("sid")) {
				struct.read_slackid = Integer.parseInt(sstr);
			} else if (pstr.equals("cost")) {
				struct.read_costfactor = Double.parseDouble(sstr);
			} else if (Pattern.matches("[\\d]+", pstr)) {
				WORD w = new WORD();
				w.wnum = Integer.parseInt(pstr);
				w.weight = Double.parseDouble(sstr);
				wlist.add(w);
				wpos++;
			}
		}

		WORD[] read_words = new WORD[wlist.size()];
		for (int i = 0; i < wlist.size(); i++) {
			read_words[i] = wlist.get(i);
		}

		struct.read_wpos = wpos;

		return read_words;
	}

	public ReadSummary nol_ll(String input_file) {

		ReadSummary summary = new ReadSummary();
		BufferedReader br = null;

		try {

			FileReader fr = null;
			fr = new FileReader(new File(input_file));
			br = new BufferedReader(fr);

		} catch (FileNotFoundException e) {
			// e.printStackTrace();
			InputStream model_is = Common.class.getResourceAsStream("/" + input_file);
			InputStreamReader model_isr = new InputStreamReader(model_is);
			br = new BufferedReader(model_isr);

		}

		String line = "";
		int temp_docs = 0;
		int temp_words = 0;
		String[] seg_arr = null;
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();

				temp_docs++;
				seg_arr = line.split("\\s+");
				if (seg_arr.length > temp_words) {
					temp_words = seg_arr.length;
				}
			}

			summary.read_max_docs = temp_docs;
			summary.read_max_words_doc = temp_words;

			br.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		return summary;

	}

	/**
	 * reads the alpha vector from a file as written by the write_alphas
	 * function
	 */
	public double[] readAlphas(String alphafile, int totdoc) {
		FileReader fr = null;
		BufferedReader br = null;
		double[] alpha = null;
		try {
			fr = new FileReader(new File(alphafile));
			br = new BufferedReader(fr);

			alpha = new double[totdoc];
			int dnum = 0;
			String line = "";

			while ((line = br.readLine()) != null) {
				alpha[dnum] = Double.parseDouble(line);
				dnum++;
			}
			fr.close();
			br.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		return (alpha);
	}

	/****************************** IO routines ***************************/
	public void writeModel(String modelfile, MODEL model) {
		FileWriter fw = null;
		PrintWriter pw = null;

		try {
			fw = new FileWriter(new File(modelfile));
			pw = new PrintWriter(fw);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		int j, i, sv_num;
		SVECTOR v;

		if (CommonStruct.verbosity >= 1) {
			System.out.println("Writing model file...");
		}

		pw.println("SVM-light Version " + ModelConstant.VERSION);
		pw.println(model.kernel_parm.kernel_type + " # kernel type");
		pw.println(model.kernel_parm.poly_degree + " # kernel parameter -d ");
		pw.println(model.kernel_parm.rbf_gamma + " # kernel parameter -g ");
		pw.println(model.kernel_parm.coef_lin + " # kernel parameter -s ");
		pw.println(model.kernel_parm.coef_const + " # kernel parameter -r ");
		pw.println(model.kernel_parm.custom + "# kernel parameter -u ");
		pw.println(model.totwords + " # highest feature index ");
		pw.println(model.totdoc + " # number of training documents ");

		sv_num = 1;
		for (i = 1; i < model.sv_num; i++) {
			for (v = model.supvec[i].fvec; v != null; v = v.next)
				sv_num++;
		}
		pw.println(sv_num + " # number of support vectors plus 1 \n");
		pw.println(model.b + " # threshold b, each following line is a SV (starting with alpha*y)\n");

		for (i = 1; i < model.sv_num; i++) {
			for (v = model.supvec[i].fvec; v != null; v = v.next) {
				pw.print(model.alpha[i] * v.factor + " ");
				for (j = 0; (v.words[j]).wnum != 0; j++) {
					pw.print((int) (v.words[j]).wnum + ":" + (double) (v.words[j]).weight + " ");
				}
				if (v.userdefined != null)
					pw.print("#" + v.userdefined + "\n");
				else
					pw.print("#\n");
			}
		}

		if (CommonStruct.verbosity >= 1) {
			System.out.println("done\n");
		}
	}

	/*
	 * Makes a copy of model where the support vectors are replaced with a
	 * single linear weight vector. NOTE: It adds the linear weight vector also
	 * to newmodel->lin_weights WARNING: This is correct only for linear models!
	 */
	public MODEL compactLinearModel(MODEL model) {
		MODEL newmodel;
		newmodel = new MODEL();
		///newmodel = model.copyMODEL();
		newmodel = model;
		addWeightVectorToLinearModel(newmodel);
		newmodel.supvec = new DOC[2];
		newmodel.alpha = new double[2];
		newmodel.index = null; // index is not copied
		newmodel.supvec[0] = null;
		newmodel.alpha[0] = 0.0;
		newmodel.supvec[1] = createExample(-1, 0, 0, 0, createSvectorN(newmodel.lin_weights, newmodel.totwords, null, 1.0));
		newmodel.alpha[1] = 1.0;
		newmodel.sv_num = 2;

		return (newmodel);
	}

	/** compute weight vector in linear case and add to model */
	public void addWeightVectorToLinearModel(MODEL model) {
		int i;
		SVECTOR f;
		model.lin_weights = createNvector(model.totwords);
		clearNvector(model.lin_weights, model.totwords);
		for (i = 1; i < model.sv_num; i++) {
			for (f = (model.supvec[i]).fvec; f != null; f = f.next)
				addVectorNs(model.lin_weights, f, f.factor * model.alpha[i]);
		}
	}

	public SVECTOR createSvectorN(double[] nonsparsevec, int maxfeatnum, String userdefined, double factor) {
		return (createSvectorNR(nonsparsevec, maxfeatnum, userdefined, factor, 0));
	}

	public SVECTOR createSvectorNR(double[] nonsparsevec, int maxfeatnum, String userdefined, double factor, double min_non_zero) {
		SVECTOR vec;
		int fnum, i;

		fnum = 0;
		for (i = 1; i <= maxfeatnum; i++)
			if ((nonsparsevec[i] < -min_non_zero) || (nonsparsevec[i] > min_non_zero))
				fnum++;

		vec = new SVECTOR();
		vec.words = new WORD[fnum + 1];
		for (int vi = 0; vi < vec.words.length; vi++) {
			vec.words[vi] = new WORD();
		}

		fnum = 0;
		for (i = 1; i <= maxfeatnum; i++) {
			if ((nonsparsevec[i] < -min_non_zero) || (nonsparsevec[i] > min_non_zero)) {
				vec.words[fnum].wnum = i;
				vec.words[fnum].weight = nonsparsevec[i];
				fnum++;
			}
		}
		vec.words[fnum].wnum = 0;
		vec.twonorm_sq = -1;

		if (userdefined != null) {
			vec.userdefined = userdefined;
		} else
			vec.userdefined = null;

		vec.kernel_id = 0;
		vec.next = null;
		vec.factor = factor;
		// //logger.info("end create_svector_n_r");
		return (vec);
	}

	public void copyright_notice() {
		System.out.println("\nCopyright: Thorsten Joachims, thorsten@joachims.org");
		System.out.println("This software is available for non-commercial use only. It must not");
		System.out.println("be modified and distributed without prior permission of the author.");
		System.out.println("The author is not responsible for implications from the use of this");
		System.out.println("software.\n\n");
	}

	public boolean checkLearningParms(LEARN_PARM learn_parm, KERNEL_PARM kernel_parm) {
		System.out.println("check_learning_parms");

		if ((learn_parm.svm_maxqpsize < 2)) {
			System.out.println("\nMaximum size of QP-subproblems not in valid range: " + learn_parm.svm_maxqpsize + " [2..]\n");
			return false;
		}
		if ((learn_parm.svm_maxqpsize < learn_parm.svm_newvarsinqp)) {
			System.out.println("\nMaximum size of QP-subproblems [" + learn_parm.svm_maxqpsize + "] must be larger than the number of\n");
			System.out.println("new variables [" + learn_parm.svm_newvarsinqp + "] entering the working set in each iteration.\n");
			return false;
		}

		if (learn_parm.svm_c < 0) {
			System.out.println("\nThe C parameter must be greater than zero!\n\n");
			return false;
		}

		if (learn_parm.svm_costratio <= 0) {
			System.out.println("\nThe COSTRATIO parameter must be greater than zero!\n\n");
			return false;
		}
		if (learn_parm.epsilon_crit <= 0) {
			System.out.println("\nThe epsilon parameter must be greater than zero!\n\n");
			return false;
		}

		System.out.println("true");
		return true;
	}

	public SVECTOR shiftS(SVECTOR a, int shift) {
		SVECTOR vec;
		WORD[] sumi;
		WORD[] ai;
		int veclength;
		String userdefined = "";
		ai = new WORD[a.words.length];
		for (int k = 0; k < ai.length; k++) {
			ai[k] = a.words[k];
		}

		veclength = ai.length;
		sumi = new WORD[veclength];
		for (int i = 0; i < ai.length; i++) {
			sumi[i] = new WORD();
			sumi[i].weight = ai[i].weight;
			sumi[i].wnum = ai[i].wnum + shift;
		}

		vec = createSvectorShallow(sumi, userdefined, a.factor);
		return vec;
	}

	public SVECTOR createSvectorShallow(WORD[] words, String userdefined, double factor) {
		SVECTOR vec;
		vec = new SVECTOR();
		vec.words = new WORD[words.length];
		for (int i = 0; i < words.length; i++) {
			if (words[i] != null) {
				vec.words[i] = copyWord(words[i]);
			} else {
				vec.words[i] = null;
			}
		}

		vec.twonorm_sq = -1;
		vec.userdefined = userdefined;
		vec.kernel_id = 0;
		vec.next = null;
		vec.factor = factor;

		return vec;
	}

	public SVECTOR addListSs(SVECTOR a) {
		return (addListSsR(a, 0));
	}

	/**
	 * computes the linear combination of the SVECTOR list weighted by the
	 * factor of each SVECTOR
	 */
	public SVECTOR addListSsR(SVECTOR a, double min_non_zero) {
		SVECTOR oldsum, sum, f;
		WORD[] empty = new WORD[2];
		for (int k = 0; k < 2; k++) {
			empty[k] = new WORD();
		}

		if (a == null) {
			empty[0].wnum = 0;
			sum = createSvector(empty, null, 1.0);
		} else if ((a != null) && (a.next == null)) {
			sum = smultS(a, a.factor);
		} else {
			sum = multaddSsR(a, a.next, a.factor, a.next.factor, min_non_zero);
			for (f = a.next.next; f != null; f = f.next) {
				oldsum = sum;
				sum = multaddSsR(oldsum, f, 1.0, f.factor, min_non_zero);
			}
		}
		return (sum);
	}

	/** scale sparse vector a by factor */
	public SVECTOR smultS(SVECTOR a, double factor) {
		SVECTOR vec;
		WORD[] sumi;
		WORD[] ai;
		int veclength;
		String userdefined = null;

		ai = a.words;
		veclength = ai.length;
		sumi = new WORD[veclength];
		ArrayList<WORD> wordlist = new ArrayList<WORD>();
		ai = a.words;
		WORD temp_word = null;
		for (int i = 0; i < veclength; i++) {
			temp_word = ai[i];
			temp_word.weight *= factor;
			if (temp_word.weight != 0) {
				wordlist.add(temp_word);
			}
		}

		sumi = new WORD[wordlist.size()];

		for (int i = 0; i < wordlist.size(); i++) {
			sumi[i] = wordlist.get(i);
		}

		if (a.userdefined != null) {
			userdefined = a.userdefined;
		}

		vec = createSvectorShallow(sumi, userdefined, 1.0);
		return (vec);
	}

	/**
	 * compute fa*a+fb*b of two sparse vectors Note: SVECTOR lists are not
	 * followed, but only the first SVECTOR is used 这个方法有错
	 * 
	 * @param a
	 * @param b
	 * @param fa
	 * @param fb
	 * @param min_non_zero
	 * @return
	 */
	public SVECTOR multaddSsR(SVECTOR a, SVECTOR b, double fa, double fb, double min_non_zero) {
		SVECTOR vec;
		WORD[] sumi;
		WORD[] ai, bj;
		int veclength;
		double weight;

		ai = a.words;
		bj = b.words;
		veclength = 0;
		int i = 0;
		int j = 0;

		while (i < ai.length && j < bj.length) {
			if (ai[i].wnum > bj[j].wnum) {
				veclength++;
				j++;
			} else if (ai[i].wnum < bj[j].wnum) {
				veclength++;
				i++;
			} else {
				veclength++;
				i++;
				j++;
			}
		}

		while (j < bj.length) {
			veclength++;
			j++;
		}

		while (i < ai.length) {
			veclength++;
			i++;
		}
		veclength++;

		sumi = new WORD[veclength];

		ai = a.words;
		bj = b.words;
		i = 0;
		j = 0;
		int s = 0;
		while (i < ai.length && j < bj.length) {
			if (ai[i].wnum > bj[j].wnum) {
				sumi[s] = bj[j];
				sumi[s].weight *= fb;
				if (sumi[s].weight != 0)
					s++;
				j++;
			} else if (ai[i].wnum < bj[j].wnum) {
				sumi[s] = ai[i];
				sumi[s].weight *= fa;
				if (sumi[s].weight != 0)
					s++;
				i++;
			} else {
				weight = fa * (double) ai[i].weight + fb * (double) bj[j].weight;
				if ((weight < -min_non_zero) || (weight > min_non_zero)) {
					sumi[s].wnum = ai[i].wnum;
					sumi[s].weight = weight;
					if (sumi[s].weight != 0)
						s++;
				}
				i++;
				j++;
			}
		}
		while (j < bj.length) {
			sumi[s] = bj[j];
			sumi[s].weight *= fb;
			if (sumi[s].weight != 0)
				s++;
			j++;
		}
		while (i < ai.length) {
			sumi[s] = ai[i];
			sumi[s].weight *= fa;
			if (sumi[s].weight != 0)
				s++;
			i++;
		}

		vec = createSvectorShallow(sumi, null, 1.0);

		return (vec);
	}

	/** classifies one example */
	public double classifyExample(MODEL model, DOC ex) {
		int i;
		double dist;

		if ((model.kernel_parm.kernel_type == ModelConstant.LINEAR) && (model.lin_weights != null)) {
			return (classifyExampleLinear(model, ex));
		}
		dist = 0;
		for (i = 1; i < model.sv_num; i++) {
			dist += kernel(model.kernel_parm, model.supvec[i], ex) * model.alpha[i];
		}
		return (dist - model.b);
	}

	/**
	 * classifies example for linear kernel
	 * 
	 * important: the model must have the linear weight vector computed use:
	 * add_weight_vector_to_linear_model(&model);
	 * 
	 * important: the feature numbers in the example to classify must not be
	 * larger than the weight vector!
	 */
	public double classifyExampleLinear(MODEL model, DOC ex) {
		double sum = 0;
		SVECTOR f;

		for (f = ex.fvec; f != null; f = f.next) {
			sum += f.factor * sprodNs(model.lin_weights, f);
		}
		return (sum - model.b);
	}

	public SVECTOR copySvector(SVECTOR vec) {
		SVECTOR newvec = null;
		if (vec != null) {
			newvec = createSvector(vec.words, vec.userdefined, vec.factor);
			newvec.kernel_id = vec.kernel_id;
			newvec.next = copySvector(vec.next);
		}
		return (newvec);
	}

	/** appends SVECTOR b to the end of SVECTOR a. */
	public void appendSvectorList(SVECTOR a, SVECTOR b) {
		SVECTOR f;

		for (f = a; f.next != null; f = f.next)
			; // find end of first vector
				// list
		f.next = b; // append the two vector lists
	}

	/**
	 * compute the sum a+b of two sparse vectors Note: SVECTOR lists are not
	 * followed, but only the first SVECTOR is used
	 */
	public SVECTOR addSs(SVECTOR a, SVECTOR b) {
		return (multaddSsR(a, b, 1.0, 1.0, 0));
	}

	/** create matrix with n rows and m colums */
	public MATRIX createMatrix(int n, int m) {
		MATRIX matrix;

		matrix = new MATRIX();
		matrix.n = n;
		matrix.m = m;
		matrix.element = new double[n][m];

		return (matrix);
	}

	/**
	 * Like add_list_sort_ss(SVECTOR *a), but rounds values smaller than
	 * min_non_zero to zero.
	 */
	public SVECTOR addListSortSsR(SVECTOR a, double min_non_zero) {
		SVECTOR sum, f;
		WORD[] empty = new WORD[2];
		WORD[] ai, concat, concat_read, concat_write;
		int length, i;
		double weight;

		int cwi = 0;
		int cri = 0;

		if (a != null) {
			// count number or entries over all vectors
			length = 0;
			for (f = a; f != null; f = f.next) {
				ai = f.words;
				for (int k = 0; k > ai.length; k++) {
					length++;
				}
			}

			// write all entries into one long array and sort by feature number
			concat = new WORD[length + 1];
			int s = 0;
			for (f = a; f != null; f = f.next) {
				ai = f.words;
				for (int k = 0; k < ai.length; k++) {
					concat[s] = ai[k];
					concat[s].weight *= f.factor;
					s++;
				}
			}

			concat = SortWordArr.sort_array(concat);

			concat_read = copyWordArr(1, concat);
			concat_write = copyWordArr(0, concat);

			for (i = 0; (i < length - 1) && (concat_write[cwi].wnum != concat_read[cri].wnum); i++) {
				cwi++;
				cri++;
			}

			weight = concat_write[cwi].weight;
			for (; (i < length - 1); i++) {
				if (concat_write[cwi].wnum == concat_read[cri].wnum) {
					weight += (double) concat_read[cri].weight;
					cri++;
				} else {
					if ((weight > min_non_zero) || (weight < -min_non_zero)) {
						concat_write[cwi].weight = weight;
						cwi++;
					}
					concat_write[cwi] = copyWord(concat_read[cri]);// ?是否正确
					weight = concat_write[cwi].weight;
					cri++;
				}
			}

			if ((length > 0) && ((weight > min_non_zero) || (weight < -min_non_zero))) {
				concat_write[cwi].weight = weight;
				cwi++;
			}

			sum = createSvectorShallow(concat, null, 1.0);

		} else {
			empty[0].wnum = 0;
			sum = createSvector(empty, null, 1.0);
		}
		return (sum);
	}

	public WORD[] copyWordArr(int start_index, WORD[] oarr) {
		WORD[] warr = new WORD[oarr.length - start_index];
		if (start_index > (oarr.length - 1)) {
			return null;
		}

		for (int i = start_index; i < oarr.length; i++) {
			warr[i - start_index] = copyWord(oarr[i]);
		}

		return warr;
	}

	public void addListNNS(double[] vec_n, SVECTOR vec_s, double faktor) {
		SVECTOR f;
		for (f = vec_s; f != null; f = f.next)
			addVectorNs(vec_n, f, f.factor * faktor);
	}

	/** multiplies the factor of each element in vector list with factor */
	public void multSvectorList(SVECTOR a, double factor) {
		SVECTOR f;

		for (f = a; f != null; f = f.next)
			f.factor *= factor;
	}

	/**
	 * computes the linear combination of the SVECTOR list weighted by the
	 * factor of each SVECTOR. assumes that the number of features is small
	 * compared to the number of elements in the list
	 */
	public SVECTOR addListNsR(SVECTOR a, double min_non_zero) {
		SVECTOR vec, f;
		WORD[] ai;
		int totwords;
		double[] sum;

		// find max feature number
		totwords = 0;
		for (f = a; f != null; f = f.next) {
			ai = f.words;
			for (int k = 0; k < ai.length; k++) {
				if (totwords < ai[k].wnum)
					totwords = ai[k].wnum;
			}
		}
		sum = createNvector(totwords);

		clearNvector(sum, totwords);
		for (f = a; f != null; f = f.next)
			addVectorNs(sum, f, f.factor);

		vec = createSvectorNR(sum, totwords, null, 1.0, min_non_zero);

		return (vec);
	}

	/**
	 * extends/shrinks matrix to n rows and m colums. Not that added elements
	 * are not initialized.
	 */
	public MATRIX reallocMatrix(MATRIX matrix, int n, int m) {
		if (matrix == null)
			return (createMatrix(n, m));

		matrix.element = reallocMatrixRow(matrix.element, n, matrix.m, m);

		matrix.n = n;
		matrix.m = m;
		return (matrix);
	}

	public double[][] reallocMatrixRow(double[][] ddarr, int n, int old_m, int m) {
		double[][] ndarr = new double[n][m];
		for (int i = 0; i < ddarr.length; i++) {
			for (int j = 0; j < old_m; j++) {
				ndarr[i][j] = ddarr[i][j];
			}
			for (int j = old_m; j < m; j++) {
				ndarr[i][j] = 0;
			}
		}

		for (int i = ddarr.length; i < n; i++) {
			for (int j = 0; j < m; j++) {
				ndarr[i][j] = 0;
			}
		}

		return ndarr;
	}

	public MODEL read_model(String modelfile) {

		MODEL model = new MODEL();
		FileReader fr = null;
		BufferedReader br = null;
		try {

			try {
				fr = new FileReader(modelfile);
				br = new BufferedReader(fr);
			} catch (Exception e2) {
				InputStream model_is = Common.class.getResourceAsStream("/" + modelfile);
				InputStreamReader model_isr = new InputStreamReader(model_is);
				br = new BufferedReader(model_isr);
			}
			int i, queryid;

			int max_words;
			String line, comment;
			WORD[] words;

			ReadSummary summary = nol_ll(modelfile);
			max_words = summary.read_max_words_doc;
			max_words += 2;
			line = br.readLine();

			model.kernel_parm = new KERNEL_PARM();

			line = br.readLine();
			model.kernel_parm.kernel_type = Short.parseShort(SSO.beforeStr(line, "#"));

			System.err.println("model.kernel_parm.kernel_type:" + model.kernel_parm.kernel_type);

			line = br.readLine();
			model.kernel_parm.poly_degree = Integer.parseInt(SSO.beforeStr(line, "#"));
			System.err.println("model.kernel_parm.poly_degree:" + model.kernel_parm.poly_degree);

			line = br.readLine();
			model.kernel_parm.rbf_gamma = Double.parseDouble(SSO.beforeStr(line, "#"));
			System.err.println("model.kernel_parm.rbf_gamma:" + model.kernel_parm.rbf_gamma);

			line = br.readLine();
			model.kernel_parm.coef_lin = Double.parseDouble(SSO.beforeStr(line, "#"));
			System.err.println("model.kernel_parm.coef_lin:" + model.kernel_parm.coef_lin);

			line = br.readLine();
			model.kernel_parm.coef_const = Double.parseDouble(SSO.beforeStr(line, "#"));
			System.err.println("model.kernel_parm.kernel_type:" + model.kernel_parm.kernel_type);

			line = br.readLine();
			model.kernel_parm.custom = line;
			System.err.println("model.kernel_parm.custom:" + model.kernel_parm.custom);

			line = br.readLine();
			model.totwords = Integer.parseInt(SSO.beforeStr(line, "#"));
			System.err.println("model.totwords:" + model.totwords);

			line = br.readLine();
			model.totdoc = Integer.parseInt(SSO.beforeStr(line, "#"));
			System.err.println("model.totdoc:" + model.totdoc);

			line = br.readLine();
			model.sv_num = Integer.parseInt(SSO.beforeStr(line, "#"));
			System.err.println("model.sv_num:" + model.sv_num);

			line = br.readLine();
			// //line = br.readLine();
			model.b = Double.parseDouble(SSO.beforeStr(line, "#"));
			System.err.println("model.b:" + model.b);

			// //line = br.readLine();
			// System.out.println("b:" + model.b);
			model.supvec = new DOC[model.sv_num];
			model.alpha = new double[model.sv_num];
			model.index = null;
			model.lin_weights = null;
			WORD[] read_words;
			for (i = 1; i < model.sv_num; i++) {
				line = br.readLine();
				line = SSO.beforeStr(line, "#");
				// System.err.println("i:"+i+" "+line);
				ReadStruct rs = new ReadStruct();
				read_words = parseDocument(line, max_words, rs);
				model.alpha[i] = rs.read_doc_label;
				queryid = rs.read_queryid;
				comment = rs.read_comment;
				words = read_words;
				model.supvec[i] = createExample(-1, 0, 0, 0.0, createSvector(words, comment, 1.0));
				model.supvec[i].fvec.kernel_id = queryid;
			}

			br.close();
			fr.close();

		} catch (Exception e) {
			// e.printStackTrace();
		}

		return model;
	}

	/* create deep copy of matrix */
	public MATRIX copy_matrix(MATRIX matrix) {
		int i, j;
		MATRIX copy;
		copy = createMatrix(matrix.n, matrix.m);
		for (i = 0; i < matrix.n; i++) {
			for (j = 0; j < matrix.m; j++) {
				copy.element[i][j] = matrix.element[i][j];
			}
		}
		return (copy);
	}

	/* Given a lower triangular matrix L, computes inverse L^-1 */
	public MATRIX invert_ltriangle_matrix(MATRIX L) {
		int i, j, k, n;
		double sum;
		MATRIX I;

		if (L.m != L.n) {
			System.out.println("ERROR: Matrix not quadratic. Cannot invert triangular matrix!");
			System.exit(1);
		}

		n = L.n;
		I = copy_matrix(L);

		for (i = 0; i < n; i++) {
			I.element[i][i] = 1.0 / L.element[i][i];
			for (j = i + 1; j < n; j++) {
				sum = 0.0;
				for (k = i; k < j; k++)
					sum -= I.element[j][k] * I.element[k][i];
				I.element[j][i] = sum / L.element[j][j];
			}
		}

		return (I);
	}

	public double[] reallocDoubleArr(double[] arr, int nsize) {

		double[] narr = new double[nsize];
		if (arr == null) {
			///for (int ni = 0; ni < nsize; ni++) {
			///	narr[ni] = 0;
			///}
			return narr;
		}

		if (nsize <= arr.length) {
			narr = new double[nsize];

			System.arraycopy(arr, 0, narr, 0, nsize);
			///for (int ni = 0; ni < nsize; ni++) {
			///	narr[ni] = arr[ni];
			///}

			///for (int ni = nsize; ni < arr.length; ni++) {
			///	narr[ni] = 0;
			///}

			return narr;
		}

		///for (int ni = 0; ni < arr.length; ni++) {
		///	narr[ni] = arr[ni];
		///}

		System.arraycopy(arr, 0, narr, 0, arr.length);

		///for (int ni = arr.length; ni < nsize; ni++) {
		///	narr[ni] = 0;
		///}

		return narr;
	}

	public int[] reallocIntArr(int[] arr, int nsize) {

		int[] narr = new int[nsize];
		if (arr == null) {
			///for (int ni = 0; ni < nsize; ni++) {
			///	narr[ni] = 0;
			///}
			return narr;
		}
		if (nsize <= arr.length) {
			narr = new int[nsize];
			///for (int ni = 0; ni < nsize; ni++) {
			///	narr[ni] = arr[ni];
			///}

			System.arraycopy(arr, 0, narr, 0, nsize);

			///for (int ni = nsize; ni < arr.length; ni++) {
			///	narr[ni] = 0;
			//}
			return narr;
		}
		///for (int ni = 0; ni < arr.length; ni++) {
		///	narr[ni] = arr[ni];
		///}

		System.arraycopy(arr, 0, narr, 0, arr.length);

		///for (int ni = arr.length; ni < nsize; ni++) {
		///	narr[ni] = 0;
		///}

		return narr;
	}

	public DOC[] reallocDOCS(DOC[] ods, int n) {

		DOC[] ndoc = new DOC[n];
		if (ods == null) {
			for (int i = 0; i < n; i++) {
				ndoc[i] = new DOC();
			}
			return ndoc;
		}
		///for (int i = 0; i < ods.length; i++) {
		///ndoc[i] = ods[i].copyDoc();
		///	ndoc[i] = ods[i];
		///}

		System.arraycopy(ods, 0, ndoc, 0, ods.length);
		for (int i = ods.length; i < n; i++) {
			ndoc[i] = new DOC();
		}

		return ndoc;
	}

	/**
	 * 重新分配alpha大小
	 * 
	 * @param alpha
	 * @param m
	 * @return
	 */
	public double[] reallocAlpha(double[] alpha, int m) {
		double[] oalpha = alpha;
		alpha = new double[m];
		//for (int i = 0; i < (m - 1); i++) {
		//	alpha[i] = oalpha[i];
		//}

		System.arraycopy(oalpha, 0, alpha, 0, m);
		alpha[m - 1] = 0;

		return alpha;
	}

	/**
	 * 重新分配alpha_list大小
	 * 
	 * @param alpha
	 * @param m
	 * @return
	 */
	public int[] reallocAlphaList(int[] alpha_list, int m) {
		int[] oalpha_list = alpha_list;
		alpha_list = new int[m];
		System.arraycopy(oalpha_list, 0, alpha_list, 0, m);
		alpha_list[m - 1] = 0;

		return alpha_list;
	}

	/**
	 * 重新分配lhs的大小
	 * 
	 * @param cset
	 */
	public void realSmalllocLhs(CONSTSET cset) {
		DOC[] olhs = cset.lhs;
		cset.lhs = new DOC[cset.m];
		System.arraycopy(olhs, 0, cset.lhs, 0, cset.m);
	}

	/**
	 * 重新分配rhs的大小
	 * 
	 * @param cset
	 */
	public void realSmalllocRhs(CONSTSET cset) {
		double[] orhs = cset.rhs;
		cset.rhs = new double[cset.m];
		///for (int i = 0; i < (cset.m); i++) {
		///	cset.rhs[i] = orhs[i];
		///}

		System.arraycopy(orhs, 0, cset.rhs, 0, cset.m);
	}

	/**
	 * 重新分配cset的内存大小
	 * 
	 * @param cset
	 */
	public void realloc(CONSTSET cset) {
		DOC[] olhs = cset.lhs;
		cset.lhs = new DOC[cset.m];

		///for (int i = 0; i < (cset.m - 1); i++) {
		///	cset.lhs[i] = olhs[i];
		///}
		System.arraycopy(olhs, 0, cset.lhs, 0, cset.m - 1);

		cset.lhs[cset.m - 1] = new DOC();
	}

	/**
	 * 重新分配rhs的大小
	 * 
	 * @param cset
	 */
	public void reallocRhs(CONSTSET cset) {
		double[] orhs = cset.rhs;
		cset.rhs = new double[cset.m];
		///for (int i = 0; i < (cset.m - 1); i++) {
		///	cset.rhs[i] = orhs[i];
		///}
		System.arraycopy(orhs, 0, cset.rhs, 0, cset.m - 1);
		cset.rhs[cset.m - 1] = 0;
	}

	/**
	 * computes an incomplete cholesky decomposition as describe bei Fine and
	 * Scheinberg (JMLR01). rank is the desired rank, and epsilon is the cutoff
	 * on the pivot score. This means it can return a solution with lower rank
	 * than specified. The cholesky matrix returned is in the lower triangular
	 * portion. index returns an array with the indices of the vectors in x that
	 * were selected in the decomposition (terminated by -1).
	 */
	public MATRIX incompleteCholesky(DOC[] x, int n, int rank, double epsilon, KERNEL_PARM kparm, int[] index) {
		int i, j, k, pivot, temp2;
		int[] pindex;
		int[] swap;
		double pscore, temp;
		double[] dG;
		MATRIX G;

		pindex = new int[rank + 1];
		swap = new int[n];
		dG = new double[n];
		G = createMatrix(n, rank);
		for (i = 0; i < n; i++) {
			swap[i] = i;
			for (j = 0; j < rank; j++) {
				G.element[i][j] = 0;
			}
		}
		for (i = 0; i < rank; i++) {
			// compute pivot score
			for (j = i; j < n; j++) {
				dG[j] = kernel(kparm, x[swap[j]], x[swap[j]]);
				for (k = 0; k <= i - 1; k++) {
					dG[j] -= G.element[j][k] * G.element[j][k];
				}
			}

			// find max pivot
			for (j = i, pivot = i, pscore = 0; j < n; j++) {
				if (pscore < dG[j]) {
					pscore = dG[j];
					pivot = j;
				}
			}
			if (pscore <= epsilon) {
				pindex[i] = -1;
				index = pindex;
				return (reallocMatrix(G, i, i));
			}

			pindex[i] = swap[pivot];
			for (j = i; j < n; j++) {
				G.element[j][i] = kernel(kparm, x[swap[j]], x[swap[pivot]]);
			}
			temp2 = swap[pivot];
			swap[pivot] = swap[i];
			swap[i] = temp2;
			for (j = 0; j <= i; j++) {
				temp = G.element[i][j];
				G.element[i][j] = G.element[pivot][j];
				G.element[pivot][j] = temp;
			}
			G.element[i][i] = dG[pivot];
			for (j = 0; j <= i - 1; j++) {
				for (k = i + 1; k < n; k++) {
					G.element[k][i] -= G.element[k][j] * G.element[i][j];
				}
			}
			G.element[i][i] = Math.sqrt(G.element[i][i]);
			for (k = i + 1; k < n; k++) {
				G.element[k][i] /= G.element[i][i];
			}
		}

		pindex[i] = -1;
		index = pindex;
		return (reallocMatrix(G, rank, rank));
	}

	/**
	 * For column vector v and a lower triangular matrix A (assumed to match in
	 * size), computes w^T=v^T*A
	 */
	public double[] prod_nvector_ltmatrix(double[] v, MATRIX A) {
		int i, j;
		double sum;
		double[] w;

		w = createNvector(A.m);

		for (i = 0; i < A.m; i++) {
			sum = 0.0;
			for (j = i; j < A.n; j++) {
				sum += v[j] * A.element[j][i];
			}
			w[i] = sum;
		}

		return (w);
	}

	/**
	 * Given a positive-semidefinite symmetric matrix A[0..n-1][0..n-1], this
	 * routine finds a subset of rows and colums that is linear independent. To
	 * do this, it constructs the Cholesky decomposition, A = L ?LT. On input,
	 * only the upper triangle of A need be given; A is not modified. The
	 * routine returns a vector in which non-zero elements indicate the linear
	 * independent subset. epsilon is the amount by which the diagonal entry of
	 * L has to be greater than zero.
	 */
	public double[] findIndepSubsetOfMatrix(MATRIX A, double epsilon) {
		int i, j, k, n;
		double sum;
		double[] indep;
		MATRIX L;

		if (A.m != A.n) {
			System.out.printf("ERROR: Matrix not quadratic. Cannot compute Cholesky!\n");
			System.exit(1);
		}
		n = A.n;
		L = copy_matrix(A);

		for (i = 0; i < n; i++) {
			for (j = i; j < n; j++) {
				for (sum = L.element[i][j], k = i - 1; k >= 0; k--)
					sum -= L.element[i][k] * L.element[j][k];
				if (i == j) {
					if (sum <= epsilon)
						sum = 0;
					L.element[i][i] = Math.sqrt(sum);
				} else if (L.element[i][i] == 0)
					L.element[j][i] = 0;
				else
					L.element[j][i] = sum / L.element[i][i];
			}
		}
		// Gather non-zero diagonal elements
		indep = createNvector(n);
		for (i = 0; i < n; i++)
			indep[i] = L.element[i][i];

		return (indep);
	}

	/**
	 * Given a positive-definite symmetric matrix A[0..n-1][0..n-1], this
	 * routine constructs its Cholesky decomposition, A = L ?LT . On input, only
	 * the upper triangle of A need be given; A is not modified. The Cholesky
	 * factor L is returned in the lower triangle.
	 */
	public MATRIX choleskyMatrix(MATRIX A) {
		int i, j, k, n;
		double sum;
		MATRIX L;

		if (A.m != A.n) {
			System.out.printf("ERROR: Matrix not quadratic. Cannot compute Cholesky!\n");
			System.exit(1);
		}
		n = A.n;
		L = copy_matrix(A);

		for (i = 0; i < n; i++) {
			for (j = i; j < n; j++) {
				for (sum = L.element[i][j], k = i - 1; k >= 0; k--)
					sum -= L.element[i][k] * L.element[j][k];
				if (i == j) {
					if (sum <= 0.0)
						System.out.printf("Cholesky: Matrix not positive definite");
					L.element[i][i] = Math.sqrt(sum);
				} else
					L.element[j][i] = sum / L.element[i][i];
			}
		}
		// set upper triangle to zero
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				L.element[i][j] = 0;

		return (L);
	}

	/**
	 * For column vector v and lower triangular matrix A (assumed to match in
	 * size), computes w=A*v
	 */
	public double[] prod_ltmatrix_nvector(MATRIX A, double[] v) {
		int i, j;
		double sum;
		double[] w;

		w = createNvector(A.n);

		for (i = 0; i < A.n; i++) {
			sum = 0.0;
			for (j = 0; j <= i; j++) {
				sum += v[j] * A.element[i][j];
			}
			w[i] = sum;
		}

		return (w);
	}

}
