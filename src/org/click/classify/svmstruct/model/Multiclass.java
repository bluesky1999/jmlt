package org.click.classify.svmstruct.model;

import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.click.classify.svmstruct.data.DOC;
import org.click.classify.svmstruct.data.EXAMPLE;
import org.click.classify.svmstruct.data.KERNEL_PARM;
import org.click.classify.svmstruct.data.LABEL;
import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.PATTERN;
import org.click.classify.svmstruct.data.ReadStruct;
import org.click.classify.svmstruct.data.SAMPLE;
import org.click.classify.svmstruct.data.STRUCTMODEL;
import org.click.classify.svmstruct.data.STRUCT_LEARN_PARM;
import org.click.classify.svmstruct.data.SVECTOR;
import org.click.classify.svmstruct.data.WORD;

/**
 * 多分类struct api
 * 
 * @author zkyz
 */
public class Multiclass extends Struct {

	public Multiclass() {
		super();
	}

	@Override
	public void initStructModel(SAMPLE sample, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm) {
		// TODO Auto-generated method stub
		int i, totwords = 0;
		WORD w;
		WORD[] temp_words;

		sparm.num_classes = 1;
		for (i = 0; i < sample.n; i++)
			// find highest class label
			if (sparm.num_classes < ((sample.examples[i].y.class_index) + 0.1))
				sparm.num_classes = (int) (sample.examples[i].y.class_index + 0.1);
		for (i = 0; i < sample.n; i++) // find highest feature number
		{
			temp_words = sample.examples[i].x.doc.fvec.words;
			for (int j = 0; j < temp_words.length; j++) {
				w = temp_words[j];
				if (totwords < w.wnum) {
					totwords = w.wnum;
				}
			}

		}

		sparm.num_features = totwords;
		sm.sizePsi = sparm.num_features * sparm.num_classes;
	}

	@Override
	public SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		SVECTOR fvec;

		fvec = com.shiftS(x.doc.fvec, (y.class_index - 1) * sparm.num_features);

		fvec.kernel_id = y.class_index;
		return fvec;
	}

	@Override
	public LABEL findMostViolatedConstraintSlackrescaling(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		LABEL ybar = new LABEL();
		DOC doc;
		int ci;
		int bestclass = -1;
		int first = 1;
		double score, score_y, score_ybar, bestscore = -1;

		// NOTE: This function could be made much more efficient by not always
		// computing a new PSI vector.
		doc = (x.doc);
		doc.fvec = psi(x, y, sm, sparm);
		score_y = com.classifyExample(sm.svm_model, doc);

		ybar.scores = null;
		ybar.num_classes = sparm.num_classes;

		for (ci = 1; ci <= sparm.num_classes; ci++) {
			ybar.class_index = ci;
			doc.fvec = psi(x, ybar, sm, sparm);
			score_ybar = com.classifyExample(sm.svm_model, doc);
			score = loss(y, ybar, sparm) * (1.0 - score_y + score_ybar);
			if ((bestscore < score) || (first != 0)) {
				bestscore = score;
				bestclass = ci;
				first = 0;
			}

		}

		ybar.class_index = bestclass;
		return (ybar);
	}

	@Override
	public LABEL findMostViolatedConstraintMarginrescaling(PATTERN x, LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		LABEL ybar = new LABEL();
		DOC doc;
		int ci = 0;
		int bestclass = -1;
		int first = 1;
		double score, bestscore = -1;

		// NOTE: This function could be made much more efficient by not always
		// computing a new PSI vector.

		doc = new DOC();
		ybar.scores = null;
		ybar.num_classes = sparm.num_classes;

		for (ci = 1; ci <= sparm.num_classes; ci++) {
			ybar.class_index = ci;

			doc.fvec = psi(x, ybar, sm, sparm);
			score = com.classifyExample(sm.svm_model, doc);
			score += loss(y, ybar, sparm);
			if ((bestscore < score) || (first != 0)) {
				bestscore = score;
				bestclass = ci;
				first = 0;
			}
		}

		ybar.class_index = bestclass;

		return (ybar);
	}

	@Override
	public double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM sparm) {

		if (sparm.loss_function == 0) { // type 0 loss: 0/1 loss
			if (y.class_index == ybar.class_index)
				return (0);
			else
				return (100);
		}
		if (sparm.loss_function == 1) { // type 1 loss: squared difference
			return ((y.class_index - ybar.class_index) * (y.class_index - ybar.class_index));
		} else {

			// Put your code for different loss functions here. But then
			// find_most_violated_constraint_???(x, y, sm) has to return the
			// highest scoring label with the largest loss.
			System.exit(1);
		}

		return 10000;
	}

	@Override
	public boolean emptyLabel(LABEL y) {
		return (y.class_index < 0.9);
	}

	@Override
	public LABEL classifyStructExample(PATTERN x, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		LABEL y = new LABEL();
		DOC doc;
		int class_index, bestclass = -1, j;
		boolean first = true;
		double score = 0.0, bestscore = -1;
		WORD[] words;

		doc= new DOC();	
		y.scores = new double[sparm.num_classes + 1];
		y.num_classes = sparm.num_classes;
		words = x.doc.fvec.words;

		for (j = 0; j < words.length; j++) {
			if (words[j].wnum > sparm.num_features) {
				return null;
				// words[j].wnum = 0;
			}
		}

		for (class_index = 1; class_index <= sparm.num_classes; class_index++) {
			y.class_index = class_index;
			doc.fvec = psi(x, y, sm, sparm);

			score = com.classifyExample(sm.svm_model, doc);
			y.scores[class_index] = score;
			if ((bestscore < score) || first) {
				bestscore = score;
				bestclass = class_index;
				first = false;
			}
		}

		y.class_index = bestclass;

		return y;
	}

	@Override
	public LABEL classifyStructDoc(DOC d, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SAMPLE readStructExamples(String file, STRUCT_LEARN_PARM sparm) {

		SAMPLE sample = new SAMPLE();
		EXAMPLE[] examples;
		int n;
		DOC[] docs;
		double[] target = null;
		int totwords, i, num_classes = 0;

		ReadStruct rs = new ReadStruct();
		docs = com.readDocuments(file, rs);

		target = rs.read_target;
		totwords = rs.read_totwords;
		n = rs.read_totdocs;

		for (int k = 0; k < docs.length; k++) {
			if (docs[k] == null || docs[k].fvec == null) {
				continue;
			}
		}
		examples = new EXAMPLE[n];
		for (int k = 0; k < n; k++) {
			examples[k] = new EXAMPLE();
			examples[k].x = new PATTERN();
			examples[k].y = new LABEL();
		}

		for (i = 0; i < n; i++)
			if (num_classes < (target[i] + 0.1))
				num_classes = (int) (target[i] + 0.1);
		for (i = 0; i < n; i++)
			if (target[i] < 1) {
				System.exit(1);
			}

		for (i = 0; i < n; i++) {
			examples[i].x.doc = docs[i];
			examples[i].y.class_index = (int) (target[i] + 0.1);
			examples[i].y.scores = null;
			examples[i].y.num_classes = num_classes;
		}

		sample.n = n;
		sample.examples = examples;

		return (sample);
	}


	@Override
	public PATTERN sample2pattern(String wordString) {
		int dnum = 0;
		int queryid = 0;
		int slackid = 0;
		double costfactor = 0;
		String read_comment = "";
		WORD[] words = null;
		words = string2words(wordString);
		DOC doc = com.createExample(dnum, queryid, slackid, costfactor, com.createSvector(words, read_comment, 1.0));
		PATTERN pat = new PATTERN();
		pat.doc = doc;

		return pat;
	}

	@Override
	public DOC sample2doc(String words) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void writeLabel(PrintWriter fp, LABEL y, LABEL ybar) {
		// TODO Auto-generated method stub

	}

}
