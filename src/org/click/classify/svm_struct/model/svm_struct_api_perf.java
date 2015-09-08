package org.click.classify.svm_struct.model;

import java.io.InputStream;
import java.util.ArrayList;

import org.click.classify.svm_struct.data.EXAMPLE;
import org.click.classify.svm_struct.data.KERNEL_PARM;
import org.click.classify.svm_struct.data.LABEL;
import org.click.classify.svm_struct.data.LEARN_PARM;
import org.click.classify.svm_struct.data.ModelConstant;
import org.click.classify.svm_struct.data.PATTERN;
import org.click.classify.svm_struct.data.ReadStruct;
import org.click.classify.svm_struct.data.SAMPLE;
import org.click.classify.svm_struct.data.STRUCTMODEL;
import org.click.classify.svm_struct.data.STRUCT_LEARN_PARM;
import org.click.classify.svm_struct.data.SVECTOR;
import org.click.classify.svm_struct.data.WORD;

public class svm_struct_api_perf extends svm_struct_api {

	@Override
	public void init_struct_model(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm) {
		// TODO Auto-generated method stub

	}

	@Override
	public SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL find_most_violated_constraint_slackrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL find_most_violated_constraint_marginrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean empty_label(LABEL y) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	/**Reads struct examples and returns them in sample. The number of
	 examples must be written into sample.n */
	public SAMPLE read_struct_examples(String file, STRUCT_LEARN_PARM sparm) {
		SAMPLE sample = new SAMPLE(); // sample
		EXAMPLE[] examples;
		int n; // number of examples
		int totwords, maxlength = 0, length, i, j, nump = 0, numn = 0;
		WORD[] words;
		WORD w;
		// we have only one big example
		examples = new EXAMPLE[1];
		examples[0] = new EXAMPLE();
		// Using the read_documents function from SVM-light
		ReadStruct rs = new ReadStruct();

		examples[0].x.docs = svm_common.read_documents(file,
				examples[0].y.class_indexs, rs);
		examples[0].x.totdoc = rs.read_totdocs;
		examples[0].y.totdoc = rs.read_totdocs;
		sample.n = 1;
		sample.examples = examples;
		n = rs.read_totdocs;
		totwords = rs.read_totwords;

		if (sparm.preimage_method == 9) {
			for (i = 0; i < n; i++) {
				examples[0].x.docs[i].fvec.next = svm_common
						.copy_svector(examples[0].x.docs[i].fvec);
				examples[0].x.docs[i].fvec.kernel_id = 0;
				examples[0].x.docs[i].fvec.next.kernel_id = 2;
			}
		}

		for (i = 0; i < sample.examples[0].x.totdoc; i++) {
			length = 1;
			if (sample.examples[0].y.class_indexs[i] > 0)
				nump++;
			else
				numn++;
			// for(w=sample.examples[0].x.docs[i].fvec.words;ww.wnum;w++) {
			for (j = 0; j < sample.examples[0].x.docs[i].fvec.words.length; j++) {
				w = sample.examples[0].x.docs[i].fvec.words[j];
				length++;
				if (length > maxlength) // find vector with max elements
					maxlength = length;
			}
		}

		// add feature for bias if necessary
		// WARNING: Currently this works correctly only for linear kernel!
		sparm.bias_featurenum = 0;
		if (sparm.bias != 0) {
			words = new WORD[maxlength + 1];
			sparm.bias_featurenum = totwords + 1;
			totwords++;
			for (i = 0; i < sample.examples[0].x.totdoc; i++) {
				for (j = 0; j < sample.examples[0].x.docs[i].fvec.words.length; j++) {
					words[j] = sample.examples[0].x.docs[i].fvec.words[j];
				}

				words[j].wnum = sparm.bias_featurenum; // bias
				words[j].weight = sparm.bias;

				sample.examples[0].x.docs[i].fvec = svm_common.create_svector(
						words, "", 1.0);
			}
		}

		// Remove all features with numbers larger than num_features, if
		// num_features is set to a positive value. This is important for
		// svm_struct_classify.
		if ((sparm.num_features > 0) && sparm.truncate_fvec != 0)
			for (i = 0; i < sample.examples[0].x.totdoc; i++)
				for (j = 0; j < sample.examples[0].x.docs[i].fvec.words.length; j++) {
					if (sample.examples[0].x.docs[i].fvec.words[j].wnum > sparm.num_features) {
						sample.examples[0].x.docs[i].fvec.words[j].wnum = 0;
						sample.examples[0].x.docs[i].fvec.words[j].weight = 0;
					}
				}

		// change label value for better scaling using thresholdmetrics
		if ((sparm.loss_function == ModelConstant.ZEROONE)
				|| (sparm.loss_function == ModelConstant.FONE)
				|| (sparm.loss_function == ModelConstant.ERRORRATE)
				|| (sparm.loss_function == ModelConstant.PRBEP)
				|| (sparm.loss_function == ModelConstant.PREC_K)
				|| (sparm.loss_function == ModelConstant.REC_K)) {
			for (i = 0; i < sample.examples[0].x.totdoc; i++) {
				if (sample.examples[0].y.class_indexs[i] > 0)
					sample.examples[0].y.class_indexs[i] = 0.5 * 100.0 / (numn + nump);
				else
					sample.examples[0].y.class_indexs[i] = -0.5 * 100.0
							/ (numn + nump);
			}
		}

		// change label value for easy computation of rankmetrics (i.e.
		// ROC-area)
		if (sparm.loss_function == ModelConstant.SWAPPEDPAIRS) {
			for (i = 0; i < sample.examples[0].x.totdoc; i++) {

				if (sample.examples[0].y.class_indexs[i] > 0)
					sample.examples[0].y.class_indexs[i] = 0.5 * 100.0 / nump;
				else
					sample.examples[0].y.class_indexs[i] = -0.5 * 100.0 / numn;
			}
		}

		if (sparm.loss_function == ModelConstant.AVGPREC) {
			for (i = 0; i < sample.examples[0].x.totdoc; i++) {
				if (sample.examples[0].y.class_indexs[i] > 0)
					sample.examples[0].y.class_indexs[i] = numn;
				else
					sample.examples[0].y.class_indexs[i] = -nump;
			}
		}

		return sample;

	}

	@Override
	public SAMPLE read_struct_examples_from_stream(InputStream is,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SAMPLE read_struct_examples_from_arraylist(ArrayList<String> list,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL classify_struct_example(PATTERN x, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public PATTERN sample2pattern(String words) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public STRUCTMODEL read_struct_model(String file, STRUCT_LEARN_PARM sparm) {

		return null;
	}

	@Override
	public void write_struct_model(String file, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {

	}

}
