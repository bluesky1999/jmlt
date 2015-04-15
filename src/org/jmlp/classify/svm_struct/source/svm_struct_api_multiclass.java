package org.jmlp.classify.svm_struct.source;

import java.io.InputStream;
import java.util.ArrayList;

import org.apache.log4j.Logger;

/**
 * 多分类struct api
 * @author zkyz
 */
public class svm_struct_api_multiclass extends svm_struct_api{
	
	private static Logger logger = Logger.getLogger(svm_struct_api_multiclass.class);
	
	@Override
	public void init_struct_model(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm) {
		// TODO Auto-generated method stub
		int i, totwords = 0;
		WORD w;
		WORD[] temp_words;

		sparm.num_classes = 1;
		for (i = 0; i < sample.n; i++)
			/* find highest class label */
			if (sparm.num_classes < ((sample.examples[i].y.class_index) + 0.1))
				sparm.num_classes = (int) (sample.examples[i].y.class_index + 0.1);
		for (i = 0; i < sample.n; i++) /* find highest feature number */
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
		if (svm_struct_common.struct_verbosity >= 0) {
			System.out.println("Training set properties: " + sparm.num_features
					+ " features " + sparm.num_classes + " classes\n");
		}

		sm.sizePsi = sparm.num_features * sparm.num_classes;
		if (svm_struct_common.struct_verbosity >= 2) {
			System.out.println("Size of Phi: " + sm.sizePsi + "\n");
		}
	}

	@Override
	public SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		SVECTOR fvec;

		fvec = svm_common.shift_s(x.doc.fvec, (y.class_index - 1)
				* sparm.num_features);

		fvec.kernel_id = y.class_index;
		return fvec;
	}

	@Override
	public LABEL find_most_violated_constraint_slackrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		LABEL ybar = new LABEL();
		DOC doc;
		int ci;
		int bestclass = -1;
		int first = 1;
		double score, score_y, score_ybar, bestscore = -1;

		/*
		 * NOTE: This function could be made much more efficient by not always
		 * computing a new PSI vector.
		 */
		doc = (x.doc);
		doc.fvec = psi(x, y, sm, sparm);
		score_y = svm_common.classify_example(sm.svm_model, doc);

		ybar.scores = null;
		ybar.num_classes = sparm.num_classes;

		for (ci = 1; ci <= sparm.num_classes; ci++) {
			ybar.class_index = ci;
			doc.fvec = psi(x, ybar, sm, sparm);
			score_ybar = svm_common.classify_example(sm.svm_model, doc);
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
	public LABEL find_most_violated_constraint_marginrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		LABEL ybar = new LABEL();
		DOC doc;
		int ci = 0;
		int bestclass = -1;
		int first = 1;
		double score, bestscore = -1;

		/*
		 * NOTE: This function could be made much more efficient by not always
		 * computing a new PSI vector.
		 */
		doc = x.doc.copyDoc();
        
		ybar.scores = null;
		ybar.num_classes = sparm.num_classes;


		for (ci = 1; ci <= sparm.num_classes; ci++) {
			ybar.class_index = ci;
			
			doc.fvec = psi(x, ybar, sm, sparm);
			score = svm_common.classify_example(sm.svm_model, doc);
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
		if (sparm.loss_function == 0) { /* type 0 loss: 0/1 loss */
			if (y.class_index == ybar.class_index)
				return (0);
			else
				return (100);
		}
		if (sparm.loss_function == 1) { /* type 1 loss: squared difference */
			return ((y.class_index - ybar.class_index) * (y.class_index - ybar.class_index));
		} else {
			/*
			 * Put your code for different loss functions here. But then
			 * find_most_violated_constraint_???(x, y, sm) has to return the
			 * highest scoring label with the largest loss.
			 */
			System.exit(1);
		}

		return 10000;
	}

	@Override
	public boolean empty_label(LABEL y) {
		return (y.class_index < 0.9);
	}

	@Override
	public LABEL classify_struct_example(PATTERN x, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		LABEL y = new LABEL();
		DOC doc;
		int class_index, bestclass = -1, j;
		boolean first = true;
		double score=0.0, bestscore = -1;
		WORD[] words;

		doc = x.doc.copyDoc();
		y.scores = new double[sparm.num_classes + 1];
		y.num_classes = sparm.num_classes;
		words = doc.fvec.words;

		
		for (j = 0; j <words.length; j++) {
			if (words[j].wnum > sparm.num_features) {
				return null;
				//words[j].wnum = 0;
			}
		}
	
		for (class_index = 1; class_index <= sparm.num_classes; class_index++) {
			y.class_index = class_index;
			doc.fvec = psi(x, y, sm, sparm);
	
			score = svm_common.classify_example(sm.svm_model, doc);	
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
	public SAMPLE read_struct_examples(String file, STRUCT_LEARN_PARM sparm) {

		SAMPLE sample = new SAMPLE(); 
		EXAMPLE[] examples;
		int n; 
		DOC[] docs; 
		double[] target = null;
		int totwords, i, num_classes = 0;

		docs = svm_common.read_documents(file, target);

		target = svm_common.read_target;
		totwords = svm_common.read_totwords;
		n = svm_common.read_totdocs;

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
	public SAMPLE read_struct_examples_from_stream(InputStream is,
			STRUCT_LEARN_PARM sparm) {
		SAMPLE sample = new SAMPLE(); 
		EXAMPLE[] examples;
		int n; 
		DOC[] docs; 
		double[] target = null;
		int totwords, i, num_classes = 0;

		docs = svm_common.read_documents_from_stream(is, target);

		target = svm_common.read_target;
		totwords = svm_common.read_totwords;
		n = svm_common.read_totdocs;
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
	public SAMPLE read_struct_examples_from_arraylist(ArrayList<String> list,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		
		SAMPLE sample = new SAMPLE(); 
		EXAMPLE[] examples;
		int n; 
		DOC[] docs; 
		double[] target = null;
		int totwords, i, num_classes = 0;
       
		logger.info("begin read documents");
		docs = svm_common.read_documents_from_arraylist(list, target);
		logger.info("end read documents");
		
		target = svm_common.read_target;
		totwords = svm_common.read_totwords;
		n = svm_common.read_totdocs;
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
		int queryid=0;
		int slackid=0;
		double costfactor=0;
		String read_comment="";
		WORD[] words=null;
		words=string2words(wordString);
		DOC doc=svm_common.create_example(dnum, queryid, slackid,costfactor,svm_common.create_svector(words, read_comment, 1.0));
		PATTERN pat=new PATTERN();
		pat.doc=doc;
		
		return pat;
	}

}
