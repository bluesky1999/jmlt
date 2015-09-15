package org.click.classify.svm_struct.model;

import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;

import org.click.classify.svm_struct.data.DOC;
import org.click.classify.svm_struct.data.EXAMPLE;
import org.click.classify.svm_struct.data.KERNEL_PARM;
import org.click.classify.svm_struct.data.LABEL;
import org.click.classify.svm_struct.data.LEARN_PARM;
import org.click.classify.svm_struct.data.MATRIX;
import org.click.classify.svm_struct.data.MODEL;
import org.click.classify.svm_struct.data.ModelConstant;
import org.click.classify.svm_struct.data.PATTERN;
import org.click.classify.svm_struct.data.ReadStruct;
import org.click.classify.svm_struct.data.SAMPLE;
import org.click.classify.svm_struct.data.STRUCTMODEL;
import org.click.classify.svm_struct.data.STRUCT_ID_SCORE;
import org.click.classify.svm_struct.data.STRUCT_LEARN_PARM;
import org.click.classify.svm_struct.data.SVECTOR;
import org.click.classify.svm_struct.data.WORD;

public class Perf extends Struct {

	public Perf() {
		super();
		//com = new Common();
	}

	@Override
	/** Initialize structmodel sm. The weight vector w does not need to be
	initialized, but you need to provide the maximum size of the
	feature space in sizePsi. This is the maximum number of different
	weights that can be learned. Later, the weight vector w will
	contain the learned weights for the model. */
	public void initStructModel(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm) {
		int i, j, k, totwords = 0, totdoc = 0, totexp = 0, nump = 0, numn = 0, new_size = 0;
		int[] select = null;
		WORD[] words;
		WORD w;
		DOC[] orgdoc, basis;
		double[] dummy;
		MATRIX G, L;
		double[] indep;
		double ii, weight;
		totdoc = sample.examples[0].x.totdoc;
		if (sparm.sparse_kernel_method > 0) { // use nystrom or inc cholesky
			sparm.sparse_kernel_type = kparm.kernel_type;
		} else
			sparm.sparse_kernel_type = 0;
		sm.sparse_kernel_type = sparm.sparse_kernel_type;
		sm.invL = null;
		sm.expansion = null;

		// When using sparse kernel approximation, this replaces the
		// original feature vector with the kernel values of the original
		// feature vector and the expansion.
		if (sm.sparse_kernel_type > 0) {
			kparm.kernel_type = (short) sm.sparse_kernel_type;

			if (!(sparm.sparse_kernel_file.equals(""))) {
				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Reading basis functions for sparse kernel expansion from '%s'...",
									sparm.sparse_kernel_file);

				ReadStruct rs = new ReadStruct();
				// com.read_documents(sparm.sparse_kernel_file,basis,dummy,totwords,totexp);
				basis = com.readDocuments(sparm.sparse_kernel_file, rs);
				dummy = rs.read_target;
				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");
			} else {
				basis = new DOC[totdoc];
				for (i = 0; i < totdoc; i++) {
					basis[i] = new DOC();
					basis[i] = sample.examples[0].x.docs[i];
					basis[i].fvec = com
							.copySvector(sample.examples[0].x.docs[i].fvec);
				}
				totexp = totdoc;
			}
			if (sparm.sparse_kernel_size > totexp)
				sparm.sparse_kernel_size = totexp;

			// determine basis functions to use in expansion: B
			if (sparm.sparse_kernel_method == 1) {
				// select expansion via random sampling
				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Selecting random sample of basis functions...");
				sm.expansion = new DOC[totexp];
				sm.expansion_size = 0;
				for (ii = 0.5; ii < totexp; ii += ((double) totexp / sparm.sparse_kernel_size)) {
					sm.expansion[sm.expansion_size] = new DOC();
					sm.expansion[sm.expansion_size] = basis[(int) ii];
					sm.expansion[sm.expansion_size].fvec = com
							.copySvector(basis[(int) ii].fvec);
					sm.expansion_size++;
				}
				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");

				// Make sure they are all independent. If not, select
				// independent subset.

				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Finding independent subset of vectors...");
				G = com.createMatrix(sm.expansion_size, sm.expansion_size);
				for (i = 0; i < sm.expansion_size; i++) { // need only upper
															// triangle
					for (j = i; j < sm.expansion_size; j++) {
						G.element[i][j] = com.kernel(kparm, sm.expansion[i],
								sm.expansion[j]);
					}
				}
				indep = com.findIndepSubsetOfMatrix(G, 0.000001);

				new_size = 0;
				for (i = 0; i < sm.expansion_size; i++) {
					if (indep[i] != 0) {
						sm.expansion[new_size] = sm.expansion[i];
						new_size++;
					} else {
						// free_example(sm.expansion[i],1);
					}
				}

				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("found %ld of %ld...", new_size,
							sm.expansion_size);
				sm.expansion_size = new_size;
				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");

				// compute matrix B^T*B
				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Computing Gram matrix for kernel expansion...");
				G = com.createMatrix(sm.expansion_size, sm.expansion_size);
				for (i = 0; i < sm.expansion_size; i++) {// need upper triangle
															// for cholesky
					for (j = i; j < sm.expansion_size; j++) {
						G.element[i][j] = com.kernel(kparm, sm.expansion[i],
								sm.expansion[j]);
					}
				}
				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");
				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Computing Cholesky decomposition and inverting...");
				L = com.choleskyMatrix(G);
				sm.invL = com.invert_ltriangle_matrix(L);

				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");
			}

			else if (sparm.sparse_kernel_method == 2) {
				// select expansion via incomplete cholesky
				if (CommonStruct.struct_verbosity > 0)
					System.out
							.printf("Computing incomplete Cholesky decomposition...");
				L = com.incompleteCholesky(basis, totexp,
						sparm.sparse_kernel_size, 0.000001, kparm, select);
				sm.invL = com.invert_ltriangle_matrix(L);

				sm.expansion = new DOC[totexp];
				sm.expansion_size = 0;
				for (i = 0; select[i] >= 0; i++) {
					sm.expansion[sm.expansion_size] = new DOC();
					sm.expansion[sm.expansion_size] = basis[select[i]];
					sm.expansion[sm.expansion_size].fvec = com
							.copySvector(basis[select[i]].fvec);
					sm.expansion_size++;
				}
				if (CommonStruct.struct_verbosity > 0)
					System.out.printf("done.\n");
			}

			// compute new features for each example x: B^T*x
			if (CommonStruct.struct_verbosity > 0)
				System.out.printf("Replacing feature vectors...");
			orgdoc = sample.examples[0].x.docs;
			sample.examples[0].x.docs = new DOC[totdoc];
			words = new WORD[sm.expansion_size + 1];
			for (i = 0; i < totdoc; i++) {
				k = 0;
				for (j = 0; j < sm.expansion_size; j++) {
					weight = com.kernel(kparm, orgdoc[i], sm.expansion[j]);
					if (weight != 0) {
						words[k].wnum = j + 1;
						words[k].weight = weight;
						k++;
					}
				}
				words[k].wnum = 0;
				sample.examples[0].x.docs[i] = com.createExample(
						orgdoc[i].docnum, orgdoc[i].queryid, orgdoc[i].slackid,
						orgdoc[i].costfactor, com.createSvector(words,
								orgdoc[i].fvec.userdefined, 1.0));
			}

			kparm.kernel_type = ModelConstant.LINEAR;
			if (CommonStruct.struct_verbosity > 0)
				System.out.printf("done.\n");
		}

		// count number of positive and negative examples
		for (i = 0; i < sample.examples[0].x.totdoc; i++) {
			if (sample.examples[0].y.class_indexs[i] > 0)
				nump++;
			else
				numn++;
		}

		totwords = 0;
		for (i = 0; i < totdoc; i++)
			// find highest feature number
			for (int m = 0; m < sample.examples[0].x.docs[i].fvec.words.length; m++) {
				w = sample.examples[0].x.docs[i].fvec.words[m];
				if (totwords < w.wnum)
					totwords = w.wnum;
			}
		sparm.num_features = totwords;
		if (CommonStruct.struct_verbosity > 0)
			System.out
					.printf("Training set properties: %d features, %ld examples (%ld pos / %ld neg)\n",
							sparm.num_features, totdoc, nump, numn);
		sm.sizePsi = sparm.num_features;
		if (CommonStruct.struct_verbosity >= 2)
			System.out.printf("Size of Phi: %ld\n", sm.sizePsi);

		// fill in default value for k when using Prec@k or Rec@k
		if (sparm.loss_function == ModelConstant.PREC_K) {
			if (sparm.prec_rec_k_frac == 0) {
				sparm.prec_rec_k_frac = 0.5;
			} else if (sparm.prec_rec_k_frac > 1) {
				System.out
						.printf("\nERROR: The value of option --k for Prec@k must not be larger than 1.0!\n\n");
				System.exit(0);
			}
		}
		if (sparm.loss_function == ModelConstant.REC_K) {
			if (sparm.prec_rec_k_frac == 0) {
				sparm.prec_rec_k_frac = 2;
			} else if (sparm.prec_rec_k_frac < 1) {
				System.out
						.printf("\nERROR: The value of option --k for Rec@k must not be smaller than 1.0!\n\n");
				System.exit(0);
			}
		}

		// make sure that the bias feature is not used with kernels
		if (((kparm.kernel_type != ModelConstant.LINEAR) || (sparm.sparse_kernel_type != ModelConstant.LINEAR))
				&& (sparm.bias != 0)) {
			System.out
					.printf("\nThe value of option --b must be zero when kernels are used.\n");
			System.out
					.printf("The option to use a bias for non-linear kernels is not implemented yet!\n\n");
			System.exit(0);
		}

	}

	@Override
	/*
	 * Returns a feature vector describing the match between pattern x and label
	 * y. The feature vector is returned as a list of SVECTOR's. Each SVECTOR is
	 * in a sparse representation of pairs <featurenumber:featurevalue>, where
	 * the last pair has featurenumber 0 as a terminator. Featurenumbers start
	 * with 1 and end with sizePsi. Featuresnumbers that are not specified
	 * default to value 0. As mentioned before, psi() actually returns a list of
	 * SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next' specifies
	 * the next element in the list, terminated by a NULL pointer. The list can
	 * be though of as a linear combination of vectors, where each vector is
	 * weighted by its 'factor'. This linear combination of feature vectors is
	 * multiplied with the learned (kernelized) weight vector to score label y
	 * for pattern x. Without kernels, there will be one weight in sm.w for each
	 * feature. Note that psi has to match find_most_violated_constraint_???(x,
	 * y, sm) and vice versa. In particular,
	 * find_most_violated_constraint_???(x, y, sm) finds that ybar!=y that
	 * maximizes psi(x,ybar,sm)*sm.w (where * is the inner vector product) and
	 * the appropriate function of the loss + margin/slack rescaling method. See
	 * that paper for details.
	 */
	public SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		SVECTOR fvec = null, fvec2;
		double[] sum, sum2;
		int i, totwords;

		// The following special case speeds up computation for the linear
		// kernel. The lines add the feature vectors for all examples into
		// a single vector. This is more efficient for the linear kernel,
		// but is invalid for all other kernels.
		if (sm.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR) {
			totwords = sparm.num_features;
			sum = new double[totwords + 1];
			com.clearNvector(sum, totwords);
			for (i = 0; i < y.totdoc; i++)
				com.addVectorNs(sum, x.docs[i].fvec, y.class_indexs[i]);

			// For sparse kernel, replace feature vector Psi with k=L^-1*Psi
			if (sm.sparse_kernel_type > 0) {
				// sum+1==sum[1]? how add one add one ?
				sum2 = com.prod_ltmatrix_nvector(sm.invL, sum);
				for (i = 0; i < totwords; i++)
					sum[i + 1] = sum2[i];

			}

			fvec = com.createSvectorN(sum, totwords, "", 1);
		} else { // general kernel
			for (i = 0; i < y.totdoc; i++) {
				fvec2 = com.copySvector(x.docs[i].fvec);
				fvec2.next = fvec;
				fvec2.factor = y.class_indexs[i];
				// printf("class %d: %f\n",i,y.class[i]);
				fvec = fvec2;
			}
		}

		return (fvec);
	}

	@Override
	public LABEL findMostViolatedConstraintSlackrescaling(PATTERN x, LABEL y,
			STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL findMostViolatedConstraintMarginrescaling(PATTERN x, LABEL y,
			STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * loss for correct label y and predicted label ybar. The loss for y==ybar
	 * has to be zero. sparm->loss_function is set with the -l option.
	 */
	@Override
	public double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM sparm) {
		int a = 0, b = 0, c = 0, d = 0, i;
		double loss = 1;

		// compute contingency table
		for (i = 0; i < y.totdoc; i++) {
			if ((y.class_indexs[i] > 0) && (ybar.class_indexs[i] > 0)) {
				a++;
			}
			if ((y.class_indexs[i] > 0) && (ybar.class_indexs[i] <= 0)) {
				c++;
			}
			if ((y.class_indexs[i] < 0) && (ybar.class_indexs[i] > 0)) {
				b++;
			}
			if ((y.class_indexs[i] < 0) && (ybar.class_indexs[i] <= 0)) {
				d++;
			}
		}

		// Return the loss according to the selected loss function.
		if (sparm.loss_function == ModelConstant.ZEROONE) { // type 0 loss:
															// 0/1loss
			// return 0, if y==ybar. return 1 else
			loss = zerooneLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.FONE) {
			loss = foneLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.ERRORRATE) {
			loss = errorrateLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.PRBEP) {
			// WARNING: only valid if called for a labeling that is at PRBEP
			loss = prbepLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.PREC_K) {
			// WARNING: only valid if for a labeling that predicts k positives
			loss = precKLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.REC_K) {
			// WARNING: only valid if for a labeling that predicts k positives
			loss = recKLoss(a, b, c, d);
		} else if (sparm.loss_function == ModelConstant.SWAPPEDPAIRS) {
			loss = swappedpairsLoss(y, ybar);
		} else if (sparm.loss_function == ModelConstant.AVGPREC) {
			loss = avgprecLoss(y, ybar);
		} else {
			// Put your code for different loss functions here. But then
			// find_most_violated_constraint_???(x, y, sm) has to return the
			// highest scoring label with the largest loss.
			System.out.printf("Unkown loss function type: %d\n",
					sparm.loss_function);
			System.exit(1);
		}

		return (loss);
	}

	@Override
	public boolean emptyLabel(LABEL y) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	/**Reads struct examples and returns them in sample. The number of
	 examples must be written into sample.n */
	public SAMPLE readStructExamples(String file, STRUCT_LEARN_PARM sparm) {
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

		examples[0].x.docs = com.readDocuments(file, rs);
		examples[0].y.class_indexs = rs.read_target;
		examples[0].x.totdoc = rs.read_totdocs;
		examples[0].y.totdoc = rs.read_totdocs;
		n = rs.read_totdocs;
		totwords = rs.read_totwords;

		sample.n = 1;
		sample.examples = examples;

		System.err.println("totwords in perf:" + totwords);

		if (sparm.preimage_method == 9) {
			for (i = 0; i < n; i++) {
				examples[0].x.docs[i].fvec.next = com
						.copySvector(examples[0].x.docs[i].fvec);
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

				sample.examples[0].x.docs[i].fvec = com.createSvector(words,
						"", 1.0);
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
	public SAMPLE readStructExamplesFromStream(InputStream is,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SAMPLE readStructExamplesFromArraylist(ArrayList<String> list,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * Finds the label yhat for pattern x that scores the highest according to
	 * the linear evaluation function in sm, especially the weights sm.w. The
	 * returned label is taken as the prediction of sm for the pattern x. The
	 * weights correspond to the features defined by psi() and range from index
	 * 1 to index sm->sizePsi. If the function cannot find a label, it shall
	 * return an empty label as recognized by the function empty_label(y).
	 */
	@Override
	public LABEL classifyStructExample(PATTERN x, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		LABEL y = new LABEL();
		int i;

		y.totdoc = x.totdoc;
		y.class_indexs = new double[y.totdoc];

		// simply classify by sign of inner product between example vector and
		// weight vector
		for (i = 0; i < x.totdoc; i++) {
			y.class_indexs[i] = com.classifyExample(sm.svm_model, x.docs[i]);
		}
		return (y);
	}

	@Override
	public LABEL classifyStructDoc(DOC d, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		LABEL y = new LABEL();

		y.dou_index = com.classifyExample(sm.svm_model, d);

		return (y);
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
		DOC doc = com.createExample(dnum, queryid, slackid, costfactor,
				com.createSvector(words, read_comment, 1.0));
		PATTERN pat = new PATTERN();
		pat.doc = doc;

		return pat;
	}

	@Override
	public DOC sample2doc(String wordString) {
		int dnum = 0;
		int queryid = 0;
		int slackid = 0;
		double costfactor = 0;
		String read_comment = "";
		WORD[] words = null;
		words = string2words(wordString);
		DOC doc = com.createExample(dnum, queryid, slackid, costfactor,
				com.createSvector(words, read_comment, 1.0));

		return doc;
	}

	/**
	 * Reads structural model sm from file file. This function is used only in
	 * the prediction module, not in the learning module.
	 */
	@Override
	public STRUCTMODEL readStructModel(String file, STRUCT_LEARN_PARM sparm) {
		STRUCTMODEL sm = new STRUCTMODEL();
		sm.svm_model = com.read_model(file);
		sparm.loss_function = ModelConstant.ERRORRATE;
		sparm.bias = 0;
		sparm.bias_featurenum = 0;
		sparm.num_features = sm.svm_model.totwords;
		if (sm.svm_model.kernel_parm.kernel_type == ModelConstant.LINEAR)
			sparm.truncate_fvec = 1;
		else
			sparm.truncate_fvec = 0;
		if (sm.svm_model.kernel_parm.kernel_type == ModelConstant.CUSTOM) // double
																			// kernel
			sparm.preimage_method = 9;
		sm.invL = null;
		sm.expansion = null;
		sm.expansion_size = 0;
		sm.sparse_kernel_type = 0;
		sm.w = sm.svm_model.lin_weights;
		sm.sizePsi = sm.svm_model.totwords;
		// if((sm.svm_model.kernel_parm.kernel_type!= ModelConstant.LINEAR) &&
		// sparm.classify_dense!=0)
		// svm_com.add_dense_vectors_to_model(sm.svm_model);
		return (sm);
	}

	@Override
	public void writeStructModel(String file, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {

	}

	/** Writes label y to file handle fp. */
	@Override
	public void writeLabel(PrintWriter fp, LABEL y) {
		int i;
		for (i = 0; i < y.totdoc; i++) {
			fp.printf("%.8f\n", y.class_indexs[i]);
		}

	}

	@Override
	public void writeLabel(PrintWriter fp, LABEL y, LABEL ybar) {
		int i;
		for (i = 0; i < y.totdoc; i++) {
			fp.printf("%.8f\t%.8f\n", y.class_indexs[i], ybar.class_indexs[i]);
		}
	}

	double zeroone(int a, int b, int c, int d) {
		if ((a + d) == (a + b + c + d))
			return (0.0);
		else
			return (1.0);
	}

	double fone(int a, int b, int c, int d) {
		if ((a == 0) || (a + b == 0) || (a + c == 0))
			return (0.0);
		double precision = prec(a, b, c, d);
		double recall = rec(a, b, c, d);
		return (2.0 * precision * recall / (precision + recall));
	}

	/** Returns precision as fractional value. */
	double prec(int a, int b, int c, int d) {
		if ((a + b) == 0)
			return (0.0);
		return ((double) a / (double) (a + b));
	}

	/** Returns recall as fractional value. */
	double rec(int a, int b, int c, int d) {
		if ((a + c) == 0)
			return (0.0);
		return ((double) a / (double) (a + c));
	}

	/** Returns number of errors. */
	double errorrate(int a, int b, int c, int d) {
		if ((a + b + c + d) == 0)
			return (0.0);
		return (((double) (b + c)) / (double) (a + b + c + d));
	}

	/**
	 * Returns percentage of swapped pos/neg pairs (i.e. 100 - ROC Area) for
	 * prediction vectors that encode the number of misranked examples for each
	 * particular example. WARNING: Works only for labels in the compressed
	 * representation
	 */
	double swappedpairs(LABEL y, LABEL ybar) {
		int i;
		double sum = 0;
		for (i = 0; i < y.totdoc; i++)
			sum += Math.abs(y.class_indexs[i] - ybar.class_indexs[i]);
		return (sum / 2.0);
	}

	double zerooneLoss(int a, int b, int c, int d) {
		return (zeroone(a, b, c, d));
	}

	double foneLoss(int a, int b, int c, int d) {
		return (100.0 * (1.0 - fone(a, b, c, d)));
	}

	double errorrateLoss(int a, int b, int c, int d) {
		return (100.0 * errorrate(a, b, c, d));
	}

	/** WARNING: Returns lower bound on PRBEP, if b!=c. */
	double prbepLoss(int a, int b, int c, int d) {
		double precision = prec(a, b, c, d);
		double recall = rec(a, b, c, d);
		if (precision < recall)
			return (100.0 * (1.0 - precision));
		else
			return (100.0 * (1.0 - recall));
	}

	/** WARNING: Only valid if called with a+c==k. */
	double precKLoss(int a, int b, int c, int d) {
		return (100.0 * (1.0 - prec(a, b, c, d)));
	}

	/** WARNING: Only valid if called with a+c==k. */
	double recKLoss(int a, int b, int c, int d) {
		return (100.0 * (1.0 - rec(a, b, c, d)));
	}

	double swappedpairsLoss(LABEL y, LABEL ybar) {
		double nump = 0, numn = 0;
		int i;
		for (i = 0; i < y.totdoc; i++) {
			if (y.class_indexs[i] > 0)
				nump++;
			else
				numn++;
		}
		return (swappedpairs(y, ybar));
	}

	/**
	 * to do
	 * 
	 * @param y
	 * @param ybar
	 * @return
	 */
	double avgprecLoss(LABEL y, LABEL ybar) {
		return 100;
		// return(100.0-avgprec_compressed(y,ybar));
	}

	/**
	 * Finds the most violated constraint for metrics that are based on a
	 * threshold.
	 */
	LABEL find_most_violated_constraint_thresholdmetric(PATTERN x, LABEL y,
			STRUCTMODEL sm, STRUCT_LEARN_PARM sparm, int loss_type) {
		LABEL ybar = new LABEL();
		int i, nump, numn, start, prec_rec_k, totwords;
		double[] score, sump, sumn;
		// STRUCT_ID_SCORE[] scorep, scoren;

		ArrayList<STRUCT_ID_SCORE> scorep = new ArrayList<STRUCT_ID_SCORE>();
		ArrayList<STRUCT_ID_SCORE> scoren = new ArrayList<STRUCT_ID_SCORE>();

		int threshp = 0, threshn = 0;
		int a, d;
		double val = 0, valmax, loss, score_y;
		double[] ortho_weights;

		MODEL svm_model;

		ybar.totdoc = x.totdoc;
		ybar.class_indexs = new double[x.totdoc];
		score = new double[ybar.totdoc + 1];
		// scorep = new STRUCT_ID_SCORE[ybar.totdoc + 1];
		for (int j = 0; j < ybar.totdoc + 1; j++) {
			scorep.add(new STRUCT_ID_SCORE());
		}
		// scoren = new STRUCT_ID_SCORE[ybar.totdoc + 1];
		for (int j = 0; j < ybar.totdoc + 1; j++) {
			// scoren[j] = new STRUCT_ID_SCORE();
			scoren.add(new STRUCT_ID_SCORE());
		}

		sump = new double[ybar.totdoc + 1];
		sumn = new double[ybar.totdoc + 1];

		totwords = sm.svm_model.totwords;
		svm_model = sm.svm_model; // is copy

		// For sparse kernel, replace weight vector with beta=gamma^T*L^-1
		if (sm.sparse_kernel_type > 0) {
			svm_model.lin_weights = new double[totwords + 1];
			// how weight add one add one?
			ortho_weights = com.prod_nvector_ltmatrix(sm.svm_model.lin_weights,
					sm.invL);
			for (i = 0; i < sm.invL.m; i++)
				svm_model.lin_weights[i + 1] = ortho_weights[i];
			svm_model.lin_weights[0] = 0;

		}

		nump = 0;
		numn = 0;
		for (i = 0; i < x.totdoc; i++) {
			score[i] = Math.abs(y.class_indexs[i])
					* com.classifyExample(svm_model, x.docs[i]);
			if (y.class_indexs[i] > 0) {
				scorep.get(nump).score = score[i];
				scorep.get(nump).tiebreak = 0;
				scorep.get(nump).id = i;
				nump++;
			} else {
				scoren.get(numn).score = score[i];
				scoren.get(numn).tiebreak = 0;
				scoren.get(numn).id = i;
				numn++;
			}
		}

		// compute score of target label
		score_y = 0;
		if (loss_type == LearnStruct.SLACK_RESCALING) {
			for (i = 0; i < x.totdoc; i++)
				score_y += score[i];
		}

		if (nump != 0) {
			Collections.sort(scorep);
			// qsort(scorep,nump,sizeof(STRUCT_ID_SCORE),comparedown);
		}
		sump[0] = 0;
		for (i = 0; i < nump; i++) {
			sump[i + 1] = sump[i] + scorep.get(i).score;
		}
		if (numn != 0) {
			Collections.sort(scoren);
			// qsort(scoren,numn,sizeof(STRUCT_ID_SCORE),compareup);
		}
		sumn[0] = 0;
		for (i = 0; i < numn; i++) {
			sumn[i + 1] = sumn[i] + scoren.get(i).score;
		}

		// find max of loss(ybar,y)+score(ybar) for margin rescaling or max of
		// loss(ybar,y)+loss*(score(ybar)-score(y)) for slack rescaling
		valmax = 0;
		start = 1;
		prec_rec_k = (int) (nump * sparm.prec_rec_k_frac);
		if (prec_rec_k < 1)
			prec_rec_k = 1;
		for (a = 0; a <= nump; a++) {
			for (d = 0; d <= numn; d++) {
				if (sparm.loss_function == ModelConstant.ZEROONE)
					loss = zerooneLoss(a, numn - d, nump - a, d);
				else if (sparm.loss_function == ModelConstant.FONE)
					loss = foneLoss(a, numn - d, nump - a, d);
				else if (sparm.loss_function == ModelConstant.ERRORRATE)
					loss = errorrateLoss(a, numn - d, nump - a, d);
				else if ((sparm.loss_function == ModelConstant.PRBEP)
						&& (a + numn - d == nump))
					loss = prbepLoss(a, numn - d, nump - a, d);
				else if ((sparm.loss_function == ModelConstant.PREC_K)
						&& (a + numn - d >= prec_rec_k))
					loss = precKLoss(a, numn - d, nump - a, d);
				else if ((sparm.loss_function == ModelConstant.REC_K)
						&& (a + numn - d <= prec_rec_k))
					loss = recKLoss(a, numn - d, nump - a, d);
				else {
					loss = 0;
				}
				if (loss > 0) {
					if (loss_type == LearnStruct.SLACK_RESCALING) {
						val = loss
								+ loss
								* (sump[a] - (sump[nump] - sump[a]) - sumn[d] + (sumn[numn]
										- sumn[d] - score_y));
					} else if (loss_type == LearnStruct.MARGIN_RESCALING) {
						val = loss + sump[a] - (sump[nump] - sump[a]) - sumn[d]
								+ (sumn[numn] - sumn[d]);
					} else {
						System.err.printf("ERROR: Unknown loss type '%d'.\n",
								loss_type);
						System.exit(1);
					}
					if ((val > valmax) || (start != 0)) {
						start = 0;
						valmax = val;
						threshp = a;
						threshn = d;
					}
				}
			}
		}

		// assign labels that maximize score
		for (i = 0; i < nump; i++) {
			if (i < threshp)
				ybar.class_indexs[scorep.get(i).id] = y.class_indexs[scorep
						.get(i).id];
			else
				ybar.class_indexs[scorep.get(i).id] = -y.class_indexs[scorep
						.get(i).id];
		}
		for (i = 0; i < numn; i++) {
			if (i < threshn)
				ybar.class_indexs[scoren.get(i).id] = y.class_indexs[scoren
						.get(i).id];
			else
				ybar.class_indexs[scoren.get(i).id] = -y.class_indexs[scoren
						.get(i).id];
		}

		if (CommonStruct.struct_verbosity >= 2) {

			if (loss_type == LearnStruct.SLACK_RESCALING)
				System.out
						.printf("\n max_ybar {loss(y_i,ybar)+loss(y_i,ybar)[w*Psi(x,ybar)-w*Psi(x,y)]}=%f\n",
								valmax);
			else
				System.out.printf(
						"\n max_ybar {loss(y_i,ybar)+w*Psi(x,ybar)}=%f\n",
						valmax);
			SVECTOR fy = psi(x, y, sm, sparm);
			SVECTOR fybar = psi(x, ybar, sm, sparm);
			DOC exy = com.createExample(0, 0, 1, 1, fy);
			DOC exybar = com.createExample(0, 0, 1, 1, fybar);
			System.out.printf(" -> w*Psi(x,y_i)=%f, w*Psi(x,ybar)=%f\n",
					com.classifyExample(sm.svm_model, exy),
					com.classifyExample(sm.svm_model, exybar));

		}

		return (ybar);
	}

}
