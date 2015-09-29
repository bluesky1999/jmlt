package org.click.classify.svmstruct.model;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import org.click.classify.svmstruct.data.CheckStruct;
import org.click.classify.svmstruct.data.DOC;
import org.click.classify.svmstruct.data.KERNEL_PARM;
import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.MODEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.QP;
import org.click.classify.svmstruct.data.SVECTOR;
import org.click.classify.svmstruct.data.WORD;

import org.click.classify.svmstruct.model.Common;

/**
 * Learning module of Support Vector Machine
 * 
 * @author lq
 * 
 */
public class Learn {

	// private static Logger logger = Logger.getLogger(Learn.class);

	public static final int MAXSHRINK = 50000;

	public Common com = null;

	public Learn() {
		com = new Common();
	}

	public void svm_learn_optimization(DOC[] docs, double[] rhs, int totdoc, int totwords, LEARN_PARM learn_parm, KERNEL_PARM kernel_parm, MODEL model, double[] alpha) {

		int i;
		int[] label;
		double[] lin;
		double[] a;
		double[] c;
		int iterations, svsetnum = 0;
		int maxslackid;
		double r_delta_avg;
		int[] index;
		int[] index2dnum;
		double[] weights;
		double[] slack;
		double[] alphaslack;

		double[] aicache;

		CheckStruct checkStruct = new CheckStruct();


		///kernel_cache_statistic = 0;
		learn_parm.totwords = totwords;
		if ((learn_parm.svm_newvarsinqp < 2) || (learn_parm.svm_newvarsinqp > learn_parm.svm_maxqpsize)) {
			learn_parm.svm_newvarsinqp = learn_parm.svm_maxqpsize;
		}

		label = new int[totdoc];
		c = new double[totdoc];
		a = new double[totdoc];
		lin = new double[totdoc];

		learn_parm.svm_cost = new double[totdoc];
		///model.supvec = new DOC[totdoc + 2];
		model.supvec = new DOC[totdoc + 1];
		///model.alpha = new double[totdoc + 2];
		model.alpha = new double[totdoc + 1];

		///model.index = new int[totdoc + 2];
		model.index = new int[totdoc + 1];
		model.at_upper_bound = 0;
		model.b = 0;
		model.supvec[0] = null;
		model.alpha[0] = 0;
		model.lin_weights = null;
		model.totwords = totwords;
		model.totdoc = totdoc;
		model.kernel_parm = kernel_parm;
		model.sv_num = 1;


		r_delta_avg = estimate_r_delta_average(docs, totdoc, kernel_parm);

		if (learn_parm.svm_c == 0) {

			learn_parm.svm_c = 1.0 / (r_delta_avg * r_delta_avg);
		}

		learn_parm.biased_hyperplane = 0;

		learn_parm.eps = 0.0;

		for (i = 0; i < totdoc; i++) {
			docs[i].docnum = i;
			a[i] = 0;
			lin[i] = 0;
			c[i] = rhs[i];
			learn_parm.svm_cost[i] = learn_parm.svm_c * learn_parm.svm_costratio * docs[i].costfactor;
			label[i] = 1;
		}

		if (learn_parm.sharedslack != 0) {
			for (i = 0; i < totdoc; i++) {
				if (docs[i].slackid == 0) {
					System.err.println("Error: Missing shared slacks definitions in some of the examples.");
					System.exit(0);
				}
			}
		}

		if (alpha != null) {

			index = new int[totdoc];
			index2dnum = new int[totdoc + 11];
			if (kernel_parm.kernel_type == ModelConstant.LINEAR) {
				weights = new double[totwords + 1];
				com.clearNvector(weights, totwords);
				aicache = null;
			} else {
				weights = null;
				aicache = new double[totdoc];
			}

			for (i = 0; i < totdoc; i++) {
				index[i] = 1;
				alpha[i] = Math.abs(alpha[i]);
				if (alpha[i] < 0) {
					alpha[i] = 0;
				}

				if (alpha[i] > learn_parm.svm_cost[i]) {
					alpha[i] = learn_parm.svm_cost[i];
				}
			}

			compute_index(index, totdoc, index2dnum);
			update_linear_component(docs, label, index2dnum, alpha, a, index2dnum, totdoc, totwords, kernel_parm, lin, aicache, weights);

			calculate_svm_model(docs, label,  lin, alpha, a, c, learn_parm, index2dnum, index2dnum, model);
			for (i = 0; i < totdoc; i++) {
				a[i] = alpha[i];
			}

		}

		if (learn_parm.remove_inconsistent != 0) {
			learn_parm.remove_inconsistent = 0;
			System.out.println("'remove inconsistent' not available in this mode. Switching option off!");
		}

		if (learn_parm.sharedslack != 0) {
			iterations = optimize_to_convergence_sharedslack(docs, label, totdoc, totwords, learn_parm, kernel_parm, model, a, lin, c, checkStruct);
		} else {
			iterations = optimize_to_convergence(docs, label, totdoc, totwords, learn_parm, kernel_parm, model, a, lin, c, -1, 1, checkStruct);
		}

		if (learn_parm.sharedslack != 0) {
			index = new int[totdoc];
			index2dnum = new int[totdoc + 11];
			maxslackid = 0;
			for (i = 0; i < totdoc; i++) { // create full index
				index[i] = 1;
				if (maxslackid < docs[i].slackid) {
					maxslackid = docs[i].slackid;
				}
			}
			compute_index(index, totdoc, index2dnum);
			slack = new double[maxslackid + 1];
			alphaslack = new double[maxslackid + 1];
			for (i = 0; i <= maxslackid; i++) { // init shared slacks
				slack[i] = 0;
				alphaslack[i] = 0;
			}
			for (i = 0; i < totdoc; i++) { // compute alpha aggregated by slack
				alphaslack[docs[i].slackid] += a[i];
			}
			compute_shared_slacks(docs, label, a, lin, c, index2dnum, learn_parm, slack, alphaslack);
			//loss = 0;
			model.at_upper_bound = 0;
			svsetnum = 0;
			for (i = 0; i <= maxslackid; i++) { // create full index
				//loss += slack[i];
				if (alphaslack[i] > (learn_parm.svm_c - learn_parm.epsilon_a)) {
					model.at_upper_bound++;
				}
				if (alphaslack[i] > learn_parm.epsilon_a)
					svsetnum++;
			}
		}

		if (alpha != null) {
			for (i = 0; i < totdoc; i++) { // copy final alphas
				alpha[i] = a[i];
			}
		}

		if (learn_parm.alphafile != null) {
			write_alphas(learn_parm.alphafile, a, label, totdoc);
		}

	}

	public void update_linear_component(DOC[] docs, int[] label, double[] a, double[] a_old, int[] working2dnum, int totdoc, int totwords, KERNEL_PARM kernel_parm, double[] lin, double[] aicache, double[] weights) {
		int i, ii,  jj;
		double tec;
		SVECTOR f;

		if (kernel_parm.kernel_type == 0) {

			for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
				if (a[i] != a_old[i]) {
					for (f = docs[i].fvec; f != null; f = f.next) {
						com.addVectorNs(weights, f, f.factor * ((a[i] - a_old[i]) * label[i]));

					}
				}
			}

			for (jj = 0; jj<lin.length; jj++) {
				for (f = docs[jj].fvec; f != null; f = f.next) {
					lin[jj] += f.factor * com.sprodNs(weights, f);
				}
			}

			for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
				if (a[i] != a_old[i]) {
					for (f = docs[i].fvec; f != null; f = f.next) {
						com.multVectorNs(weights, f, 0.0);
					}
				}
			}
		} else {
			for (jj = 0; jj<a.length; jj++) {
				if (a[jj] != a_old[jj]) {
					get_kernel_row(docs, jj, aicache, kernel_parm);
					for (ii = 0; ii<lin.length; ii++) {
						tec = aicache[ii];
						lin[ii] += (((a[jj] * tec) - (a_old[jj] * tec)) * (double) label[jj]);

					}
				}
			}
		}

	}
	
	public void update_linear_component(DOC[] docs, int[] label, int[] active2dnum, double[] a, double[] a_old, int[] working2dnum, int totdoc, int totwords, KERNEL_PARM kernel_parm, double[] lin, double[] aicache, double[] weights) {
		int i, ii, j, jj;
		double tec;
		SVECTOR f;

		if (kernel_parm.kernel_type == 0) {

			for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
				if (a[i] != a_old[i]) {
					for (f = docs[i].fvec; f != null; f = f.next) {
						com.addVectorNs(weights, f, f.factor * ((a[i] - a_old[i]) * label[i]));

					}
				}
			}

			for (jj = 0; (j = active2dnum[jj]) >= 0; jj++) {
				for (f = docs[j].fvec; f != null; f = f.next) {
					lin[j] += f.factor * com.sprodNs(weights, f);
				}
			}

			for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
				if (a[i] != a_old[i]) {
					for (f = docs[i].fvec; f != null; f = f.next) {
						com.multVectorNs(weights, f, 0.0);
					}
				}
			}
		} else {
			for (jj = 0; (i = working2dnum[jj]) >= 0; jj++) {
				if (a[i] != a_old[i]) {
					get_kernel_row(docs, i, aicache, kernel_parm);
					for (ii = 0; (j = active2dnum[ii]) >= 0; ii++) {
						tec = aicache[j];
						lin[j] += (((a[i] * tec) - (a_old[i] * tec)) * (double) label[i]);

					}
				}
			}
		}

	}

	public double estimate_r_delta_average(DOC[] docs, int totdoc, KERNEL_PARM kernel_parm) {
		int i;
		double avgxlen;
		DOC nulldoc;
		WORD[] nullword = new WORD[1];
		nullword[0] = new WORD();
		nullword[0].wnum = 0;
		nulldoc = com.createExample(-2, 0, 0, 0.0, com.createSvector(nullword, "", 1.0));
		avgxlen = 0;

		for (i = 0; i < totdoc; i++) {
			avgxlen += Math.sqrt(com.kernel(kernel_parm, docs[i], docs[i]) - 2 * com.kernel(kernel_parm, docs[i], nulldoc) + com.kernel(kernel_parm, nulldoc, nulldoc));
		}

		return (avgxlen / totdoc);
	}

	//shrink_state.active, totdoc, active2dnum
	public int compute_index(int[] binfeature, int range, int[] index) {
		int i, ii;
		ii = 0;

		for (i = 0; i < range; i++) {
			if (binfeature[i] != 0) {
				index[ii] = i;
				ii++;
			}
		}

		for (i = 0; i < 4; i++) {
			index[ii + i] = -1;
		}

		return ii;
	}

	public void get_kernel_row(DOC[] docs, int docix,  double[] buffer, KERNEL_PARM kernel_parm) {
		int i;
		DOC ex;
		ex = docs[docix];

		for (i = 0; i<docs.length; i++) {

			buffer[i] = com.kernel(kernel_parm, ex, docs[i]);
		}
	}

	public int calculate_svm_model(DOC[] docs, int[] label,  double[] lin, double[] a, double[] a_old, double[] c, LEARN_PARM learn_parm, int[] working2dnum, int[] active2dnum, MODEL model) {
		int i, ii, pos, b_calculated = 0, first_low, first_high;
		double ex_c, b_temp, b_low, b_high;

		if (learn_parm.biased_hyperplane == 0) {
			model.b = 0;
			b_calculated = 1;
		}

		for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
			if ((a_old[i] > 0) && (a[i] == 0)) {
				// remove from model
				pos = model.index[i];
				model.index[i] = -1;
				(model.sv_num)--;

				model.supvec[pos] = model.supvec[model.sv_num];
				model.alpha[pos] = model.alpha[model.sv_num];

				model.index[model.supvec[pos].docnum] = pos;
			} else if ((a_old[i] == 0) && (a[i] > 0)) {
				// add to model
				model.supvec[model.sv_num] = docs[i];
				model.alpha[model.sv_num] = a[i] * ((double) label[i]);
				model.index[i] = model.sv_num;
				(model.sv_num)++;
			} else if (a_old[i] == a[i]) {

			} else {
				model.alpha[model.index[i]] = a[i] * ((double) label[i]);
			}

			ex_c = learn_parm.svm_cost[i] - learn_parm.epsilon_a;

			if (learn_parm.sharedslack == 0) {
				if ((a_old[i] >= ex_c) && (a[i] < ex_c)) {
					(model.at_upper_bound)--;
				} else if ((a_old[i] < ex_c) && (a[i] >= ex_c)) {
					(model.at_upper_bound)++;
				}
			}

			if ((b_calculated == 0) && (a[i] > learn_parm.epsilon_a) && (a[i] < ex_c)) {
				model.b = ((double) label[i]) * learn_parm.eps - c[i] + lin[i];
				b_calculated = 1;
			}

		}

		// No alpha in the working set not at bounds, so b was not calculated in
		// the usual way. The following handles this special case.

		if ((learn_parm.biased_hyperplane != 0) && (b_calculated == 0) && ((model.sv_num - 1) == model.at_upper_bound)) {
			first_low = 1;
			first_high = 1;
			b_low = 0;
			b_high = 0;

			for (ii = 0; (i = active2dnum[ii]) >= 0; ii++) {
				ex_c = learn_parm.svm_cost[i] - learn_parm.epsilon_a;
				if (a[i] < ex_c) {
					if (label[i] > 0) {
						b_temp = -(learn_parm.eps - c[i] + lin[i]);
						if ((b_temp > b_low) || (first_low != 0)) {
							b_low = b_temp;
							first_low = 0;
						}
					} else {
						b_temp = -(-learn_parm.eps - c[i] + lin[i]);
						if ((b_temp < b_high) || (first_high != 0)) {
							b_high = b_temp;
							first_high = 0;
						}
					}
				} else {
					if (label[i] < 0) {
						b_temp = -(-learn_parm.eps - c[i] + lin[i]);
						if ((b_temp > b_low) || (first_low != 0)) {
							b_low = b_temp;
							first_low = 0;
						}
					} else {
						b_temp = -(learn_parm.eps - c[i] + lin[i]);
						if ((b_temp < b_high) || (first_high != 0)) {
							b_high = b_temp;
							first_high = 0;
						}
					}
				}
			}// for

			if (first_high != 0) {
				model.b = -b_low;
			} else if (first_low != 0) {
				model.b = -b_high;
			} else {
				model.b = -(b_high + b_low) / 2.0;
			}

		}

		return (model.sv_num - 1);
	}

	public int calculate_svm_model(DOC[] docs, int[] label,  double[] lin, double[] a, double[] a_old, double[] c, LEARN_PARM learn_parm, int[] working2dnum,  MODEL model) {
		int i, ii, pos, b_calculated = 0, first_low, first_high;
		double ex_c, b_temp, b_low, b_high;

		if (learn_parm.biased_hyperplane == 0) {
			model.b = 0;
			b_calculated = 1;
		}

		for (ii = 0; (i = working2dnum[ii]) >= 0; ii++) {
			if ((a_old[i] > 0) && (a[i] == 0)) {
				// remove from model
				pos = model.index[i];
				model.index[i] = -1;
				(model.sv_num)--;

				model.supvec[pos] = model.supvec[model.sv_num];
				model.alpha[pos] = model.alpha[model.sv_num];

				model.index[model.supvec[pos].docnum] = pos;
			} else if ((a_old[i] == 0) && (a[i] > 0)) {
				// add to model
				model.supvec[model.sv_num] = docs[i];
				model.alpha[model.sv_num] = a[i] * ((double) label[i]);
				model.index[i] = model.sv_num;
				(model.sv_num)++;
			} else if (a_old[i] == a[i]) {

			} else {
				model.alpha[model.index[i]] = a[i] * ((double) label[i]);
			}

			ex_c = learn_parm.svm_cost[i] - learn_parm.epsilon_a;

			if (learn_parm.sharedslack == 0) {
				if ((a_old[i] >= ex_c) && (a[i] < ex_c)) {
					(model.at_upper_bound)--;
				} else if ((a_old[i] < ex_c) && (a[i] >= ex_c)) {
					(model.at_upper_bound)++;
				}
			}

			if ((b_calculated == 0) && (a[i] > learn_parm.epsilon_a) && (a[i] < ex_c)) {
				model.b = ((double) label[i]) * learn_parm.eps - c[i] + lin[i];
				b_calculated = 1;
			}

		}

		// No alpha in the working set not at bounds, so b was not calculated in
		// the usual way. The following handles this special case.

		if ((learn_parm.biased_hyperplane != 0) && (b_calculated == 0) && ((model.sv_num - 1) == model.at_upper_bound)) {
			first_low = 1;
			first_high = 1;
			b_low = 0;
			b_high = 0;

			for (ii = 0; ii<lin.length; ii++) {
				ex_c = learn_parm.svm_cost[ii] - learn_parm.epsilon_a;
				if (a[ii] < ex_c) {
					if (label[ii] > 0) {
						b_temp = -(learn_parm.eps - c[ii] + lin[ii]);
						if ((b_temp > b_low) || (first_low != 0)) {
							b_low = b_temp;
							first_low = 0;
						}
					} else {
						b_temp = -(-learn_parm.eps - c[ii] + lin[ii]);
						if ((b_temp < b_high) || (first_high != 0)) {
							b_high = b_temp;
							first_high = 0;
						}
					}
				} else {
					if (label[ii] < 0) {
						b_temp = -(-learn_parm.eps - c[ii] + lin[ii]);
						if ((b_temp > b_low) || (first_low != 0)) {
							b_low = b_temp;
							first_low = 0;
						}
					} else {
						b_temp = -(learn_parm.eps - c[ii] + lin[ii]);
						if ((b_temp < b_high) || (first_high != 0)) {
							b_high = b_temp;
							first_high = 0;
						}
					}
				}
			}// for

			if (first_high != 0) {
				model.b = -b_low;
			} else if (first_low != 0) {
				model.b = -b_high;
			} else {
				model.b = -(b_high + b_low) / 2.0;
			}

		}

		return (model.sv_num - 1);
	}
	
	public int optimize_to_convergence(DOC[] docs, int[] label, int totdoc, int totwords, LEARN_PARM learn_parm, KERNEL_PARM kernel_parm,  MODEL model,   double[] a, double[] lin, double[] c, int heldout, int retrain, CheckStruct struct) {

		int[] chosen;
		int[] key;
		int i, j, jj;
		int[] last_suboptimal_at;
		CommonStruct.verbosity = 0;
		int  choosenum, already_chosen = 0, iteration;
		int supvecnum = 0;
	
		int[] working2dnum;
		int[] selexam;
		double eq;
		double[] a_old;

		double epsilon_crit_org;
		double bestmaxdiff;
		int bestmaxdiffiter, terminate;

		double[] selcrit;
		double[] aicache;
		double[] weights;
		QP qp = null;

		qp = new QP();

		epsilon_crit_org = learn_parm.epsilon_crit;
		if (kernel_parm.kernel_type == ModelConstant.LINEAR) {
			learn_parm.epsilon_crit = 2.0;
		}
		learn_parm.epsilon_shrink = 2;
		struct.maxdiff = 1;

		learn_parm.totwords = totwords;

		chosen = new int[totdoc];
		last_suboptimal_at = new int[totdoc];
		key = new int[totdoc + 11];
		selcrit = new double[totdoc];
		selexam = new int[totdoc];
		a_old = new double[totdoc];
		aicache = new double[totdoc];
		working2dnum = new int[totdoc + 11];

		qp.opt_ce = new double[learn_parm.svm_maxqpsize];
		qp.opt_ce0 = new double[1];
		qp.opt_g = new double[(learn_parm.svm_maxqpsize) * (learn_parm.svm_maxqpsize)];
		qp.opt_g0 = new double[learn_parm.svm_maxqpsize];
		qp.opt_xinit = new double[learn_parm.svm_maxqpsize];
		qp.opt_low = new double[learn_parm.svm_maxqpsize];
		qp.opt_up = new double[learn_parm.svm_maxqpsize];

		if (kernel_parm.kernel_type == ModelConstant.LINEAR) {
			weights = com.createNvector(totwords);
			com.clearNvector(weights, totwords);
		} else {
			weights = null;
		}

		choosenum = 0;

		if (retrain == 0) {
			retrain = 1;
		}

		iteration = 1;
		bestmaxdiffiter = 1;
		bestmaxdiff = 999999999;
		terminate = 0;

		for (i = 0; i < totdoc; i++) {
			chosen[i] = 0;
			a_old[i] = a[i];
			last_suboptimal_at[i] = 1;
		}

		clear_index(working2dnum);

		////// main loop 
		for (; (retrain != 0) && (terminate == 0); iteration++) {

			if (learn_parm.svm_newvarsinqp > learn_parm.svm_maxqpsize) {
				learn_parm.svm_newvarsinqp = learn_parm.svm_maxqpsize;
			}

			i = 0;

			for (jj = 0; (j = working2dnum[jj]) >= 0; jj++) {

				if ((chosen[j] >= (learn_parm.svm_maxqpsize / Math.min(learn_parm.svm_maxqpsize, learn_parm.svm_newvarsinqp)))  || (j == heldout)) {
					chosen[j] = 0;
					choosenum--;
				} else {
					chosen[j]++;
					working2dnum[i++] = j;
				}
			}

			working2dnum[i] = -1;
			if (retrain == 2) {
				choosenum = 0;
				for (jj = 0; (j = working2dnum[jj]) >= 0; jj++) {
					chosen[j] = 0;
				}

				clear_index(working2dnum);

				for (i = 0; i < totdoc; i++) {
					if (( (heldout == i)) && (a[i] != 0)) {
						chosen[i] = 99999;
						choosenum++;
						a[i] = 0;
					}
				}

				if (learn_parm.biased_hyperplane != 0) {
					eq = 0;
					for (i = 0; i < totdoc; i++) {
						eq += a[i] * label[i];
					}

					for (i = 0; (i < totdoc) && (Math.abs(eq) > learn_parm.epsilon_a); i++) {
						if ((eq * label[i] > 0) && (a[i] > 0)) {
							chosen[i] = 88888;
							choosenum++;
							if ((eq * label[i]) > a[i]) {
								eq -= (a[i] * label[i]);
								a[i] = 0;
							} else {
								a[i] -= (eq * label[i]);
								eq = 0;
							}
						}
					}
				}
				compute_index(chosen, totdoc, working2dnum);

			}// retrain==2
			else {
				if ((iteration % 101) != 0) {
					already_chosen = 0;

					if ((Math.min(learn_parm.svm_newvarsinqp, learn_parm.svm_maxqpsize - choosenum) >= 4) && (kernel_parm.kernel_type != ModelConstant.LINEAR)) {
						already_chosen = select_next_qp_subproblem_grad(label,  a, lin, c, totdoc, (Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp) / 2), learn_parm,  working2dnum, selcrit, selexam, 1, key, chosen);

						choosenum += already_chosen;
					}

					choosenum += select_next_qp_subproblem_grad(label, a, lin, c, totdoc, Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp - already_chosen), learn_parm,  working2dnum, selcrit, selexam, 0, key, chosen);

				} else {
					choosenum += select_next_qp_subproblem_rand(label,  a, lin, c, totdoc, Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp), learn_parm,  working2dnum, selcrit, selexam, key, chosen, iteration);
				}
			}

			if (retrain != 2) {
				optimize_svm(docs, label,  0.0, chosen,  model, totdoc, working2dnum, choosenum, a, lin, c, learn_parm, aicache, kernel_parm, qp, epsilon_crit_org);
			}

			update_linear_component(docs, label, a, a_old, working2dnum, totdoc, totwords, kernel_parm, lin, aicache, weights);

			supvecnum = calculate_svm_model(docs, label, lin, a, a_old, c, learn_parm, working2dnum,  model);

			for (jj = 0; (i = working2dnum[jj]) >= 0; jj++) {
				a_old[i] = a[i];
			}

			retrain = check_optimality(model, label,  a, lin, c, totdoc, learn_parm, epsilon_crit_org,   last_suboptimal_at, iteration, kernel_parm, struct);

			// checking whether optimizer got stuck
			if (struct.maxdiff < bestmaxdiff) {
				bestmaxdiff = struct.maxdiff;
				bestmaxdiffiter = iteration;
			}

			if (iteration > (bestmaxdiffiter + learn_parm.maxiter)) {
				// long time no progress?
				terminate = 1;
				retrain = 0;
			}

			if ((retrain == 0) && ((kernel_parm.kernel_type == ModelConstant.LINEAR))) {

				// reset watchdog
				bestmaxdiff = struct.maxdiff;
				bestmaxdiffiter = iteration;

				// termination criterion
				retrain = 0;
				if (struct.maxdiff > learn_parm.epsilon_crit) {
					retrain = 1;
				}
			}

			if ((retrain == 0) && (learn_parm.epsilon_crit > struct.maxdiff)) {
				learn_parm.epsilon_crit = struct.maxdiff;
			}

			if ((retrain == 0) && (learn_parm.epsilon_crit > epsilon_crit_org)) {
				learn_parm.epsilon_crit /= 2.0;
				retrain = 1;
			}
			if (learn_parm.epsilon_crit < epsilon_crit_org) {
				learn_parm.epsilon_crit = epsilon_crit_org;
			}
		
		}// end of loop

		learn_parm.epsilon_crit = epsilon_crit_org;
		model.maxdiff = struct.maxdiff;

		return iteration;
	}

	public void clear_index(int[] index) {
		for (int i = 0; i < index.length; i++) {
			index[i] = -1;
		}
	}

	/**
	 * Working set selection
	 * 
	 * @param label
	 * @param a
	 * @param lin
	 * @param c
	 * @param totdoc
	 * @param qp_size
	 * @param learn_parm
	 * @param inconsistent
	 * @param active2dnum
	 * @param working2dnum
	 * @param selcrit
	 * @param select
	 * @param cache_only
	 * @param key
	 * @param chosen
	 * @return
	 */
	public int select_next_qp_subproblem_grad(int[] label,  double[] a, double[] lin, double[] c, int totdoc, int qp_size, LEARN_PARM learn_parm,  int[] working2dnum, double[] selcrit, int[] select, int cache_only, int[] key, int[] chosen) {

		int choosenum, i, k, activedoc, inum, valid;
		double s;

		for (inum = 0; working2dnum[inum] >= 0; inum++)
			;
		choosenum = 0;
		activedoc = 0;

		for (i = 0; i<label.length; i++) {
			s = -label[i];

			valid = 1;

			if ((valid != 0) && (!((a[i] <= (0 + learn_parm.epsilon_a)) && (s < 0))) && (!((a[i] >= (learn_parm.svm_cost[i] - learn_parm.epsilon_a)) && (s > 0))) && (chosen[i] == 0) && (label[i] != 0)) {

				selcrit[activedoc] = (double) label[i] * (learn_parm.eps - (double) label[i] * c[i] + (double) label[i] * lin[i]);

				if (Math.abs(selcrit[activedoc]) > (double) (0.5)) {
					key[activedoc] = i;
					activedoc++;
				}
			}
		}
		
		

		select_top_n(selcrit, activedoc, select, qp_size / 2);

		for (k = 0; (choosenum < ((qp_size) / 2)) && (k < (qp_size / 2)) && (k < activedoc); k++) {

			i = key[select[k]];
			chosen[i] = 1;
			working2dnum[inum + choosenum] = i;

			choosenum += 1;

		}

		activedoc = 0;
		for (i = 0; i<label.length; i++) {
			s = label[i];
			valid = 1;

			if (valid != 0 && (!((a[i] <= (0 + learn_parm.epsilon_a)) && (s < 0))) && (!((a[i] >= (learn_parm.svm_cost[i] - learn_parm.epsilon_a)) && (s > 0))) && (chosen[i] == 0) && (label[i] != 0) ) {
				selcrit[activedoc] = -(double) label[i] * (learn_parm.eps - (double) label[i] * c[i] + (double) label[i] * lin[i]);

				if (Math.abs(selcrit[activedoc]) > (double) (0.5)) {
					key[activedoc] = i;
					activedoc++;
				}
			}
		}
		

		select_top_n(selcrit, activedoc, select, qp_size / 2);

		for (k = 0; (choosenum < qp_size) && (k < ((qp_size) / 2)) && (k < activedoc); k++) {
			i = key[select[k]];

			chosen[i] = 1;
			working2dnum[inum + choosenum] = i;
			choosenum += 1;
		}

		working2dnum[inum + choosenum] = -1; // complete index
		return (choosenum);
	}

	void select_top_n(double[] selcrit, int range, int[] select, int n) {

		int i, j;

		for (i = 0; (i < n) && (i < range); i++) {
			for (j = i; j >= 0; j--) {
				if ((j > 0) && (selcrit[select[j - 1]] < selcrit[i])) {
					select[j] = select[j - 1];
				} else {
					select[j] = i;
					j = -1;
				}
			}
		}

		if (n > 0) {
			for (i = n; i < range; i++) {

				if (selcrit[i] > selcrit[select[n - 1]]) {
					for (j = n - 1; j >= 0; j--) {
						if ((j > 0) && (selcrit[select[j - 1]] < selcrit[i])) {
							select[j] = select[j - 1];

						} else {
							select[j] = i;
							j = -1;
						}
					}
				}

			}
		}
	}

	public int select_next_qp_subproblem_rand(int[] label,  double[] a, double[] lin, double[] c, int totdoc, int qp_size, LEARN_PARM learn_parm, int[] working2dnum, double[] selcrit, int[] select, int[] key, int[] chosen, int iteration) {
		int choosenum, i,  k, activedoc, inum;
		double s = 0;

		for (inum = 0; working2dnum[inum] >= 0; inum++)
			;
		choosenum = 0;
		activedoc = 0;

		for (i = 0;i<label.length; i++) {
			s -= label[i];

			if ((!((a[i] <= (0 + learn_parm.epsilon_a)) && (s < 0))) && (!((a[i] >= (learn_parm.svm_cost[i] - learn_parm.epsilon_a)) && (s > 0))) && (label[i] != 0) && (chosen[i] == 0)) {
				selcrit[activedoc] = (i + iteration) % totdoc;
				key[activedoc] = i;
				activedoc++;
			}
		}

		select_top_n(selcrit, activedoc, select, (qp_size / 2));

		for (k = 0; (choosenum < (qp_size / 2)) && (k < (qp_size / 2)) && (k < activedoc); k++) {
			i = key[select[k]];
			chosen[i] = 1;
			working2dnum[inum + choosenum] = i;
			choosenum += 1;
		}

		activedoc = 0;
		for (i = 0; i<label.length; i++) {
			s = label[i];
			if ((!((a[i] <= (0 + learn_parm.epsilon_a)) && (s < 0))) && (!((a[i] >= (learn_parm.svm_cost[i] - learn_parm.epsilon_a)) && (s > 0))) && (label[i] != 0) && (chosen[i] == 0)) {
				selcrit[activedoc] = (i + iteration) % totdoc;
				key[activedoc] = i;
				activedoc++;
			}
		}

		select_top_n(selcrit, activedoc, select, (qp_size / 2));

		for (k = 0; (choosenum < qp_size) && (k < (qp_size / 2)) && (k < activedoc); k++) {
			i = key[select[k]];
			chosen[i] = 1;
			working2dnum[inum + choosenum] = i;
			choosenum += 1;
		}
		working2dnum[inum + choosenum] = -1; // complete index
		return (choosenum);

	}

	public void optimize_svm(DOC[] docs, int[] label,  double eq_target, int[] chosen, MODEL model, int totdoc, int[] working2dnum, int varnum, double[] a, double[] lin, double[] c, LEARN_PARM learn_parm, double[] aicache, KERNEL_PARM kernel_parm, QP qp, double epsilon_crit_target) {
		int i;
		double[] a_v;

		compute_matrices_for_optimization(docs, label,  eq_target, chosen, working2dnum, model, a, lin, c, varnum, totdoc, learn_parm, aicache, kernel_parm, qp);

		// call the qp-subsolver
		Hideo shid = new Hideo();

		a_v = shid.optimizeQp(qp, epsilon_crit_target, learn_parm.svm_maxqpsize, (model.b), learn_parm);

		for (i = 0; i < varnum; i++) {
			a[working2dnum[i]] = a_v[i];
		}

	}

	public void compute_matrices_for_optimization(DOC[] docs, int[] label,  double eq_target, int[] chosen,  int[] key, MODEL model, double[] a, double[] lin, double[] c, int varnum, int totdoc, LEARN_PARM learn_parm, double[] aicache, KERNEL_PARM kernel_parm, QP qp) {

		int ki, kj, i, j;
		double kernel_temp;

		qp.opt_n = varnum;
		qp.opt_ce0[0] = -eq_target;

		for (j = 1; j < model.sv_num; j++) {

			if ((chosen[model.supvec[j].docnum] == 0)) {
				qp.opt_ce0[0] += model.alpha[j];
			}
		}

		if (learn_parm.biased_hyperplane != 0) {
			qp.opt_m = 1;
		} else {
			qp.opt_m = 0;
		}

		for (i = 0; i < varnum; i++) {

			qp.opt_g0[i] = lin[key[i]];
		}

		for (i = 0; i < varnum; i++) {
			ki = key[i];

			qp.opt_ce[i] = label[ki];
			qp.opt_low[i] = 0;
			qp.opt_up[i] = learn_parm.svm_cost[ki];

			kernel_temp = com.kernel(kernel_parm, docs[ki], docs[ki]);
			qp.opt_g0[i] -= (kernel_temp * a[ki] * (double) label[ki]);
			qp.opt_g[varnum * i + i] = kernel_temp;

			for (j = i + 1; j < varnum; j++) {
				kj = key[j];
				kernel_temp = com.kernel(kernel_parm, docs[ki], docs[kj]);

				qp.opt_g0[i] -= (kernel_temp * a[kj] * (double) label[kj]);
				qp.opt_g0[j] -= (kernel_temp * a[ki] * (double) label[ki]);

				// compute quadratic part of objective function
				qp.opt_g[varnum * i + j] = (double) label[ki] * (double) label[kj] * kernel_temp;
				qp.opt_g[varnum * j + i] = (double) label[ki] * (double) label[kj] * kernel_temp;

			}

		}

		for (i = 0; i < varnum; i++) {
			// assure starting at feasible point
			qp.opt_xinit[i] = a[key[i]];
			// set linear part of objective function
			qp.opt_g0[i] = (learn_parm.eps - (double) label[key[i]] * c[key[i]]) + qp.opt_g0[i] * (double) label[key[i]];
		}

	}

	/**
	 * Return value of objective function. Works only relative to the active
	 * variables!
	 */
	public double compute_objective_function(double[] a, double[] lin, double[] c, double eps, int[] label, int[] active2dnum) {
		int i, ii;
		double criterion;
		// calculate value of objective function
		criterion = 0;
		for (ii = 0; active2dnum[ii] >= 0; ii++) {
			i = active2dnum[ii];
			criterion = criterion + (eps - (double) label[i] * c[i]) * a[i] + 0.5 * a[i] * label[i] * lin[i];
		}
		return (criterion);
	}

	public int check_optimality(MODEL model, int[] label,  double[] a, double[] lin, double[] c, int totdoc, LEARN_PARM learn_parm, double epsilon_crit_org,  int[] last_suboptimal_at, int iteration, KERNEL_PARM kernel_parm, CheckStruct struct) {
		int  ii, retrain;
		double dist = 0, ex_c, target;
		if (kernel_parm.kernel_type == ModelConstant.LINEAR) { // be optimistic
			learn_parm.epsilon_shrink = -learn_parm.epsilon_crit + epsilon_crit_org;
		} else { // be conservative
			learn_parm.epsilon_shrink = learn_parm.epsilon_shrink * 0.7 + (struct.maxdiff) * 0.3;
		}

		retrain = 0;
		struct.maxdiff = 0;
		struct.misclassified = 0;

		for (ii = 0; ii<lin.length; ii++) {
			if ( (label[ii] != 0)) {
				dist = (lin[ii] - model.b) * (double) label[ii];
				target = -(learn_parm.eps - (double) label[ii] * c[ii]);
				ex_c = learn_parm.svm_cost[ii] - learn_parm.epsilon_a;

				if (dist <= 0) {
					struct.misclassified++;
				}

				if ((a[ii] > learn_parm.epsilon_a) && (dist > target)) {
					if ((dist - target) > struct.maxdiff) {
						struct.maxdiff = dist - target;
					}
				} else if ((a[ii] < ex_c) && (dist < target)) {
					if ((target - dist) > struct.maxdiff) // largest violation
					{
						struct.maxdiff = target - dist;
					}
				}

				if ((a[ii] > (learn_parm.epsilon_a)) && (a[ii] < ex_c)) {
					last_suboptimal_at[ii] = iteration; // not at bound
				} else if ((a[ii] <= (learn_parm.epsilon_a)) && (dist < (target + learn_parm.epsilon_shrink))) {
					last_suboptimal_at[ii] = iteration; // not likely optimal
				} else if ((a[ii] >= ex_c) && (dist > (target - learn_parm.epsilon_shrink))) {
					last_suboptimal_at[ii] = iteration; // not likely optimal
				}
			}
		}

		// termination criterion
		if ((retrain == 0) && (struct.maxdiff > (learn_parm.epsilon_crit))) {
			retrain = 1;
		}
		return (retrain);

	}


	public void write_prediction(String predfile, MODEL model, double[] lin, double[] a, int[] label, int totdoc, LEARN_PARM learn_parm) {
		FileWriter fw = null;
		PrintWriter pw = null;
		try {
			fw = new FileWriter(new File(predfile));
			pw = new PrintWriter(fw);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		int i;
		double dist, a_max;

		a_max = learn_parm.epsilon_a;
		for (i = 0; i < totdoc; i++) {
			if ((a[i] > a_max)) {
				a_max = a[i];
			}
		}
		for (i = 0; i < totdoc; i++) {
			
				if ((a[i] > (learn_parm.epsilon_a))) {
					dist = (double) label[i] * (1.0 - learn_parm.epsilon_crit - a[i] / (a_max * 2.0));
				} else {
					dist = (lin[i] - model.b);
				}
				if (dist > 0) {
					System.out.println(dist + ":+1 " + (-dist) + ":-1");
				} else {
					System.out.println((-dist) + ":-1 " + dist + ":1");
				}
			
		}
		try {
			fw.close();
			pw.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

	}
	
	/***
	 * 此方法有问题
	 * 
	 * @param docs
	 * @param label
	 * @param totdoc
	 * @param totwords
	 * @param learn_parm
	 * @param kernel_parm
	 * @param kernel_cache
	 * @param model
	 * @param a
	 * @param lin
	 * @param c
	 * @return
	 */
	public int optimize_to_convergence_sharedslack(DOC[] docs, int[] label, int totdoc, int totwords, LEARN_PARM learn_parm, KERNEL_PARM kernel_parm,  MODEL model, double[] a, double[] lin, double[] c, CheckStruct struct) {

		int[] chosen;
		int[] key;
		int i, j, jj;
		int[] last_suboptimal_at;

		int choosenum, already_chosen = 0, iteration;
		int supvecnum = 0;	
		int[] working2dnum;
		int[] selexam;
		int[] ignore;
		int  retrain, maxslackid, slackset, jointstep;
		double criterion, eq_target;
		double[] a_old;
		double[] alphaslack;

		double epsilon_crit_org;
		double bestmaxdiff;
		int bestmaxdiffiter, terminate;

		double[] selcrit; // buffer for sorting
		double[] aicache; // buffer to keep one row of hessian
		double[] weights; // buffer for weight vector in linear case
		QP qp = new QP(); // buffer for one quadratic program
		double[] slack; // vector of slack variables for optimization with
						// shared slacks

		CommonStruct.verbosity = 0;
		epsilon_crit_org = learn_parm.epsilon_crit; //save org 
		if (kernel_parm.kernel_type == ModelConstant.LINEAR) {
			learn_parm.epsilon_crit = 2.0;
		}
		learn_parm.epsilon_shrink = 2;
		struct.maxdiff = 1;
		learn_parm.totwords = totwords;

		chosen = new int[totdoc];

		ignore = new int[totdoc];
		last_suboptimal_at = new int[totdoc];
		key = new int[totdoc + 11];
		selcrit = new double[totdoc];
		selexam = new int[totdoc];
		a_old = new double[totdoc];
		aicache = new double[totdoc];
		working2dnum = new int[totdoc + 11];
		//active2dnum = new int[totdoc + 11];
		qp.opt_ce = new double[learn_parm.svm_maxqpsize];
		qp.opt_ce0 = new double[1];
		qp.opt_g = new double[learn_parm.svm_maxqpsize * learn_parm.svm_maxqpsize];

		qp.opt_g0 = new double[learn_parm.svm_maxqpsize];
		qp.opt_xinit = new double[learn_parm.svm_maxqpsize];
		qp.opt_low = new double[learn_parm.svm_maxqpsize];
		qp.opt_up = new double[learn_parm.svm_maxqpsize];

		if (kernel_parm.kernel_type == ModelConstant.LINEAR) {
			weights = com.createNvector(totwords);
			com.clearNvector(weights, totwords);
		} else {
			weights = null;
		}

		maxslackid = 0;

		for (i = 0; i < totdoc; i++) { // determine size of slack array
			if (maxslackid < docs[i].slackid) {
				maxslackid = docs[i].slackid;
			}
		}

		slack = new double[maxslackid+1];
		alphaslack = new double[maxslackid + 1];

		for (i = 0; i < maxslackid; i++) {
			slack[i] = 0;
			alphaslack[i] = 0;
		}

		choosenum = 0;
		retrain = 1;
		iteration = 1;
		bestmaxdiffiter = 1;
		bestmaxdiff = 999999999;
		terminate = 0;

		for (i = 0; i < totdoc; i++) { // various inits
			chosen[i] = 0;

			ignore[i] = 0;
			alphaslack[docs[i].slackid] += a[i];
			a_old[i] = a[i];
			last_suboptimal_at[i] = 1;
		}

		clear_index(working2dnum);

		// call to init slack and alphaslack
		compute_shared_slacks(docs, label, a, lin, c,  learn_parm, slack, alphaslack);

		for (; (retrain != 0) && (terminate == 0); iteration++) {

			for (int ci = 0; ci < totdoc; ci++) {
				selcrit[ci] = 0.0;
			}

			if (learn_parm.svm_newvarsinqp > learn_parm.svm_maxqpsize) {
				learn_parm.svm_newvarsinqp = learn_parm.svm_maxqpsize;
			}

			// select working set according to steepest gradient
			jointstep = 0;
			eq_target = 0;

			if ((iteration % 101) != 0) {
				slackset = select_next_qp_slackset(docs, label, a, lin, slack, alphaslack, c, learn_parm,  struct);
				if (((iteration % 100) == 0) || (slackset == 0) || (struct.maxsharedviol < learn_parm.epsilon_crit)) {
					// do a step with examples from different slack sets
					i = 0;
					for (jj = 0; (j = working2dnum[jj]) >= 0; jj++) {
						if ((chosen[j] >= (learn_parm.svm_maxqpsize / Math.min(learn_parm.svm_maxqpsize, learn_parm.svm_newvarsinqp)))) {
							chosen[j] = 0;
							choosenum--;
						} else {
							chosen[j]++;
							working2dnum[i++] = j;
						}
					}
					working2dnum[i] = -1;

					already_chosen = 0;
					if ((Math.min(learn_parm.svm_newvarsinqp, learn_parm.svm_maxqpsize - choosenum) >= 4) && (kernel_parm.kernel_type != ModelConstant.LINEAR)) {
						// select part of the working set from cache
						already_chosen = select_next_qp_subproblem_grad(label,  a, lin, c, totdoc, (int) (Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp) / 2), learn_parm,   working2dnum, selcrit, selexam, 1, key, chosen);

						choosenum += already_chosen;
					}
					choosenum += select_next_qp_subproblem_grad(label, a, lin, c, totdoc, Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp - already_chosen), learn_parm,   working2dnum, selcrit, selexam, 0, key, chosen);

				} else {// do a step with all examples from same slack set

					jointstep = 1;
					// clear working set
					for (jj = 0; (j = working2dnum[jj]) >= 0; jj++) {
						chosen[j] = 0;
					}
					working2dnum[0] = -1;
					eq_target = alphaslack[slackset];
					for (j = 0; j < totdoc; j++) { // mask all but slackset

						if (docs[j].slackid != slackset) {
							ignore[j] = 1;
						} else {
							ignore[j] = 0;
							learn_parm.svm_cost[j] = learn_parm.svm_c;
						}
					}
					learn_parm.biased_hyperplane = 1;
					choosenum = select_next_qp_subproblem_grad(label, a, lin, c, totdoc, learn_parm.svm_maxqpsize, learn_parm,   working2dnum, selcrit, selexam, 0, key, chosen);
					learn_parm.biased_hyperplane = 0;
				}
			} else {

				// once in a while, select a somewhat random working set to
				// get unlocked of infinite loops due to numerical
				// inaccuracies in the core qp-solver
				choosenum += select_next_qp_subproblem_rand(label,  a, lin, c, totdoc, Math.min(learn_parm.svm_maxqpsize - choosenum, learn_parm.svm_newvarsinqp), learn_parm,  working2dnum, selcrit, selexam, key, chosen, iteration);
			}

			if (jointstep != 0) {
				learn_parm.biased_hyperplane = 1;
			}

			optimize_svm(docs, label,  eq_target, chosen,  model, totdoc, working2dnum, choosenum, a, lin, c, learn_parm, aicache, kernel_parm, qp, epsilon_crit_org);

			learn_parm.biased_hyperplane = 0;

			// recompute sums of alphas
			for (jj = 0; (i = working2dnum[jj]) >= 0; jj++) {
				alphaslack[docs[i].slackid] += (a[i] - a_old[i]);
			}

			// reduce alpha to fulfill constraints
			for (jj = 0; (i = working2dnum[jj]) >= 0; jj++) {
				if (alphaslack[docs[i].slackid] > learn_parm.svm_c) {
					if (a[i] < (alphaslack[docs[i].slackid] - learn_parm.svm_c)) {
						alphaslack[docs[i].slackid] -= a[i];
						a[i] = 0;
					} else {
						a[i] -= (alphaslack[docs[i].slackid] - learn_parm.svm_c);
						alphaslack[docs[i].slackid] = learn_parm.svm_c;
					}
				}
			}

			for (jj = 0; jj<a.length; jj++) {
				learn_parm.svm_cost[jj] = a[jj] + (learn_parm.svm_c - alphaslack[docs[jj].slackid]);
			}

			model.at_upper_bound = 0;
			for (jj = 0; jj <= maxslackid; jj++) {
				if (alphaslack[jj] > (learn_parm.svm_c - learn_parm.epsilon_a)) {
					model.at_upper_bound++;
				}
			}
			
			update_linear_component(docs, label,  a, a_old, working2dnum, totdoc, totwords, kernel_parm, lin, aicache, weights);
			compute_shared_slacks(docs, label, a, lin, c,  learn_parm, slack, alphaslack);

			supvecnum = calculate_svm_model(docs, label,  lin, a, a_old, c, learn_parm, working2dnum, model);

			for (jj = 0; (i = working2dnum[jj]) >= 0; jj++) {
				a_old[i] = a[i];
			}

			retrain = check_optimality_sharedslack(docs, model, label, a, lin, c, slack, alphaslack, totdoc, learn_parm, epsilon_crit_org,  last_suboptimal_at, iteration, kernel_parm, struct);
			// maxdiff?传值 or 传地址?

			// checking whether optimizer got stuck
			if (struct.maxdiff < bestmaxdiff) {
				bestmaxdiff = struct.maxdiff;
				bestmaxdiffiter = iteration;
			}

			if (iteration > (bestmaxdiffiter + learn_parm.maxiter)) {
				// long time no progress?
				terminate = 1;
				retrain = 0;
			}

			if ((retrain == 0) && (learn_parm.epsilon_crit > struct.maxdiff)) {
				learn_parm.epsilon_crit = struct.maxdiff;
			}
			if ((retrain == 0) && (learn_parm.epsilon_crit > epsilon_crit_org)) {
				learn_parm.epsilon_crit /= 2.0;
				retrain = 1;
			}
			if (learn_parm.epsilon_crit < epsilon_crit_org) {
				learn_parm.epsilon_crit = epsilon_crit_org;
			}
		}

		learn_parm.epsilon_crit = epsilon_crit_org; // restore org
		model.maxdiff = struct.maxdiff;

		return (iteration);

	}

	/** compute the value of shared slacks and the joint alphas */
	public void compute_shared_slacks(DOC[] docs, int[] label, double[] a, double[] lin, double[] c, LEARN_PARM learn_parm, double[] slack, double[] alphaslack) {
		int jj;
		double dist, target;

		for (jj = 0;jj<lin.length; jj++) { // clear slack											// variables
			slack[docs[jj].slackid] = 0.0;
		}
		for (jj = 0; jj<lin.length; jj++) {

			dist = (lin[jj]) * (double) label[jj];

			target = -(learn_parm.eps - (double) label[jj] * c[jj]);
			if (((target - dist) > slack[docs[jj].slackid]) && Math.abs(target - dist) > (double) (0.5)) {

				slack[docs[jj].slackid] = target - dist;
			}
		}
	}
	
	// compute the value of shared slacks and the joint alphas 
	public void compute_shared_slacks(DOC[] docs, int[] label, double[] a,
			double[] lin, double[] c, int[] index2dnum, LEARN_PARM learn_parm,
			double[] slack, double[] alphaslack)
	{
		int jj, i;
		double dist, target;

		for (jj = 0; (i = index2dnum[jj]) >= 0; jj++) { /* clear slack variables */
			slack[docs[i].slackid] = 0.0;
		}
		for (jj = 0; (i = index2dnum[jj]) >= 0; jj++) {

			dist = (lin[i]) * (double) label[i];

			target = -(learn_parm.eps- (double) label[i]* c[i]);

			if (((target - dist) > slack[docs[i].slackid])
					&& Math.abs(target - dist) > (double) (0.5)) {
				slack[docs[i].slackid] =target-dist;
			}
		}
	}

	/** returns the slackset with the largest internal violation */
	public int select_next_qp_slackset(DOC[] docs, int[] label, double[] a, double[] lin, double[] slack, double[] alphaslack, double[] c, LEARN_PARM learn_parm,  CheckStruct struct) {
		int  ii, maxdiffid;
		double dist, target, maxdiff, ex_c;

		maxdiff = 0;
		maxdiffid = 0;
		for (ii = 0; ii<lin.length; ii++) {
			ex_c = learn_parm.svm_c - learn_parm.epsilon_a;
			if (alphaslack[docs[ii].slackid] >= ex_c) {
				dist = (lin[ii]) * (double) label[ii] + slack[docs[ii].slackid]; // distance
				target = -(learn_parm.eps - (double) label[ii] * c[ii]); // rhs of
																		// constraint

				if ((a[ii] > learn_parm.epsilon_a) && (dist > target)) {
					if ((dist - target) > maxdiff) { // largest violation
						maxdiff = dist - target;
						maxdiffid = docs[ii].slackid;
					}
				}
			}
		}
		struct.maxsharedviol = maxdiff;
		return (maxdiffid);
	}

	/** Check KT-conditions */
	public int check_optimality_sharedslack(DOC[] docs, MODEL model, int[] label, double[] a, double[] lin, double[] c, double[] slack, double[] alphaslack, int totdoc, LEARN_PARM learn_parm, double epsilon_crit_org,  int[] last_suboptimal_at, int iteration, KERNEL_PARM kernel_parm, CheckStruct struct) {
		int  ii, retrain;
		double dist, dist_noslack, ex_c = 0, target;

		if (kernel_parm.kernel_type == ModelConstant.LINEAR) { // be optimistic
			learn_parm.epsilon_shrink = -learn_parm.epsilon_crit / 2.0;
		} else {// be conservative
			learn_parm.epsilon_shrink = learn_parm.epsilon_shrink * 0.7 + struct.maxdiff * 0.3;
		}

		retrain = 0;
		struct.maxdiff = 0;
		struct.misclassified = 0;
		for (ii = 0; ii<label.length; ii++) {

			// distance' from hyperplane
			dist_noslack = (lin[ii] - model.b) * (double) label[ii];
			dist = dist_noslack + slack[docs[ii].slackid];
			target = -(learn_parm.eps - (double) label[ii] * c[ii]);
			ex_c = learn_parm.svm_c - learn_parm.epsilon_a;
			if ((a[ii] > learn_parm.epsilon_a) && (dist > target)) {
				if ((dist - target) > struct.maxdiff) { // largest violation
					struct.maxdiff = dist - target;
				}
			}
			if ((alphaslack[docs[ii].slackid] < ex_c) && (slack[docs[ii].slackid] > 0)) {
				if ((slack[docs[ii].slackid]) > (struct.maxdiff)) { // largest
																	// violation
					struct.maxdiff = slack[docs[ii].slackid];
				}
			}

			// Count how long a variable was at lower/upper bound (and optimal).
			// Variables, which were at the bound and optimal for a long
			// time are unlikely to become support vectors. In case our
			// cache is filled up, those variables are excluded to save
			// kernel evaluations. (See chapter 'Shrinking').
			if ((a[ii] <= learn_parm.epsilon_a) && (dist < (target + learn_parm.epsilon_shrink))) {
				last_suboptimal_at[ii] = iteration; /* not likely optimal */
			} else if ((alphaslack[docs[ii].slackid] < ex_c) && (a[ii] > learn_parm.epsilon_a) && (Math.abs(dist_noslack - target) > -learn_parm.epsilon_shrink)) {
				last_suboptimal_at[ii] = iteration; /* not at lower bound */
			} else if ((alphaslack[docs[ii].slackid] >= ex_c) && (a[ii] > learn_parm.epsilon_a) && (Math.abs(target - dist) > -learn_parm.epsilon_shrink)) {
				last_suboptimal_at[ii] = iteration; /* not likely optimal */
			}
		}
		// termination criterion
		if ((retrain == 0) && ((struct.maxdiff) > learn_parm.epsilon_crit)) {
			retrain = 1;
		}
		return (retrain);
	}

	/**
	 * Approximates the radius of the ball containing length of the longest
	 * support vector. This is pretty good for text categorization, since all
	 * documents have feature vectors of length 1. It assumes that the center of
	 * the ball is at theorigin of the space. the support vectors by bounding it
	 * with the
	 */
	public double estimate_sphere(MODEL model) {
		int j;
		double xlen, maxxlen = 0;
		DOC nulldoc;
		WORD[] nullword = new WORD[1];
		KERNEL_PARM kernel_parm = model.kernel_parm;
		nullword[0] = new WORD();
		nullword[0].wnum = 0;
		nulldoc = com.createExample(-2, 0, 0, 0.0, com.createSvector(nullword, "", 1.0));

		for (j = 1; j < model.sv_num; j++) {
			xlen = Math.sqrt(com.kernel(kernel_parm, model.supvec[j], model.supvec[j]) - 2 * com.kernel(kernel_parm, model.supvec[j], nulldoc) + com.kernel(kernel_parm, nulldoc, nulldoc));
			if (xlen > maxxlen) {
				maxxlen = xlen;
			}
		}

		return (maxxlen);
	}


	public void write_alphas(String alphafile, double[] a, int[] label, int totdoc) {

		alphafile = "alpha.txt";
		FileWriter fw = null;
		PrintWriter pw = null;
		try {
			fw = new FileWriter(new File(alphafile));
			pw = new PrintWriter(fw);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		int i;

		for (i = 0; i < totdoc; i++) {
			pw.println(a[i] * (double) label[i]);
		}
		try {
			pw.close();
			fw.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

	}

	public double estimate_r_delta(DOC[] docs, int totdoc, KERNEL_PARM kernel_parm) {
		int i;
		double maxxlen, xlen;
		DOC nulldoc; // assumes that the center of the ball is at the
		WORD[] nullword = new WORD[1]; // origin of the space.

		nullword[0] = new WORD();
		nullword[0].wnum = 0;
		nulldoc = com.createExample(-2, 0, 0, 0.0, com.createSvector(nullword, "", 1.0));

		maxxlen = 0;
		for (i = 0; i < totdoc; i++) {

			xlen = Math.sqrt(com.kernel(kernel_parm, docs[i], docs[i]) - 2 * com.kernel(kernel_parm, docs[i], nulldoc) + com.kernel(kernel_parm, nulldoc, nulldoc));
			if (xlen > maxxlen) {
				maxxlen = xlen;
			}
		}

		return (maxxlen);
	}

	public void clear_index(double[] index) {
		index[0] = -1;
	}

	public static void main(String[] args) {

	}

}
