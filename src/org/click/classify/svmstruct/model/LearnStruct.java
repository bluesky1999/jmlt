package org.click.classify.svmstruct.model;

import org.click.classify.svmstruct.data.CONSTSET;
import org.click.classify.svmstruct.data.DOC;
import org.click.classify.svmstruct.data.EXAMPLE;
import org.click.classify.svmstruct.data.KERNEL_PARM;
import org.click.classify.svmstruct.data.LABEL;
import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.MATRIX;
import org.click.classify.svmstruct.data.MODEL;
import org.click.classify.svmstruct.data.ModelConstant;
import org.click.classify.svmstruct.data.MostViolStruct;
import org.click.classify.svmstruct.data.SAMPLE;
import org.click.classify.svmstruct.data.STRUCTMODEL;
import org.click.classify.svmstruct.data.STRUCT_LEARN_PARM;
import org.click.classify.svmstruct.data.SVECTOR;
import org.click.classify.svmstruct.data.WORD;

/**
 * Basic algorithm for learning structured outputs (e.g. parses, sequences,
 * multi-label classification) with a Support Vector Machine.
 * 
 * @author lq
 */

public class LearnStruct {

	public static final short SLACK_RESCALING = 1;
	public static final short MARGIN_RESCALING = 2;
	public static final short NSLACK_ALG = 0;
	public static final short NSLACK_SHRINK_ALG = 1;
	public static final short ONESLACK_PRIMAL_ALG = 2;
	public static final short ONESLACK_DUAL_ALG = 3;
	public static final short ONESLACK_DUAL_CACHE_ALG = 4;
	public Common com = null;

	private Struct ssa = null;

	public LearnStruct() {
		com = new Common();
		ssa = FactoryStruct.get_svm_struct_api();
	}

	public void svm_learn_struct(SAMPLE sample, STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm, STRUCTMODEL sm, int alg_type) {
		int i, j;

		int newconstraints = 0, totconstraints = 0, activenum = 0;
		int opti_round;
		int[] opti;
		int  use_shrinking;
		int old_totconstraints = 0;
		double epsilon, svmCnorm;
		int tolerance, new_precision = 1, dont_stop = 0;
		double lossval, factor, dist;
		double margin = 0;
		double slack;
		double[] slacks;
		double  ceps = 0;
		int sizePsi = 0;
		double[] alpha = null;
		int[] alphahist = null;
		int optcount = 0, lastoptcount = 0;
		CONSTSET cset;
		SVECTOR diff = null;
		SVECTOR fy, fybar, f;
		SVECTOR[] fycache = null;
		SVECTOR slackvec;
		WORD[] slackv = new WORD[2];
		MODEL svmModel = null;
		LABEL ybar;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;

		ssa.initStructModel(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1;

		if (alg_type == CommonStruct.NSLACK_SHRINK_ALG) {
			use_shrinking = 1;
		} else {
			use_shrinking = 0;
		}

		opti = new int[n];
		for (i = 0; i < n; i++) {
			opti[i] = 0;
		}

		opti_round = 0;
		svmCnorm = sparm.C / n;

		if (sparm.slack_norm == 1) {
			lparm.svm_c = svmCnorm;
			lparm.sharedslack = 1;
		} else if (sparm.slack_norm == 2) {
			lparm.svm_c = 999999999999999.0; // upper bound C must never be reached
			lparm.sharedslack = 0;
			if (kparm.kernel_type != ModelConstant.LINEAR) {
				System.exit(0);
			}
		} else {
			// logger.error("ERROR: Slack norm must be L1 or L2!");
		}

		epsilon = 100.0;
		tolerance = Math.min(n / 3, Math.max(n / 100, 5));
		lparm.biased_hyperplane = 0;

		cset = ssa.initStructConstraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha = new double[cset.m];
			alphahist = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha[i] = 0;
				alphahist[i] = -1; // -1 makes sure these constraints are never removed
			}
		}

		// set initial model and slack variables
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;
		Learn sl = new Learn();

		////sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi + n, lparm, kparm, svmModel, alpha);
		sl.svm_learn_optimization(cset.lhs, cset.rhs,  sizePsi + n, lparm, kparm, svmModel, alpha);
		com.addWeightVectorToLinearModel(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights;

		// ========== main loop=======
		do { // iteratively increase precision

			epsilon = Math.max(epsilon * 0.49999999999, sparm.epsilon);
			new_precision = 1;

			if (epsilon == sparm.epsilon) // for final precision, find all SV
			{
				tolerance = 0;
			}

			lparm.epsilon_crit = epsilon / 2; // svm precision must be higher than eps

			do {
				// iteration until (approx) all SV are found for current
				// precision and tolerance

				opti_round++;
				activenum = n;
				dont_stop = 0;
				old_totconstraints = totconstraints;

				do {
					// with shrinking turned on, go through examples that keep
					// producing new constraints

					ceps = 0;


					for (i = 0; i < n; i++) {// example loop

						if ((use_shrinking == 0) || (opti[i] != opti_round)) {

							// if the example is not shrunk away, then see if it
							// is necessary to add a new constraint
							if (sparm.loss_type == SLACK_RESCALING) {
								ybar = ssa.findMostViolatedConstraintSlackrescaling(ex[i].x, ex[i].y, sm, sparm);
							} else {
								ybar = ssa.findMostViolatedConstraintMarginrescaling(ex[i].x, ex[i].y, sm, sparm);
							}

							if (ssa.emptyLabel(ybar)) {
								if (opti[i] != opti_round) {
									activenum--;
									opti[i] = opti_round;
								}
								continue;
							}

							// get psi(y)-psi(ybar)
							if (fycache != null) {
								fy = com.copySvector(fycache[i]);
							} else {
								fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);
							}

							fybar = ssa.psi(ex[i].x, ybar, sm, sparm);

							// scale feature vector and margin by loss
							lossval = ssa.loss(ex[i].y, ybar, sparm);
							if (sparm.slack_norm == 2)
								lossval = Math.sqrt(lossval);
							if (sparm.loss_type == SLACK_RESCALING)
								factor = lossval;
							else
								// do not rescale vector for
								factor = 1.0; // margin rescaling loss type
							for (f = fy; f != null; f = f.next) {
								f.factor *= factor;
							}

							for (f = fybar; f != null; f = f.next)
								f.factor *= -factor;
							margin = lossval;

							// create constraint for current ybar
							com.appendSvectorList(fy, fybar);
							doc = com.createExample(cset.m, 0, i + 1, 1, fy);

							// compute slack for this example
							slack = 0;
							for (j = 0; j < cset.m; j++)
								if (cset.lhs[j].slackid == i + 1) {
									if (sparm.slack_norm == 2)
										slack = Math.max(slack, cset.rhs[j] - (com.classifyExample(svmModel, cset.lhs[j]) - sm.w[sizePsi + i] / (Math.sqrt(2 * svmCnorm))));
									else
										slack = Math.max(slack, cset.rhs[j] - com.classifyExample(svmModel, cset.lhs[j]));
								}

							// if `error' add constraint and recompute
							dist = com.classifyExample(svmModel, doc);
							ceps = Math.max(ceps, margin - dist - slack);

							if ((dist + slack) < (margin - epsilon)) {

								// resize constraint matrix and add new constraint
								cset.m++;
								com.realloc(cset);

								if (kparm.kernel_type == ModelConstant.LINEAR) {
									diff = com.addListSs(fy);
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = com.createExample(cset.m - 1, 0, i + 1, 1, com.copySvector(diff));
									else if (sparm.slack_norm == 2) {

										// add squared slack variable to feature vector
										slackv[0].wnum = sizePsi + i;
										slackv[0].weight = 1 / (Math.sqrt(2 * svmCnorm));
										slackv[1].wnum = 0; // terminator
										slackvec = com.createSvector(slackv, null, 1.0);
										cset.lhs[cset.m - 1] = com.createExample(cset.m - 1, 0, i + 1, 1, com.addSs(diff, slackvec));
									}
								} else {// kernel is used
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = com.createExample(cset.m - 1, 0, i + 1, 1, com.copySvector(fy));
									else if (sparm.slack_norm == 2)
										System.exit(1);
								}
								com.reallocRhs(cset);
								cset.rhs[cset.m - 1] = margin;

								alpha = com.reallocAlpha(alpha, cset.m);
								alpha[cset.m - 1] = 0;
								alphahist = com.reallocAlphaList(alphahist, cset.m);
								alphahist[cset.m - 1] = optcount;
								newconstraints++;
								totconstraints++;
							} else {
								if (opti[i] != opti_round) {
									activenum--;
									opti[i] = opti_round;
								}
							}
						}// if use shrinking

						// get new QP solution
						if ((newconstraints >= sparm.newconstretrain) || ((newconstraints > 0) && (i == n - 1)) || ((new_precision != 0) && (i == n - 1))) {

							svmModel = new MODEL();

							// Run the QP solver on cset.
							///sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi + n, lparm, kparm, svmModel, alpha);
							sl.svm_learn_optimization(cset.lhs, cset.rhs, sizePsi + n, lparm, kparm, svmModel, alpha);

							// Always add weight vector, in case part of the
							// kernel is linear. If not, ignore the weight
							// vector since its content is bogus.
							com.addWeightVectorToLinearModel(svmModel);
							sm.svm_model = svmModel;
							sm.w = new double[svmModel.lin_weights.length];
							for (int k = 0; k < svmModel.lin_weights.length; k++) {
								sm.w[k] = svmModel.lin_weights[k];
							}

							optcount++;

							// keep track of when each constraint was last
							// active. constraints marked with -1 are not
							// updated
							for (j = 0; j < cset.m; j++) {
								if ((alphahist[j] > -1) && (alpha[j] != 0)) {
									alphahist[j] = optcount;
								}
							}

							if ((new_precision != 0) && (epsilon <= sparm.epsilon))
								dont_stop = 1; // make sure we take one final pass

							new_precision = 0;
							newconstraints = 0;
						}

					}// exmample loop

					System.out.println("(NumConst=" + cset.m + ", SV=" + (svmModel.sv_num - 1) + ", CEps=" + ceps + ", QPEps=" + svmModel.maxdiff + ")\n");

					remove_inactive_constraints(cset, alpha, optcount, alphahist, Math.max(50, optcount - lastoptcount));

					lastoptcount = optcount;
				} while ((use_shrinking != 0) && (activenum > 0));
			} while (((totconstraints - old_totconstraints) > tolerance) || (dont_stop != 0));

		} while ((epsilon > sparm.epsilon)); // main_loop

		if (CommonStruct.struct_verbosity >= 1) {
			// compute sum of slacks
			// WARNING: If positivity constraints are used, then the maximum
			// slack id is larger than what is allocated below
			slacks = new double[n + 1];
			for (i = 0; i <= n; i++) {
				slacks[i] = 0;
			}

			if (sparm.slack_norm == 1) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(slacks[cset.lhs[j].slackid], cset.rhs[j] - com.classifyExample(svmModel, cset.lhs[j]));
			} else if (sparm.slack_norm == 2) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(slacks[cset.lhs[j].slackid], cset.rhs[j] - (com.classifyExample(svmModel, cset.lhs[j]) - sm.w[sizePsi + cset.lhs[j].slackid - 1] / (Math.sqrt(2 * svmCnorm))));
			}
		}

		if (svmModel != null) {
			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; // short cut to weight vector
		}

		ssa.printStructLearningStats(sample, sm, cset, alpha, sparm);

	}

	public void svm_learn_struct_joint(SAMPLE sample, STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm, STRUCTMODEL sm, int alg_type) {
		int i, j;
		int numIt = 0;
		int kernel_type_org;
		double epsilon;
		double lhsXw;
		MostViolStruct violStruct = new MostViolStruct();
		double rhs = 0;
		double slack, ceps;
		double  alphasum;
		int sizePsi;
		double[] alpha = null;
		int[] alphahist = null;
		int optcount = 0;

		CONSTSET cset;
		double[] lhs_n = null;
		SVECTOR fy;
		SVECTOR[] fycache = null;
		SVECTOR lhs;
		MODEL svmModel = null;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;

		Learn sl = new Learn();

		ssa.initStructModel(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1; // sm must contain size of psi on return

		if (sparm.slack_norm == 1) {
			lparm.svm_c = sparm.C; // set upper bound C
			lparm.sharedslack = 1;
		}

		lparm.biased_hyperplane = 0; // set threshold to zero
		epsilon = 100.0; // start with low precision and increase later

		cset = ssa.initStructConstraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha = new double[cset.m];
			alphahist = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha[i] = 0;
				alphahist[i] = -1;
			}
		}

		kparm.gram_matrix = null;
		if ((alg_type == CommonStruct.ONESLACK_DUAL_ALG) || (alg_type == CommonStruct.ONESLACK_DUAL_CACHE_ALG))
			kparm.gram_matrix = init_kernel_matrix(cset, kparm);

		// set initial model and slack variables
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;

		///sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi, lparm, kparm, svmModel, alpha);
		sl.svm_learn_optimization(cset.lhs, cset.rhs, sizePsi, lparm, kparm, svmModel, alpha);
		
		// //logger.info("sl totwords:"+svmModel.totwords);
		com.addWeightVectorToLinearModel(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights; // short cut to weight vector

		// create a cache of the feature vectors for the correct labels
		fycache = new SVECTOR[n];
		for (i = 0; i < n; i++) {
			fy = null;
			fycache[i] = fy;
		}

		if (kparm.kernel_type == ModelConstant.LINEAR) {
			// //logger.info("kernel type is LINEAR \n");
			lhs_n = com.createNvector(sm.sizePsi);
		}

		// === main loop=======
		do { // iteratively find and add constraints to working set

			// compute current slack
			alphasum = 0;
			for (j = 0; (j < cset.m); j++) {
				alphasum += alpha[j];
			}
			for (j = 0, slack = -1; (j < cset.m) && (slack == -1); j++) {
				if (alpha[j] > alphasum / cset.m) {
					slack = Math.max(0, cset.rhs[j] - com.classifyExample(svmModel, cset.lhs[j]));
				}
			}
			slack = Math.max(0, slack);
			// find a violated joint constraint
			lhs = null;
			rhs = 0;

			if (kparm.kernel_type == ModelConstant.LINEAR)
				com.clearNvector(lhs_n, sm.sizePsi);


			for (i = 0; i < n; i++) {

				// compute most violating fydelta=fy-fybar and rhs for
				// example i
				find_most_violated_constraint(ex[i], fycache[i], n, sm, sparm, violStruct);
				// add current fy-fybar to lhs of constraint
				if (kparm.kernel_type == ModelConstant.LINEAR) {
					com.addListNNS(lhs_n, violStruct.fydelta, 1.0);
				} else {
					com.appendSvectorList(violStruct.fydelta, lhs);
					lhs = violStruct.fydelta;
				}
				rhs += violStruct.rhs;// add loss to rhs
			} // end of example loop

			// create sparse vector from dense sum
			if (kparm.kernel_type == ModelConstant.LINEAR) {
				lhs = com.createSvectorNR(lhs_n, sm.sizePsi, null, 1.0, CommonStruct.COMPACT_ROUNDING_THRESH);
			}
			doc = com.createExample(cset.m, 0, 1, 1, lhs);
			lhsXw = com.classifyExample(svmModel, doc);

			ceps = Math.max(0, rhs - lhsXw - slack);
			if ((ceps > sparm.epsilon)) {
				// resize constraint matrix and add new constraint
				int ti = 0;
				ti = cset.m + 1;
				cset.lhs = com.reallocDOCS(cset.lhs, ti);
				cset.lhs[cset.m] = com.createExample(cset.m, 0, 1, 1, lhs);
				cset.rhs = com.reallocDoubleArr(cset.rhs, cset.m + 1);
				cset.rhs[cset.m] = rhs;
				alpha = com.reallocDoubleArr(alpha, cset.m + 1);
				alpha[cset.m] = 0;
				alphahist = com.reallocIntArr(alphahist, cset.m + 1);
				alphahist[cset.m] = optcount;
				cset.m++;
				if ((alg_type == ONESLACK_DUAL_ALG) || (alg_type == ONESLACK_DUAL_CACHE_ALG)) {
					kparm.gram_matrix = update_kernel_matrix(kparm.gram_matrix, cset.m - 1, cset, kparm);
				}

				// set svm precision so that higher than eps of most violated constr best eps so far
				epsilon = Math.min(epsilon, ceps); 
				lparm.epsilon_crit = epsilon / 2;

				svmModel = new MODEL();
				// Run the QP solver on cset.
				kernel_type_org = kparm.kernel_type;
				if ((alg_type == CommonStruct.ONESLACK_DUAL_ALG) || (alg_type == CommonStruct.ONESLACK_DUAL_CACHE_ALG))
				// use kernel stored in kparm
				kparm.kernel_type = ModelConstant.GRAM; 

				///sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi, lparm, kparm, svmModel, alpha);
				sl.svm_learn_optimization(cset.lhs, cset.rhs, sizePsi, lparm, kparm, svmModel, alpha);
				kparm.kernel_type = (short) kernel_type_org;

				svmModel.kernel_parm.kernel_type = (short) kernel_type_org;

				// Always add weight vector, in case part of the kernel is
				// linear. If not, ignore the weight vector since its content is
				// bogus.
				com.addWeightVectorToLinearModel(svmModel);

				sm.svm_model = svmModel;

				// sm.svm_model = svmModel; sm.w = svmModel.lin_weights;
				sm.w = new double[svmModel.lin_weights.length];
				for (int iw = 0; iw < svmModel.lin_weights.length; iw++) {
					sm.w[iw] = svmModel.lin_weights[iw];
				}

				optcount++;

				// keep track of when each constraint was last active.
				// constraints marked with -1 are not updated
				for (j = 0; j < cset.m; j++) {
					if ((alphahist[j] > -1) && (alpha[j] != 0)) {
						alphahist[j] = optcount;
					}
				}

				// 在这里要将某些限制去掉
				remove_inactive_constraints(cset, alpha, optcount, alphahist, 50);

			}

			System.out.println("Iter " + numIt + " (NumConst=" + cset.m + ", SV=" + (svmModel.sv_num - 1) + ", CEps=" + ceps + ", QPEps=" + svmModel.maxdiff + ")");
			numIt++;
		} while (ceps > sparm.epsilon);

		if (CommonStruct.struct_verbosity >= 1) {
			slack = 0;
			for (j = 0; j < cset.m; j++)
				slack = Math.max(slack, cset.rhs[j] - com.classifyExample(svmModel, cset.lhs[j]));
			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha[i] * cset.rhs[i];

		}

		if (svmModel != null) {

			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; /* short cut to weight vector */
		}

		ssa.printStructLearningStats(sample, sm, cset, alpha, sparm);

	}

	/**
	 * removes the constraints from cset (and alpha) for which alphahist
	 * indicates that they have not been active for at least mininactive
	 * iterations
	 */
	public void remove_inactive_constraints(CONSTSET cset, double[] alpha, int currentiter, int[] alphahist, int mininactive) {
		int i, m;

		m = 0;
		for (i = 0; i < cset.m; i++) {
			if ((alphahist[i] < 0) || ((currentiter - alphahist[i]) < mininactive)) {

				// keep constraints that are marked as -1 or which have recently
				// been active
				cset.lhs[m] = cset.lhs[i];
				cset.lhs[m].docnum = m;
				cset.rhs[m] = cset.rhs[i];
				alpha[m] = alpha[i];
				alphahist[m] = alphahist[i];
				m++;
			} else {
			}
		}

		if (cset.m != m) {
			cset.m = m;
			com.realSmalllocLhs(cset);
			com.realSmalllocRhs(cset);

			alpha = com.reallocDoubleArr(alpha, cset.m);
			alphahist = com.reallocIntArr(alphahist, cset.m);

		}
	}

	/**
	 * assigns a kernelid to each constraint in cset and creates the
	 * corresponding kernel matrix.
	 */
	public MATRIX init_kernel_matrix(CONSTSET cset, KERNEL_PARM kparm) {
		int i, j;
		double kval;
		MATRIX matrix;

		// assign kernel id to each new constraint
		for (i = 0; i < cset.m; i++)
			cset.lhs[i].kernelid = i;

		// allocate kernel matrix as necessary
		matrix = com.createMatrix(i + 50, i + 50);

		for (j = 0; j < cset.m; j++) {
			for (i = j; i < cset.m; i++) {
				kval = com.kernel(kparm, cset.lhs[j], cset.lhs[i]);
				matrix.element[j][i] = kval;
				matrix.element[i][j] = kval;
			}
		}
		return (matrix);
	}

	/**
	 * returns fydelta=fy-fybar and rhs scalar value that correspond to the most
	 * violated constraint for example ex
	 */
	public void find_most_violated_constraint(EXAMPLE ex, SVECTOR fycached, int n, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm, MostViolStruct struct) {
		LABEL ybar;
		SVECTOR fybar, fy;
		double factor, lossval;

		struct.argmaxCount++;
		if (sparm.loss_type == SLACK_RESCALING) {
			ybar = ssa.findMostViolatedConstraintSlackrescaling(ex.x, ex.y, sm, sparm);
		} else {
			ybar = ssa.findMostViolatedConstraintMarginrescaling(ex.x, ex.y, sm, sparm);
		}

		// get psi(x,y) and psi(x,ybar)
		if (fycached != null)
			fy = com.copySvector(fycached);
		else
			fy = ssa.psi(ex.x, ex.y, sm, sparm);
		fybar = ssa.psi(ex.x, ybar, sm, sparm);

		lossval = ssa.loss(ex.y, ybar, sparm);

		// scale feature vector and margin by loss
		if (sparm.loss_type == SLACK_RESCALING)
			factor = lossval / n;
		else
			factor = 1.0 / n; // do not rescale vector formargin rescaling loss
								// type
		com.multSvectorList(fy, factor);
		com.multSvectorList(fybar, -factor);
		com.appendSvectorList(fybar, fy); // compute fy-fybar

		struct.fydelta = fybar;
		struct.rhs = lossval / (double) n;
	}

	/**
	 * assigns new kernelid to constraint in position newpos and fills the
	 * corresponding part of the kernel matrix
	 */
	public MATRIX update_kernel_matrix(MATRIX matrix, int newpos, CONSTSET cset, KERNEL_PARM kparm) {
		int i, maxkernelid = 0, newid;
		double kval;
		double[] used;

		// find free kernelid to assign to new constraint
		for (i = 0; i < cset.m; i++) {
			if (i != newpos) {
				maxkernelid = Math.max(maxkernelid, cset.lhs[i].kernelid);
			}
		}
		used = com.createNvector(maxkernelid + 2);
		com.clearNvector(used, maxkernelid + 2);
		for (i = 0; i < cset.m; i++)
			if (i != newpos)
				used[cset.lhs[i].kernelid] = 1;
		for (newid = 0; used[newid] != 0; newid++)
			;
		cset.lhs[newpos].kernelid = newid;

		// extend kernel matrix if necessary
		maxkernelid = Math.max(maxkernelid, newid);
		if ((matrix == null) || (maxkernelid >= matrix.m))
			matrix = com.reallocMatrix(matrix, maxkernelid + 50, maxkernelid + 50);

		for (i = 0; i < cset.m; i++) {
			kval = com.kernel(kparm, cset.lhs[newpos], cset.lhs[i]);
			matrix.element[newid][cset.lhs[i].kernelid] = kval;
			matrix.element[cset.lhs[i].kernelid][newid] = kval;
		}
		return (matrix);
	}

}
