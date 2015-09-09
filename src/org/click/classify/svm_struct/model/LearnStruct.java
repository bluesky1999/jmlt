package org.click.classify.svm_struct.model;

import org.apache.log4j.Logger;
import org.click.classify.svm_struct.data.CONSTSET;
import org.click.classify.svm_struct.data.DOC;
import org.click.classify.svm_struct.data.EXAMPLE;
import org.click.classify.svm_struct.data.KERNEL_CACHE;
import org.click.classify.svm_struct.data.KERNEL_PARM;
import org.click.classify.svm_struct.data.LABEL;
import org.click.classify.svm_struct.data.LEARN_PARM;
import org.click.classify.svm_struct.data.MATRIX;
import org.click.classify.svm_struct.data.MODEL;
import org.click.classify.svm_struct.data.ModelConstant;
import org.click.classify.svm_struct.data.SAMPLE;
import org.click.classify.svm_struct.data.STRUCTMODEL;
import org.click.classify.svm_struct.data.STRUCT_LEARN_PARM;
import org.click.classify.svm_struct.data.SVECTOR;
import org.click.classify.svm_struct.data.WORD;
import org.click.classify.svm_struct.data.WU;

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

	public double rhs_g = 0;
	public SVECTOR lhs_g = null;
	public double rt_cachesum_g = 0;
	public double rhs_i_g;
	public double rt_viol_g;
	public double rt_psi_g;
	public double[] lhs_n;
	public int argmax_count_g;
	//public CCACHE ccache_g;
	public SVECTOR fydelta_g = null;

	public double[] alpha_g = null;
	public int[] alphahist_g = null;
	public int[] remove_g = null;

	private Struct ssa=null;
	
	private static Logger logger = Logger.getLogger(LearnStruct.class);
	
	public LearnStruct()
	{
		ssa=FactoryStruct.get_svm_struct_api();
	}
	
	public void svm_learn_struct(SAMPLE sample, STRUCT_LEARN_PARM sparm,
			LEARN_PARM lparm, KERNEL_PARM kparm, STRUCTMODEL sm, int alg_type) {
		int i, j;
		int numIt = 0;
		alpha_g = null;
		alphahist_g = null;
		int argmax_count = 0;
		int newconstraints = 0, totconstraints = 0, activenum = 0;
		int opti_round;
		int[] opti;
		int fullround, use_shrinking;
		int old_totconstraints = 0;
		double epsilon, svmCnorm;
		int tolerance, new_precision = 1, dont_stop = 0;
		double lossval, factor, dist;
		double margin = 0;
		double slack;
		double[] slacks;
		double slacksum, ceps = 0;
		double dualitygap, modellength, alphasum;
		int sizePsi = 0;
		// double[] alpha = null;
		// int[] alphahist = null;
		int optcount = 0, lastoptcount = 0;
		CONSTSET cset;
		SVECTOR diff = null;
		SVECTOR fy, fybar, f;
		SVECTOR[] fycache = null;
		SVECTOR slackvec;
		WORD[] slackv = new WORD[2];
		MODEL svmModel = null;
		KERNEL_CACHE kcache = null;
		LABEL ybar;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;
		double rt_total = 0, rt_opt = 0, rt_init = 0, rt_psi = 0, rt_viol = 0;
		double rt1, rt2;
		rt1 = Common.get_runtime();

		ssa.initStructModel(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1;
		logger.info("the sizePsi2 is " + sizePsi);
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
			if (kparm.kernel_type !=  ModelConstant.LINEAR) {
				logger.error("ERROR: Kernels are not implemented for L2 slack norm!");
				System.exit(0);
			}
		} else {
			logger.error("ERROR: Slack norm must be L1 or L2!");
		}

		epsilon = 100.0;
		tolerance = Math.min(n / 3, Math.max(n / 100, 5));
		lparm.biased_hyperplane = 0;

		cset = ssa.initStructConstraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha_g = new double[cset.m];
			alphahist_g = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha_g[i] = 0;
				alphahist_g[i] = -1; // -1 makes sure these constraints are never removed
			}
		}

		//set initial model and slack variables
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;
		if (kparm.kernel_type !=  ModelConstant.LINEAR) {
			kcache = Learn.kernel_cache_init(Math.max(cset.m, 1),
					lparm.kernel_cache_size);
		}
		Learn sl = new Learn();
		logger.info("sizePsi:" + sizePsi + "   n:" + n);
		sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi + n,
				lparm, kparm, kcache, svmModel, alpha_g);
		Common.add_weight_vector_to_linear_model(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights;

		if ( ModelConstant.USE_FYCACHE != 0) {
			fycache = new SVECTOR[n];
			for (i = 0; i < n; i++) {
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);// temp
																		// point
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					diff = Common.add_list_ss(fy);
					fy = diff;
				}

			}

		}

		rt_init += Math.max(Common.get_runtime() - rt1, 0);
		rt_total += Math.max(Common.get_runtime() - rt1, 0);


		//========== main loop=======
		do { // iteratively increase precision 

			epsilon = Math.max(epsilon * 0.49999999999, sparm.epsilon);
			new_precision = 1;

			if (epsilon == sparm.epsilon) // for final precision, find all SV
			{
				tolerance = 0;
			}

			lparm.epsilon_crit = epsilon / 2; // svm precision must be higher than eps
											
			if (CommonStruct.struct_verbosity >= 1) {
				logger.info("Setting current working precision to " + epsilon);
			}

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

					if (CommonStruct.struct_verbosity >= 1) {
						logger.info("Iter " + (++numIt) + " (" + activenum
								+ " active):");
					}

					ceps = 0;
					if (activenum == n) {
						fullround = 1;
					} else {
						fullround = 0;
					}

					for (i = 0; i < n; i++) {
						// example loop 
						rt1 = Common.get_runtime();

						if ((use_shrinking == 0) || (opti[i] != opti_round)) {

							// if the example is not shrunk away, then see if it
							// is necessary to add a new constraint					
							rt2 = Common.get_runtime();
							argmax_count++;
							if (sparm.loss_type == SLACK_RESCALING) {
								ybar = ssa.findMostViolatedConstraintSlackrescaling(
												ex[i].x, ex[i].y, sm, sparm);
							} else {
								ybar = ssa.findMostViolatedConstraintMarginrescaling(
												ex[i].x, ex[i].y, sm, sparm);
							}
							rt_viol += Math.max(Common.get_runtime() - rt2,
									0);

							if (ssa.emptyLabel(ybar)) {
								if (opti[i] != opti_round) {
									activenum--;
									opti[i] = opti_round;
								}
								if (CommonStruct.struct_verbosity >= 2)
									logger.info("no-incorrect-found(" + i
											+ ") ");
								continue;
							}

							// get psi(y)-psi(ybar) 
							rt2 = Common.get_runtime();
							if (fycache != null) {
								fy = Common.copy_svector(fycache[i]);
							} else {
								fy = ssa.psi(ex[i].x, ex[i].y, sm,
										sparm);
							}

							fybar = ssa.psi(ex[i].x, ybar, sm, sparm);
;
							rt_psi += Math.max(Common.get_runtime() - rt2,
									0);

							// scale feature vector and margin by loss
							lossval = ssa.loss(ex[i].y, ybar, sparm);
							if (sparm.slack_norm == 2)
								lossval = Math.sqrt(lossval);
							if (sparm.loss_type == SLACK_RESCALING)
								factor = lossval;
							else
								//do not rescale vector for 
								factor = 1.0; // margin rescaling loss type 
							for (f = fy; f != null; f = f.next) {
								f.factor *= factor;
							}

							for (f = fybar; f != null; f = f.next)
								f.factor *= -factor;
							margin = lossval;

							// create constraint for current ybar
							Common.append_svector_list(fy, fybar);
							doc = Common.create_example(cset.m, 0, i + 1,
									1, fy);

							// compute slack for this example
							slack = 0;
							for (j = 0; j < cset.m; j++)
								if (cset.lhs[j].slackid == i + 1) {
									if (sparm.slack_norm == 2)
										slack = Math.max(slack,cset.rhs[j]- (Common.classify_example(svmModel,cset.lhs[j]) - sm.w[sizePsi+ i]/ (Math.sqrt(2 * svmCnorm))));
									else
										slack = Math.max(slack,cset.rhs[j]- Common.classify_example(svmModel,cset.lhs[j]));
								}

							//if `error' add constraint and recompute 
							dist = Common.classify_example(svmModel, doc);
							ceps = Math.max(ceps, margin - dist - slack);
							if (slack > (margin - dist + 0.0001)) {
								logger.debug("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
								logger.debug("         set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
								logger.debug("Ex " + i + ": slack=" + slack
										+ ", newslack=" + (margin - dist)
										+ "\n");
							}
							if ((dist + slack) < (margin - epsilon)) {
								if (CommonStruct.struct_verbosity >= 2) {
									logger.info("(" + i + ",eps="
											+ (margin - dist - slack) + ") ");
								}
								if (CommonStruct.struct_verbosity == 1) {
									System.out.print(".");
								}


								// resize constraint matrix and add new constraint
								cset.m++;
								Common.realloc(cset);

								if (kparm.kernel_type ==  ModelConstant.LINEAR) {
									diff = Common.add_list_ss(fy);
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = Common.create_example(cset.m - 1,0,i + 1,1,Common.copy_svector(diff));
									else if (sparm.slack_norm == 2) {

										// add squared slack variable to feature vector
										slackv[0].wnum = sizePsi + i;
										slackv[0].weight = 1 / (Math
												.sqrt(2 * svmCnorm));
										slackv[1].wnum = 0; //terminator 
										slackvec = Common.create_svector(slackv, null, 1.0);
										cset.lhs[cset.m - 1] = Common.create_example(cset.m - 1, 0,i + 1, 1,Common.add_ss(diff,slackvec));
									}
								} else {//kernel is used 
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = Common
												.create_example(cset.m - 1,0,
														i + 1,
														1,
														Common
																.copy_svector(fy));
									else if (sparm.slack_norm == 2)
										System.exit(1);
								}
								Common.realloc_rhs(cset);
								cset.rhs[cset.m - 1] = margin;

								alpha_g = Common.realloc_alpha(alpha_g,
										cset.m);
								alpha_g[cset.m - 1] = 0;
								alphahist_g = Common.realloc_alpha_list(alphahist_g, cset.m);
								alphahist_g[cset.m - 1] = optcount;
								newconstraints++;
								totconstraints++;
							} else {
								// logger.info("+");
								if (opti[i] != opti_round) {
									activenum--;
									opti[i] = opti_round;
								}
							}
						}// if use shrinking

						//get new QP solution
						if ((newconstraints >= sparm.newconstretrain)
								|| ((newconstraints > 0) && (i == n - 1))
								|| ((new_precision != 0) && (i == n - 1))) {
							if (CommonStruct.struct_verbosity >= 1) {
								// logger.info("*");
							}
							rt2 = Common.get_runtime();

							svmModel = new MODEL();

							// Always get a new kernel cache. It is not possible
							// to use the same cache for two different training
							// runs

							if (kparm.kernel_type !=  ModelConstant.LINEAR)
								kcache = sl.kernel_cache_init(
										Math.max(cset.m, 1),
										lparm.kernel_cache_size);
							// Run the QP solver on cset.
							sl.svm_learn_optimization(cset.lhs, cset.rhs,
									cset.m, sizePsi + n, lparm, kparm, kcache,
									svmModel, alpha_g);
						
							// Always add weight vector, in case part of the
							//kernel is linear. If not, ignore the weight
							// vector since its content is bogus.
							Common.add_weight_vector_to_linear_model(svmModel);
							sm.svm_model = svmModel.copyMODEL();
							sm.w = new double[svmModel.lin_weights.length];
							for (int k = 0; k < svmModel.lin_weights.length; k++) {
								sm.w[k] = svmModel.lin_weights[k];
							}
							// sm.w = svmModel.lin_weights;
							optcount++;

							// keep track of when each constraint was last
							 // active. constraints marked with -1 are not
							 // updated						 
							for (j = 0; j < cset.m; j++) {
								if ((alphahist_g[j] > -1) && (alpha_g[j] != 0)) {
									alphahist_g[j] = optcount;
								}
								// logger.info("j="+j+"");
							}
							rt_opt += Math.max(Common.get_runtime() - rt2,
									0);

							if ((new_precision != 0)
									&& (epsilon <= sparm.epsilon))
								dont_stop = 1; // make sure we take one final pass
		
							new_precision = 0;
							newconstraints = 0;
						}

						rt_total += Math.max(Common.get_runtime() - rt1, 0);
					}// exmample loop

					rt1 = Common.get_runtime();

					//if (svm_struct_common.struct_verbosity >= 1)
						logger.info("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");
						System.out.println("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");

					if (CommonStruct.struct_verbosity >= 2)
						logger.info("Reducing working set...");

					remove_inactive_constraints(cset, optcount,
							Math.max(50, optcount - lastoptcount));

					lastoptcount = optcount;
					if (CommonStruct.struct_verbosity >= 2)
						logger.info("done. (NumConst=" + cset.m + ")\n");

					rt_total += Math.max(Common.get_runtime() - rt1, 0);

				} while ((use_shrinking != 0) && (activenum > 0));
			} while (((totconstraints - old_totconstraints) > tolerance)
					|| (dont_stop != 0));

		} while ((epsilon > sparm.epsilon)
				|| ssa.finalizeIteration(ceps, 0, sample, sm, cset,
						alpha_g, sparm)); // main_loop

		if (CommonStruct.struct_verbosity >= 1) {
			//compute sum of slacks 
			 //WARNING: If positivity constraints are used, then the maximum
			 // slack id is larger than what is allocated below
			slacks = new double[n + 1];
			for (i = 0; i <= n; i++) {
				slacks[i] = 0;
			}

			if (sparm.slack_norm == 1) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(
							slacks[cset.lhs[j].slackid],
							cset.rhs[j]
									- Common.classify_example(svmModel,
											cset.lhs[j]));
			} else if (sparm.slack_norm == 2) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(
							slacks[cset.lhs[j].slackid],
							cset.rhs[j]
									- (Common.classify_example(svmModel,
											cset.lhs[j]) - sm.w[sizePsi
											+ cset.lhs[j].slackid - 1]
											/ (Math.sqrt(2 * svmCnorm))));
			}
			slacksum = 0;
			for (i = 1; i <= n; i++)
				slacksum += slacks[i];

			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha_g[i] * cset.rhs[i];
			modellength = Common.model_length_s(svmModel);
			dualitygap = (0.5 * modellength * modellength + svmCnorm
					* (slacksum + n * ceps))
					- (alphasum - 0.5 * modellength * modellength);

			logger.info("Final epsilon on KKT-Conditions: "
					+ Math.max(svmModel.maxdiff, epsilon) + "\n");
			logger.info("Upper bound on duality gap: " + dualitygap + "\n");
			logger.info("Dual objective value: dval="
					+ (alphasum - 0.5 * modellength * modellength) + "\n");
			logger.info("Total number of constraints in final working set: "
					+ (int) cset.m + " (of " + (int) totconstraints + ")\n");
			logger.info("Number of iterations:" + numIt + "\n");
			logger.info("Number of calls to 'find_most_violated_constraint': "
					+ argmax_count + "\n");
			if (sparm.slack_norm == 1) {
				logger.info("Number of SV: " + (svmModel.sv_num - 1) + " \n");
				logger.info("Number of non-zero slack variables: "
						+ svmModel.at_upper_bound + " (out of " + n + ")\n");
				logger.info("Norm of weight vector: |w|=" + modellength + "\n");
			} else if (sparm.slack_norm == 2) {
				logger.info("Number of SV: " + (svmModel.sv_num - 1)
						+ " (including " + svmModel.at_upper_bound
						+ " at upper bound)\n");
				logger.info("Norm of weight vector (including L2-loss): |w|="
						+ modellength + "\n");
			}

			logger.info("Norm. sum of slack variables (on working set): sum(xi_i)/n="
					+ slacksum / n + "\n");
			logger.info("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||="
					+ sl.length_of_longest_document_vector(cset.lhs, cset.m,
							kparm) + "\n");
			logger.info("Runtime in cpu-seconds: " + rt_total / 100.0 + " ("
					+ (100.0 * rt_opt) / rt_total + " for QP, "
					+ (100.0 * rt_viol) / rt_total + " for Argmax, "
					+ (100.0 * rt_psi) / rt_total + " for Psi, "
					+ (100.0 * rt_init) / rt_total + " for init)\n");
		}

		if (CommonStruct.struct_verbosity >= 4)
			logger.info(CommonStruct.printW(sm.w, sizePsi, n, lparm.svm_c));

		if (svmModel != null) {
			// sm.svm_model = svm_common.copy_model(svmModel);
			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; // short cut to weight vector
			String wstr = "";
			for (int wi = 0; wi < sm.w.length; wi++) {
				wstr += (wi + ":" + sm.w[wi] + " ");
			}
			logger.info("wstr:" + wstr);
		}

		ssa.printStructLearningStats(sample, sm, cset, alpha_g,
				sparm);

	}

	public void svm_learn_struct_joint(SAMPLE sample, STRUCT_LEARN_PARM sparm,
			LEARN_PARM lparm, KERNEL_PARM kparm, STRUCTMODEL sm, int alg_type) {
		int i, j;
		int numIt = 0;
		// int argmax_count=0;
		argmax_count_g = 0;
		int totconstraints = 0;
		int kernel_type_org;
		double epsilon, epsilon_cached;
		double lhsXw;
		// double rhs_i;
		rhs_i_g = 0;
		rhs_g = 0;
		// double rhs=0;
		double slack, ceps;
		double dualitygap, modellength, alphasum;
		int sizePsi;
		// double[] alpha = null;
		// int[] alphahist = null;
		int optcount = 0;

		CONSTSET cset;
		SVECTOR diff = null;
		// double[] lhs_n=null;
		SVECTOR fy;
		// SVECTOR fydelta=null;
		SVECTOR[] fycache = null;
		// SVECTOR lhs;
		MODEL svmModel = null;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;
		double rt_total = 0;
		double rt_opt = 0;
		double rt_init = 0;
		// double rt_psi=0;
		// double rt_viol=0;
		rt_psi_g = 0;
		rt_viol_g = 0;
		double rt_kernel = 0;

		double rt_cacheupdate = 0, rt_cacheconst = 0, rt_cacheadd = 0;
		// double rt_cachesum=0;
		rt_cachesum_g = 0;
		double rt1 = 0, rt2 = 0;
		int progress;

		// CCACHE ccache=null;
		//ccache_g = null;
		int cached_constraint;
		double viol=0, viol_est, epsilon_est = 0;
		int uptr = 0;
		int[] randmapping = null;
		int batch_size = n;

		Learn sl = new Learn();
		rt1 = Common.get_runtime();
		if (sparm.batch_size < 100)
			batch_size = (int) ((sparm.batch_size * n) / 100.0);
		ssa.initStructModel(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1; /* sm must contain size of psi on return */
		// logger.info("the sizePsi is "+sizePsi+" \n");

		if (sparm.slack_norm == 1) {
			lparm.svm_c = sparm.C; /* set upper bound C */
			// logger.info("lparm->svm_c:"+lparm.svm_c+" \n");	
			 lparm.sharedslack = 1;
            // logger.info(" lparm.sharedslack is 1");
		} else if (sparm.slack_norm == 2) {
			logger.info("ERROR: The joint algorithm does not apply to L2 slack norm!");

			System.exit(0);
		} else {
			logger.info("ERROR: Slack norm must be L1 or L2!");
			System.exit(0);
		}

		lparm.biased_hyperplane = 0; // set threshold to zero 
		epsilon = 100.0; //start with low precision and increase later
						
		epsilon_cached = epsilon;  //epsilon to use for iterations using constraints constructed from the constraint cache
								

		cset = ssa.initStructConstraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha_g = new double[cset.m];
			alphahist_g = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha_g[i] = 0;
				alphahist_g[i] = -1;
			}
		}

		kparm.gram_matrix = null;
		if ((alg_type == CommonStruct.ONESLACK_DUAL_ALG)
				|| (alg_type == CommonStruct.ONESLACK_DUAL_CACHE_ALG))
			kparm.gram_matrix = init_kernel_matrix(cset, kparm);

		// set initial model and slack variables
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;

		
		sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi, lparm,
				kparm, null, svmModel, alpha_g);

		// logger.info("sl totwords:"+svmModel.totwords);
		Common.add_weight_vector_to_linear_model(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights; // short cut to weight vector

		// create a cache of the feature vectors for the correct labels
		fycache = new SVECTOR[n];
		for (i = 0; i < n; i++) {
			if ( ModelConstant.USE_FYCACHE != 0) {
				// logger.info("USE THE FYCACHE \n");
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					diff = Common.add_list_sort_ss_r(fy,
							CommonStruct.COMPACT_ROUNDING_THRESH);
					fy = diff;
				}
			} else
				fy = null;
			fycache[i] = fy;
		}

		// initialize the constraint cache
		if (alg_type == ONESLACK_DUAL_CACHE_ALG) {
			// logger.info("create ONESLACK_DUAL_CACHE");
			// ccache_g=
			//create_constraint_cache(sample, sparm, sm);

			for (i = 0; i < n; i++)
				if (ssa.loss(ex[i].y, ex[i].y, sparm) != 0) {
					// logger.info("ERROR: Loss function returns non-zero value loss(y_"+i+",y_"+i+")\n");
					// logger.info("       W4 algorithm assumes that loss(y_i,y_i)=0 for all i.\n");
				}
		}

		if (kparm.kernel_type ==  ModelConstant.LINEAR) {
			// logger.info("kernel type is LINEAR \n");
			lhs_n = Common.create_nvector(sm.sizePsi);
		}
		// randomize order or training examples 
		if (batch_size < n)
			randmapping = Common.random_order(n);

		rt_init += Math.max(Common.get_runtime() - rt1, 0);
		rt_total += rt_init;

	
		//=== main loop=======
		do { // iteratively find and add constraints to working set 

			if (CommonStruct.struct_verbosity >= 1) {
				logger.info("in loop Iter " + (++numIt) + ": ");
				System.out.println("in loop Iter " + (numIt) + ": ");
			}

			rt1 = Common.get_runtime();

			// compute current slack 
			alphasum = 0;
			for (j = 0; (j < cset.m); j++)
			{
			    //logger.info("alpha_g["+j+"]="+alpha_g[j]);
				alphasum += alpha_g[j];
			}
			for (j = 0, slack = -1; (j < cset.m) && (slack == -1); j++) {
				//logger.info("alpha_g["+j+"]="+alpha_g[j]+" alphasum:"+alphasum+" cset.m:"+cset.m+" cset.rhs[j]:"+cset.rhs[j]+" ce:"+svm_common.classify_example(svmModel,cset.lhs[j]));
				if (alpha_g[j] > alphasum / cset.m) {
					slack = Math.max(0,cset.rhs[j]- Common.classify_example(svmModel,cset.lhs[j]));
				}
			}
			slack = Math.max(0, slack);
			//logger.info("slack val is :" + slack);

			rt_total += Math.max(Common.get_runtime() - rt1, 0);

			// find a violated joint constraint 
			lhs_g = null;
			rhs_g = 0;
			if (alg_type == ONESLACK_DUAL_CACHE_ALG) {
				rt1 = Common.get_runtime();
				//Compute violation of constraints in cache for current w 
				if (CommonStruct.struct_verbosity >= 2)
					rt2 = Common.get_runtime();

				// logger.info("svm model top weigths"+svmModel.topWeights()) ;
				//update_constraint_cache_for_model(svmModel);
				if (CommonStruct.struct_verbosity >= 2)
					rt_cacheupdate += Math.max(Common.get_runtime() - rt2,
							0);
				logger.info("epsilon_est is " + epsilon_est + " \n");
				System.out.println("epsilon_est is " + epsilon_est + " \n");

				// Is there is a sufficiently violated constraint in cache?
				//viol = compute_violation_of_constraint_in_cache(epsilon_est / 2);

				logger.info("viol=" + viol + " slack=" + slack + " epsilon_est"
						+ epsilon_est + " and sparm epsilon=" + sparm.epsilon);

				if (viol - slack > Math.max(epsilon_est / 10, sparm.epsilon)) {

					logger.info("There is a sufficiently violated constraint in cache");

					// There is a sufficiently violated constraint in cache, so
					// use this constraint in this iteration.
					if (CommonStruct.struct_verbosity >= 2)
						rt2 = Common.get_runtime();
					//viol = find_most_violated_joint_constraint_in_cache(epsilon_est / 2);
					if (CommonStruct.struct_verbosity >= 2)
						rt_cacheconst += Math.max(Common.get_runtime()
								- rt2, 0);
					cached_constraint = 1;
				} else {
					logger.info("There is no violated constraint in cache");
	
					// There is no sufficiently violated constraint in cache, so
					// update cache byprint_percent_progress computing most
					// violated constraint explicitly for batch_size examples
					viol_est = 0;
					progress = 0;
					Common.progress_n = progress;
					//viol = compute_violation_of_constraint_in_cache(0);
					for (j = 0; (j < batch_size)
							|| ((j < n) && (viol - slack < sparm.epsilon)); j++) {
						if (CommonStruct.struct_verbosity >= 1)
							Common.print_percent_progress(n, 10, ".");
						uptr = uptr % n;
						if (randmapping != null)
							i = randmapping[uptr];
						else
							i = uptr;
			
						//find most violating fydelta=fy-fybar and rhs for example i	
						find_most_violated_constraint(ex[i], fycache[i], n, sm,
								sparm);
						
						// add current fy-fybar and loss to cache
						if (CommonStruct.struct_verbosity >= 2)
							rt2 = Common.get_runtime();

						//viol += add_constraint_to_constraint_cache(
						//		sm.svm_model, i, 0.0001 * sparm.epsilon / n,
						//		sparm.ccache_size);

						if (CommonStruct.struct_verbosity >= 2)
							rt_cacheadd += Math.max(Common.get_runtime()
									- rt2, 0);
						//viol_est += ccache_g.constlist[i].viol;
						// logger.info("viol_est:"+viol_est+" i="+i);
						uptr++;
					}
					if (j < n) {
						cached_constraint = 1;
					} else {
						cached_constraint = 0;
					}
					if (CommonStruct.struct_verbosity >= 2)
						rt2 = Common.get_runtime();
					
					//if (cached_constraint != 0) {
						// logger.info("cached_constraint is not 0");
						// System.out.println("cached_constraint is not 0");
						//viol = find_most_violated_joint_constraint_in_cache(epsilon_est / 2);
						// logger.info(" cached_constraint lhs is :"+lhs.toString()+"  lhs_n is:"+lhs_n.toString());
					//} else {
						// logger.info("cached_constraint is  0");
						// System.out.println("cached_constraint is  0");
						//viol = find_most_violated_joint_constraint_in_cache(0);
						// logger.info(" no cached_constraint lhs is :"+lhs_g.toString());

					//}
					
					
					if (CommonStruct.struct_verbosity >= 2)
						rt_cacheconst += Math.max(Common.get_runtime()
								- rt2, 0);
					viol_est *= ((double) n / (double) j);
					logger.info("viol_est=" + viol_est + " slack=" + slack + "");
					epsilon_est = (1 - (double) j / (double) n) * epsilon_est
							+ (double) j / (double) n * (viol_est - slack);// epsilon_est璧嬪�
					logger.info("epsilon_est cal=" + epsilon_est);
					if ((CommonStruct.struct_verbosity >= 1) && (j != n))
						logger.info("(upd=" + (100.0 * j / n) + ",eps^="
								+ (viol_est - slack) + ",eps*=" + epsilon_est
								+ ")");
				}

				lhsXw = rhs_g - viol;

				rt_total += Math.max(Common.get_runtime() - rt1, 0);
			} else {
				// do not use constraint from cache
				rt1 = Common.get_runtime();
				cached_constraint = 0;
				if (kparm.kernel_type ==  ModelConstant.LINEAR)
					Common.clear_nvector(lhs_n, sm.sizePsi);
				Common.progress_n = 0;
				rt_total += Math.max(Common.get_runtime() - rt1, 0);

				for (i = 0; i < n; i++) {
					rt1 = Common.get_runtime();

					if (CommonStruct.struct_verbosity >= 1)
						Common.print_percent_progress(n, 10, ".");


					// compute most violating fydelta=fy-fybar and rhs for example i
					find_most_violated_constraint(ex[i], fycache[i], n, sm,
							sparm);
					// add current fy-fybar to lhs of constraint
					if (kparm.kernel_type ==  ModelConstant.LINEAR) {
						Common.add_list_n_ns(lhs_n, fydelta_g, 1.0);
					} else {
						Common.append_svector_list(fydelta_g, lhs_g);

						lhs_g = fydelta_g;
					}
					rhs_g += rhs_i_g;// add loss to rhs 

					rt_total += Math.max(Common.get_runtime() - rt1, 0);

				} // end of example loop 

				rt1 = Common.get_runtime();

				// create sparse vector from dense sum 
				System.out.println("kernel type is " + kparm.kernel_type);
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					lhs_g = Common.create_svector_n_r(lhs_n, sm.sizePsi,
							null, 1.0,
							CommonStruct.COMPACT_ROUNDING_THRESH);
				}
				doc = Common.create_example(cset.m, 0, 1, 1, lhs_g);
				lhsXw = Common.classify_example(svmModel, doc);

				viol = rhs_g - lhsXw;

				rt_total += Math.max(Common.get_runtime() - rt1, 0);

			} // end of finding most violated joint constraint

			rt1 = Common.get_runtime();

			// if `error', then add constraint and recompute QP 
			if (slack > (rhs_g - lhsXw + 0.000001)) {
				logger.info("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
				logger.info(" set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
				logger.info("slack=" + slack + ", newslack=" + (rhs_g - lhsXw)
						+ "\n");

			}
			ceps = Math.max(0, rhs_g - lhsXw - slack);
			if ((ceps > sparm.epsilon) || cached_constraint != 0) {
				// resize constraint matrix and add new constraint
				// cset.lhs=new DOC[cset.m+1];
				int ti = 0;
				ti = cset.m + 1;
				cset.lhs = Common.reallocDOCS(cset.lhs, ti);

				cset.lhs[cset.m] = Common.create_example(cset.m, 0, 1, 1,
						lhs_g);

				// cset.rhs = new double[cset.m + 1];
				cset.rhs = Common
						.reallocDoubleArr(cset.rhs, cset.m + 1);
				cset.rhs[cset.m] = rhs_g;
				// alpha = new double[cset.m + 1];
				alpha_g = Common.reallocDoubleArr(alpha_g, cset.m + 1);
				alpha_g[cset.m] = 0;
				// alphahist = new int[cset.m + 1];
				alphahist_g = Common.reallocIntArr(alphahist_g,
						cset.m + 1);
				alphahist_g[cset.m] = optcount;
				cset.m++;
				totconstraints++;
				if ((alg_type == ONESLACK_DUAL_ALG)
						|| (alg_type == ONESLACK_DUAL_CACHE_ALG)) {
					if (CommonStruct.struct_verbosity >= 2)
						rt2 = Common.get_runtime();
					kparm.gram_matrix = update_kernel_matrix(kparm.gram_matrix,
							cset.m - 1, cset, kparm);
					if (CommonStruct.struct_verbosity >= 2)
						rt_kernel += Math
								.max(Common.get_runtime() - rt2, 0);
				}

				// get new QP solution
				if (CommonStruct.struct_verbosity >= 1) {
					// logger.info("*");
				}
				if (CommonStruct.struct_verbosity >= 2)
					rt2 = Common.get_runtime();

				// set svm precision so that higher than eps of most violated constr
				if (cached_constraint != 0) {
					epsilon_cached = Math.min(epsilon_cached, ceps);
					lparm.epsilon_crit = epsilon_cached / 2;
				} else {
					epsilon = Math.min(epsilon, ceps); /* best eps so far */
					lparm.epsilon_crit = epsilon / 2;
					epsilon_cached = epsilon;
				}

				svmModel = new MODEL();
				// Run the QP solver on cset. 
				kernel_type_org = kparm.kernel_type;
				if ((alg_type == CommonStruct.ONESLACK_DUAL_ALG)
						|| (alg_type == CommonStruct.ONESLACK_DUAL_CACHE_ALG))
					kparm.kernel_type =  ModelConstant.GRAM; // use kernel stored in kparm

				sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi,
						lparm, kparm, null, svmModel, alpha_g);
				kparm.kernel_type = (short) kernel_type_org;

				svmModel.kernel_parm.kernel_type = (short) kernel_type_org;

				// Always add weight vector, in case part of the kernel is
				// linear. If not, ignore the weight vector since its content is bogus.
				Common.add_weight_vector_to_linear_model(svmModel);

				// sm.svm_model = svmModel.copyMODEL();
				
				sm.svm_model = svmModel;

				// sm.svm_model = svmModel; sm.w = svmModel.lin_weights;
				sm.w = new double[svmModel.lin_weights.length];
				for (int iw = 0; iw < svmModel.lin_weights.length; iw++) {
					sm.w[iw] = svmModel.lin_weights[iw];
				}


				optcount++;

				//keep track of when each constraint was last active.
				// constraints marked with -1 are not updated
				for (j = 0; j < cset.m; j++) {
					if ((alphahist_g[j] > -1) && (alpha_g[j] != 0)) {
						alphahist_g[j] = optcount;
					}
				}
				if (CommonStruct.struct_verbosity >= 2)
					rt_opt += Math.max(Common.get_runtime() - rt2, 0);


				//Check if some of the linear constraints have not been active
				// in a while. Those constraints are then removed to avoid
				// bloating the working set beyond necessity.
				if (CommonStruct.struct_verbosity >= 3) {
					logger.info("Reducing working set...");
				}

			    // 在这里要将某些限制去掉
				// logger.info("cset.m before:"+cset.m);
				remove_inactive_constraints(cset, optcount, 50);
				// logger.info("cset.m after:"+cset.m);
				if (CommonStruct.struct_verbosity >= 3)
					logger.info("done. ");

			} else {

			}

			//if (svm_struct_common.struct_verbosity >= 1) {
				logger.info("(NumConst=" + cset.m + ", SV="
						+ (svmModel.sv_num - 1) + ", CEps=" + ceps + ", QPEps="
						+ svmModel.maxdiff + ")\n");
				System.out.println("(NumConst=" + cset.m + ", SV="
						+ (svmModel.sv_num - 1) + ", CEps=" + ceps + ", QPEps="
						+ svmModel.maxdiff + ")\n");
			//}

			rt_total += Math.max(Common.get_runtime() - rt1, 0);

			
		} while (cached_constraint != 0
				|| (ceps > sparm.epsilon)
				|| ssa.finalizeIteration(ceps, cached_constraint,
						sample, sm, cset, alpha_g, sparm));

		if (CommonStruct.struct_verbosity >= 1) {
			logger.info("Final epsilon on KKT-Conditions: "
					+ (Math.max(svmModel.maxdiff, ceps)) + "\n");

			slack = 0;
			for (j = 0; j < cset.m; j++)
				slack = Math.max(
						slack,
						cset.rhs[j]
								- Common.classify_example(svmModel,
										cset.lhs[j]));
			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha_g[i] * cset.rhs[i];
			if (kparm.kernel_type ==  ModelConstant.LINEAR)
				modellength = Common.model_length_n(svmModel);
			else
				modellength = Common.model_length_s(svmModel);
			dualitygap = (0.5 * modellength * modellength + sparm.C * viol)
					- (alphasum - 0.5 * modellength * modellength);

			logger.info("Upper bound on duality gap: " + dualitygap + "\n");
			logger.info("Dual objective value: dval="
					+ (alphasum - 0.5 * modellength * modellength) + "\n");
			logger.info("Primal objective value: pval="
					+ (0.5 * modellength * modellength + sparm.C * viol) + "\n");
			logger.info("Total number of constraints in final working set: "
					+ ((int) cset.m) + " (of " + ((int) totconstraints) + ")\n");
			logger.info("Number of iterations: " + numIt + "\n");
			logger.info("Number of calls to 'find_most_violated_constraint': "
					+ argmax_count_g + "\n");
			logger.info("Number of SV: " + (svmModel.sv_num - 1) + " \n");
			logger.info("Norm of weight vector: |w|=" + modellength + "\n");
			logger.info("Value of slack variable (on working set): xi=" + slack
					+ "\n");
			logger.info("Value of slack variable (global): xi=" + viol + "\n");
			logger.info("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||="
					+ (sl.length_of_longest_document_vector(cset.lhs, cset.m,
							kparm)) + "\n");
			if (CommonStruct.struct_verbosity >= 2)
				logger.info("Runtime in cpu-seconds: " + (rt_total / 100.0)
						+ " (" + ((100.0 * rt_opt) / rt_total) + " for QP, "
						+ ((100.0 * rt_kernel) / rt_total) + " for kernel, "
						+ ((100.0 * rt_viol_g) / rt_total) + "for Argmax, "
						+ ((100.0 * rt_psi_g) / rt_total) + " for Psi, "
						+ ((100.0 * rt_init) / rt_total) + " for init, "
						+ ((100.0 * rt_cacheupdate) / rt_total)
						+ " for cache update, "
						+ ((100.0 * rt_cacheconst) / rt_total)
						+ " for cache const, "
						+ ((100.0 * rt_cacheadd) / rt_total)
						+ " for cache add (incl. "
						+ ((100.0 * rt_cachesum_g) / rt_total) + " for sum))\n");
			else if (CommonStruct.struct_verbosity == 1)
				logger.info("Runtime in cpu-seconds: " + (rt_total / 100.0)
						+ "\n");
		}


		if (CommonStruct.struct_verbosity >= 4)
			CommonStruct.printW(sm.w, sizePsi, n, lparm.svm_c);

		if (svmModel != null) {
			if (svmModel.kernel_parm == null) {
				logger.info("svmModel kernel_parm is null");
			}

			logger.info("sv num there:" + svmModel.sv_num);
			// sm.svm_model = svm_common.copy_model(svmModel);
			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; /* short cut to weight vector */

		}


		ssa.printStructLearningStats(sample, sm, cset, alpha_g,
				sparm);

	}

	/**
	 * removes the constraints from cset (and alpha) for which alphahist
	 * indicates that they have not been active for at least mininactive
	 * iterations
	 */
	public void remove_inactive_constraints(CONSTSET cset, int currentiter,
			int mininactive)
	{
		int i, m;

		m = 0;
		for (i = 0; i < cset.m; i++) {
			if ((alphahist_g[i] < 0)
					|| ((currentiter - alphahist_g[i]) < mininactive)) {
			
				// keep constraints that are marked as -1 or which have recently
				// been active
				cset.lhs[m] = cset.lhs[i];
				cset.lhs[m].docnum = m;
				cset.rhs[m] = cset.rhs[i];
				alpha_g[m] = alpha_g[i];
				alphahist_g[m] = alphahist_g[i];
				m++;
			} else {
			}
		}

		if (cset.m != m) {
			cset.m = m;
			Common.realsmallloc_lhs(cset);
			Common.realsmallloc_rhs(cset);

			alpha_g = Common.reallocDoubleArr(alpha_g, cset.m);
			alphahist_g = Common.reallocIntArr(alphahist_g, cset.m);

		}
	}

	/**
	 * assigns a kernelid to each constraint in cset and creates the
	 * corresponding kernel matrix.
	 */
	public MATRIX init_kernel_matrix(CONSTSET cset, KERNEL_PARM kparm)
	{
		int i, j;
		double kval;
		MATRIX matrix;

		// assign kernel id to each new constraint 
		for (i = 0; i < cset.m; i++)
			cset.lhs[i].kernelid = i;

		//allocate kernel matrix as necessary
		matrix = Common.create_matrix(i + 50, i + 50);

		for (j = 0; j < cset.m; j++) {
			for (i = j; i < cset.m; i++) {
				kval = Common.kernel(kparm, cset.lhs[j], cset.lhs[i]);
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
	public void find_most_violated_constraint(EXAMPLE ex, SVECTOR fycached,
			int n, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm)
	{
		double rt2 = 0;
		LABEL ybar;
		SVECTOR fybar, fy;
		double factor, lossval;


		if (CommonStruct.struct_verbosity >= 2)
			rt2 = Common.get_runtime();
		argmax_count_g++;
		if (sparm.loss_type == SLACK_RESCALING) {
			ybar = ssa.findMostViolatedConstraintSlackrescaling(
					ex.x, ex.y, sm, sparm);
		} else {
			ybar =ssa.findMostViolatedConstraintMarginrescaling(ex.x, ex.y,
							sm, sparm);
		}
		if (CommonStruct.struct_verbosity >= 2)
			rt_viol_g += Math.max(Common.get_runtime() - rt2, 0);

		if (ssa.emptyLabel(ybar)) {
			logger.info("ERROR: empty label was returned for example\n");
		}

		// get psi(x,y) and psi(x,ybar)
		if (CommonStruct.struct_verbosity >= 2)
			rt2 = Common.get_runtime();
		if (fycached != null)
			fy = Common.copy_svector(fycached);
		else
			fy = ssa.psi(ex.x, ex.y, sm, sparm);
		fybar = ssa.psi(ex.x, ybar, sm, sparm);
		if (CommonStruct.struct_verbosity >= 2)
			rt_psi_g += Math.max(Common.get_runtime() - rt2, 0);
		lossval = ssa.loss(ex.y, ybar, sparm);

		// scale feature vector and margin by loss 
		if (sparm.loss_type == SLACK_RESCALING)
			factor = lossval / n;
		else
			factor = 1.0 / n; // do not rescale vector formargin rescaling loss type 
		Common.mult_svector_list(fy, factor);
		Common.mult_svector_list(fybar, -factor);
		Common.append_svector_list(fybar, fy); // compute fy-fybar 

		fydelta_g = fybar;

		//rhs_i_g = lossval / n;
		rhs_i_g=WU.div(lossval, n, 20);

	}
	
	/**
	 * assigns new kernelid to constraint in position newpos and fills the
	 * corresponding part of the kernel matrix
	 */
	public MATRIX update_kernel_matrix(MATRIX matrix, int newpos,
			CONSTSET cset, KERNEL_PARM kparm)
	{
		int i, maxkernelid = 0, newid;
		double kval;
		double[] used;

		// find free kernelid to assign to new constraint 
		for (i = 0; i < cset.m; i++) {
			if (i != newpos) {
				maxkernelid = Math.max(maxkernelid, cset.lhs[i].kernelid);
			}
		}
		used = Common.create_nvector(maxkernelid + 2);
		Common.clear_nvector(used, maxkernelid + 2);
		for (i = 0; i < cset.m; i++)
			if (i != newpos)
				used[cset.lhs[i].kernelid] = 1;
		for (newid = 0; used[newid] != 0; newid++)
			;
		cset.lhs[newpos].kernelid = newid;

		// extend kernel matrix if necessary
		maxkernelid = Math.max(maxkernelid, newid);
		if ((matrix == null) || (maxkernelid >= matrix.m))
			matrix = Common.realloc_matrix(matrix, maxkernelid + 50,
					maxkernelid + 50);

		for (i = 0; i < cset.m; i++) {
			kval = Common.kernel(kparm, cset.lhs[newpos], cset.lhs[i]);
			matrix.element[newid][cset.lhs[i].kernelid] = kval;
			matrix.element[cset.lhs[i].kernelid][newid] = kval;
		}
		return (matrix);
	}

	public void svm_learn_struct_joint_custom(SAMPLE sample,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm,
			STRUCTMODEL sm) {

	}

}
