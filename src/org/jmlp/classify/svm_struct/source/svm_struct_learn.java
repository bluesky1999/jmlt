package org.jmlp.classify.svm_struct.source;

import org.apache.log4j.Logger;

/**
 * Basic algorithm for learning structured outputs (e.g. parses, sequences,
 * multi-label classification) with a Support Vector Machine.
 * 
 * @author lq
 */

public class svm_struct_learn {

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
	public CCACHE ccache_g;
	public SVECTOR fydelta_g = null;

	public double[] alpha_g = null;
	public int[] alphahist_g = null;
	public int[] remove_g = null;

	private svm_struct_api ssa=null;
	
	private static Logger logger = Logger.getLogger(svm_struct_learn.class);
	
	public svm_struct_learn()
	{
		ssa=svm_struct_api_factory.get_svm_struct_api();
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
		rt1 = svm_common.get_runtime();

		ssa.init_struct_model(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1;
		logger.info("the sizePsi2 is " + sizePsi);
		if (alg_type == svm_struct_common.NSLACK_SHRINK_ALG) {
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
			lparm.svm_c = 999999999999999.0; /*
											 * upper bound C must never be
											 * reached
											 */
			lparm.sharedslack = 0;
			if (kparm.kernel_type != svm_common.LINEAR) {
				logger.error("ERROR: Kernels are not implemented for L2 slack norm!");
				System.exit(0);
			}
		} else {
			logger.error("ERROR: Slack norm must be L1 or L2!");
		}

		epsilon = 100.0;
		tolerance = Math.min(n / 3, Math.max(n / 100, 5));
		lparm.biased_hyperplane = 0;

		cset = svm_struct_api.init_struct_constraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha_g = new double[cset.m];
			alphahist_g = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha_g[i] = 0;
				alphahist_g[i] = -1; /*
									 * -1 makes sure these constraints are never
									 * removed
									 */
			}
		}

		/* set initial model and slack variables */
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;
		if (kparm.kernel_type != svm_common.LINEAR) {
			kcache = svm_learn.kernel_cache_init(Math.max(cset.m, 1),
					lparm.kernel_cache_size);
		}
		svm_learn sl = new svm_learn();
		logger.info("sizePsi:" + sizePsi + "   n:" + n);
		sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi + n,
				lparm, kparm, kcache, svmModel, alpha_g);
		svm_common.add_weight_vector_to_linear_model(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights;

		if (svm_common.USE_FYCACHE != 0) {
			fycache = new SVECTOR[n];
			for (i = 0; i < n; i++) {
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);// temp
																		// point
				if (kparm.kernel_type == svm_common.LINEAR) {
					diff = svm_common.add_list_ss(fy);
					fy = diff;
				}

			}

		}

		rt_init += Math.max(svm_common.get_runtime() - rt1, 0);
		rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

		/*****************/
		/*** main loop ***/
		/*****************/

		do { /* iteratively increase precision */

			epsilon = Math.max(epsilon * 0.49999999999, sparm.epsilon);
			new_precision = 1;

			if (epsilon == sparm.epsilon) /* for final precision, find all SV */
			{
				tolerance = 0;
			}

			lparm.epsilon_crit = epsilon / 2; /*
											 * svm precision must be higher than
											 * eps
											 */
			if (svm_struct_common.struct_verbosity >= 1) {
				logger.info("Setting current working precision to " + epsilon);
			}

			do {/*
				 * iteration until (approx) all SV are found for current
				 * precision and tolerance
				 */

				opti_round++;
				activenum = n;
				dont_stop = 0;
				old_totconstraints = totconstraints;

				do { /*
					 * with shrinking turned on, go through examples that keep
					 * producing new constraints
					 */

					if (svm_struct_common.struct_verbosity >= 1) {
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
						/*** example loop ***/

						rt1 = svm_common.get_runtime();

						if ((use_shrinking == 0) || (opti[i] != opti_round)) {
							/*
							 * if the example is not shrunk away, then see if it
							 * is necessary to add a new constraint
							 */
							rt2 = svm_common.get_runtime();
							argmax_count++;
							if (sparm.loss_type == SLACK_RESCALING) {
								ybar = ssa.find_most_violated_constraint_slackrescaling(
												ex[i].x, ex[i].y, sm, sparm);
							} else {
								ybar = ssa.find_most_violated_constraint_marginrescaling(
												ex[i].x, ex[i].y, sm, sparm);
								// logger.info("i="+i+"  ybar is "+ybar.class_index+" ex.y is "+ex[i].y.class_index);
							}
							rt_viol += Math.max(svm_common.get_runtime() - rt2,
									0);

							if (ssa.empty_label(ybar)) {
								if (opti[i] != opti_round) {
									activenum--;
									opti[i] = opti_round;
								}
								if (svm_struct_common.struct_verbosity >= 2)
									logger.info("no-incorrect-found(" + i
											+ ") ");
								continue;
							}

							/**** get psi(y)-psi(ybar) ****/
							rt2 = svm_common.get_runtime();
							if (fycache != null) {
								fy = svm_common.copy_svector(fycache[i]);
							} else {
								fy = ssa.psi(ex[i].x, ex[i].y, sm,
										sparm);
							}

							fybar = ssa.psi(ex[i].x, ybar, sm, sparm);
							// logger.info("fy:"+fy.toString());
							// logger.info("fybar:"+fybar.toString());
							rt_psi += Math.max(svm_common.get_runtime() - rt2,
									0);

							/**** scale feature vector and margin by loss ****/
							lossval = ssa.loss(ex[i].y, ybar, sparm);
							if (sparm.slack_norm == 2)
								lossval = Math.sqrt(lossval);
							if (sparm.loss_type == SLACK_RESCALING)
								factor = lossval;
							else
								/* do not rescale vector for */
								factor = 1.0; /* margin rescaling loss type */
							for (f = fy; f != null; f = f.next) {
								f.factor *= factor;
							}

							for (f = fybar; f != null; f = f.next)
								f.factor *= -factor;
							margin = lossval;

							/**** create constraint for current ybar ****/
							svm_common.append_svector_list(fy, fybar);
							doc = svm_common.create_example(cset.m, 0, i + 1,
									1, fy);

							/**** compute slack for this example ****/
							slack = 0;
							for (j = 0; j < cset.m; j++)
								if (cset.lhs[j].slackid == i + 1) {
									if (sparm.slack_norm == 2)
										slack = Math
												.max(slack,
														cset.rhs[j]
																- (svm_common
																		.classify_example(
																				svmModel,
																				cset.lhs[j]) - sm.w[sizePsi
																		+ i]
																		/ (Math.sqrt(2 * svmCnorm))));
									else
										slack = Math
												.max(slack,
														cset.rhs[j]
																- svm_common
																		.classify_example(
																				svmModel,
																				cset.lhs[j]));
								}

							/**** if `error' add constraint and recompute ****/
							dist = svm_common.classify_example(svmModel, doc);
							ceps = Math.max(ceps, margin - dist - slack);
							if (slack > (margin - dist + 0.0001)) {
								logger.debug("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
								logger.debug("         set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
								logger.debug("Ex " + i + ": slack=" + slack
										+ ", newslack=" + (margin - dist)
										+ "\n");
								/* exit(1); */
							}
							if ((dist + slack) < (margin - epsilon)) {
								if (svm_struct_common.struct_verbosity >= 2) {
									logger.info("(" + i + ",eps="
											+ (margin - dist - slack) + ") ");
								}
								if (svm_struct_common.struct_verbosity == 1) {
									// logger.info(".");
									System.out.print(".");
								}

								/****
								 * resize constraint matrix and add new
								 * constraint
								 ****/
								cset.m++;
								svm_struct_api.realloc(cset);

								if (kparm.kernel_type == svm_common.LINEAR) {
									diff = svm_common.add_list_ss(fy);
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = svm_common
												.create_example(
														cset.m - 1,
														0,
														i + 1,
														1,
														svm_common
																.copy_svector(diff));
									else if (sparm.slack_norm == 2) {
										/****
										 * add squared slack variable to feature
										 * vector
										 ****/
										slackv[0].wnum = sizePsi + i;
										slackv[0].weight = 1 / (Math
												.sqrt(2 * svmCnorm));
										slackv[1].wnum = 0; /* terminator */
										slackvec = svm_common.create_svector(
												slackv, null, 1.0);
										cset.lhs[cset.m - 1] = svm_common
												.create_example(cset.m - 1, 0,
														i + 1, 1,
														svm_common.add_ss(diff,
																slackvec));
									}
								} else { /* kernel is used */
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = svm_common
												.create_example(
														cset.m - 1,
														0,
														i + 1,
														1,
														svm_common
																.copy_svector(fy));
									else if (sparm.slack_norm == 2)
										System.exit(1);
								}
								svm_struct_api.realloc_rhs(cset);
								cset.rhs[cset.m - 1] = margin;

								alpha_g = svm_struct_api.realloc_alpha(alpha_g,
										cset.m);
								alpha_g[cset.m - 1] = 0;
								alphahist_g = svm_struct_api
										.realloc_alpha_list(alphahist_g, cset.m);
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

						/**** get new QP solution ****/
						if ((newconstraints >= sparm.newconstretrain)
								|| ((newconstraints > 0) && (i == n - 1))
								|| ((new_precision != 0) && (i == n - 1))) {
							if (svm_struct_common.struct_verbosity >= 1) {
								// logger.info("*");
							}
							rt2 = svm_common.get_runtime();

							svmModel = new MODEL();
							/*
							 * Always get a new kernel cache. It is not possible
							 * to use the same cache for two different training
							 * runs
							 */

							if (kparm.kernel_type != svm_common.LINEAR)
								kcache = sl.kernel_cache_init(
										Math.max(cset.m, 1),
										lparm.kernel_cache_size);
							/* Run the QP solver on cset. */
							sl.svm_learn_optimization(cset.lhs, cset.rhs,
									cset.m, sizePsi + n, lparm, kparm, kcache,
									svmModel, alpha_g);
							/*
							 * Always add weight vector, in case part of the
							 * kernel is linear. If not, ignore the weight
							 * vector since its content is bogus.
							 */
							svm_common
									.add_weight_vector_to_linear_model(svmModel);
							sm.svm_model = svmModel.copyMODEL();
							sm.w = new double[svmModel.lin_weights.length];
							for (int k = 0; k < svmModel.lin_weights.length; k++) {
								sm.w[k] = svmModel.lin_weights[k];
							}
							// sm.w = svmModel.lin_weights;
							optcount++;
							/*
							 * keep track of when each constraint was last
							 * active. constraints marked with -1 are not
							 * updated
							 */
							for (j = 0; j < cset.m; j++) {
								if ((alphahist_g[j] > -1) && (alpha_g[j] != 0)) {
									alphahist_g[j] = optcount;
								}
								// logger.info("j="+j+"");
							}
							rt_opt += Math.max(svm_common.get_runtime() - rt2,
									0);

							if ((new_precision != 0)
									&& (epsilon <= sparm.epsilon))
								dont_stop = 1; /*
												 * make sure we take one final
												 * pass
												 */
							new_precision = 0;
							newconstraints = 0;
						}

						rt_total += Math.max(svm_common.get_runtime() - rt1, 0);
					}// exmample loop

					rt1 = svm_common.get_runtime();

					//if (svm_struct_common.struct_verbosity >= 1)
						logger.info("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");
						System.out.println("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");

					if (svm_struct_common.struct_verbosity >= 2)
						logger.info("Reducing working set...");

					remove_inactive_constraints(cset, optcount,
							Math.max(50, optcount - lastoptcount));

					lastoptcount = optcount;
					if (svm_struct_common.struct_verbosity >= 2)
						logger.info("done. (NumConst=" + cset.m + ")\n");

					rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

				} while ((use_shrinking != 0) && (activenum > 0));
			} while (((totconstraints - old_totconstraints) > tolerance)
					|| (dont_stop != 0));

		} while ((epsilon > sparm.epsilon)
				|| svm_struct_api.finalize_iteration(ceps, 0, sample, sm, cset,
						alpha_g, sparm)); // main_loop

		if (svm_struct_common.struct_verbosity >= 1) {
			/**** compute sum of slacks ****/
			/****
			 * WARNING: If positivity constraints are used, then the maximum
			 * slack id is larger than what is allocated below
			 ****/
			slacks = new double[n + 1];
			for (i = 0; i <= n; i++) {
				slacks[i] = 0;
			}

			if (sparm.slack_norm == 1) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(
							slacks[cset.lhs[j].slackid],
							cset.rhs[j]
									- svm_common.classify_example(svmModel,
											cset.lhs[j]));
			} else if (sparm.slack_norm == 2) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(
							slacks[cset.lhs[j].slackid],
							cset.rhs[j]
									- (svm_common.classify_example(svmModel,
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
			modellength = svm_common.model_length_s(svmModel);
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

		if (svm_struct_common.struct_verbosity >= 4)
			logger.info(svm_struct_common.printW(sm.w, sizePsi, n, lparm.svm_c));

		if (svmModel != null) {
			// sm.svm_model = svm_common.copy_model(svmModel);
			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; /* short cut to weight vector */
			String wstr = "";
			for (int wi = 0; wi < sm.w.length; wi++) {
				wstr += (wi + ":" + sm.w[wi] + " ");
			}
			logger.info("wstr:" + wstr);
		}

		svm_struct_api.print_struct_learning_stats(sample, sm, cset, alpha_g,
				sparm);

		/*
		 * svm_struct_api.write_struct_model("temp/med/testmodel",sm,sparm);
		 * String[] argt=new String[3];
		 * 
		 * argt[0]="temp/med/test.txt"; argt[1]="temp/med/model";
		 * argt[2]="temp/med/pred"; svm_struct_classify ssc=new
		 * svm_struct_classify(); ssc.classfiy(argt, sm,sparm,svmModel,sample);
		 */
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
		ccache_g = null;
		int cached_constraint;
		double viol, viol_est, epsilon_est = 0;
		int uptr = 0;
		int[] randmapping = null;
		int batch_size = n;

		svm_learn sl = new svm_learn();
		rt1 = svm_common.get_runtime();
		if (sparm.batch_size < 100)
			batch_size = (int) ((sparm.batch_size * n) / 100.0);
		ssa.init_struct_model(sample, sm, sparm, lparm, kparm);
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

		lparm.biased_hyperplane = 0; /* set threshold to zero */
		epsilon = 100.0; /*
						 * start with low precision and increase later
						 */
		epsilon_cached = epsilon; /*
								 * epsilon to use for iterations using
								 * constraints constructed from the constraint
								 * cache
								 */

		cset = svm_struct_api.init_struct_constraints(sample, sm, sparm);

		if (cset.m > 0) {
			alpha_g = new double[cset.m];
			alphahist_g = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha_g[i] = 0;
				alphahist_g[i] = -1;
			}
		}

		kparm.gram_matrix = null;
		if ((alg_type == svm_struct_common.ONESLACK_DUAL_ALG)
				|| (alg_type == svm_struct_common.ONESLACK_DUAL_CACHE_ALG))
			kparm.gram_matrix = init_kernel_matrix(cset, kparm);

		/* set initial model and slack variables */
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;

		
		sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi, lparm,
				kparm, null, svmModel, alpha_g);

		// logger.info("sl totwords:"+svmModel.totwords);
		svm_common.add_weight_vector_to_linear_model(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights; /* short cut to weight vector */

		/* create a cache of the feature vectors for the correct labels */
		fycache = new SVECTOR[n];
		for (i = 0; i < n; i++) {
			if (svm_common.USE_FYCACHE != 0) {
				// logger.info("USE THE FYCACHE \n");
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);
				if (kparm.kernel_type == svm_common.LINEAR) {
					diff = svm_common.add_list_sort_ss_r(fy,
							svm_struct_common.COMPACT_ROUNDING_THRESH);
					fy = diff;
				}
			} else
				fy = null;
			fycache[i] = fy;
		}

		/* initialize the constraint cache */
		if (alg_type == ONESLACK_DUAL_CACHE_ALG) {
			// logger.info("create ONESLACK_DUAL_CACHE");
			// ccache_g=
			create_constraint_cache(sample, sparm, sm);

			/* NOTE: */
			for (i = 0; i < n; i++)
				if (ssa.loss(ex[i].y, ex[i].y, sparm) != 0) {
					// logger.info("ERROR: Loss function returns non-zero value loss(y_"+i+",y_"+i+")\n");
					// logger.info("       W4 algorithm assumes that loss(y_i,y_i)=0 for all i.\n");
				}
		}

		if (kparm.kernel_type == svm_common.LINEAR) {
			// logger.info("kernel type is LINEAR \n");
			lhs_n = svm_common.create_nvector(sm.sizePsi);
		}
		/* randomize order or training examples */
		if (batch_size < n)
			randmapping = svm_common.random_order(n);

		rt_init += Math.max(svm_common.get_runtime() - rt1, 0);
		rt_total += rt_init;

		/*****************/
		/*** main loop ***/
		/*****************/
		do { /* iteratively find and add constraints to working set */

			if (svm_struct_common.struct_verbosity >= 1) {
				logger.info("in loop Iter " + (++numIt) + ": ");
				System.out.println("in loop Iter " + (numIt) + ": ");
			}

			rt1 = svm_common.get_runtime();

			/**** compute current slack ****/
			alphasum = 0;
			for (j = 0; (j < cset.m); j++)
			{
			    //logger.info("alpha_g["+j+"]="+alpha_g[j]);
				alphasum += alpha_g[j];
			}
			for (j = 0, slack = -1; (j < cset.m) && (slack == -1); j++) {
				//logger.info("alpha_g["+j+"]="+alpha_g[j]+" alphasum:"+alphasum+" cset.m:"+cset.m+" cset.rhs[j]:"+cset.rhs[j]+" ce:"+svm_common.classify_example(svmModel,cset.lhs[j]));
				if (alpha_g[j] > alphasum / cset.m) {
					slack = Math.max(0,cset.rhs[j]- svm_common.classify_example(svmModel,cset.lhs[j]));
				}
			}
			slack = Math.max(0, slack);
			//logger.info("slack val is :" + slack);

			rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

			/**** find a violated joint constraint ****/
			lhs_g = null;
			rhs_g = 0;
			if (alg_type == ONESLACK_DUAL_CACHE_ALG) {
				rt1 = svm_common.get_runtime();
				/* Compute violation of constraints in cache for current w */
				if (svm_struct_common.struct_verbosity >= 2)
					rt2 = svm_common.get_runtime();

				// logger.info("svm model top weigths"+svmModel.topWeights()) ;
				update_constraint_cache_for_model(svmModel);
				if (svm_struct_common.struct_verbosity >= 2)
					rt_cacheupdate += Math.max(svm_common.get_runtime() - rt2,
							0);
				logger.info("epsilon_est is " + epsilon_est + " \n");
				System.out.println("epsilon_est is " + epsilon_est + " \n");

				/* Is there is a sufficiently violated constraint in cache? */
				viol = compute_violation_of_constraint_in_cache(epsilon_est / 2);

				logger.info("viol=" + viol + " slack=" + slack + " epsilon_est"
						+ epsilon_est + " and sparm epsilon=" + sparm.epsilon);

				if (viol - slack > Math.max(epsilon_est / 10, sparm.epsilon)) {

					logger.info("There is a sufficiently violated constraint in cache");
					/*
					 * There is a sufficiently violated constraint in cache, so
					 * use this constraint in this iteration.
					 */
					if (svm_struct_common.struct_verbosity >= 2)
						rt2 = svm_common.get_runtime();
					viol = find_most_violated_joint_constraint_in_cache(epsilon_est / 2);
					if (svm_struct_common.struct_verbosity >= 2)
						rt_cacheconst += Math.max(svm_common.get_runtime()
								- rt2, 0);
					cached_constraint = 1;
				} else {
					logger.info("There is no violated constraint in cache");
					/*
					 * There is no sufficiently violated constraint in cache, so
					 * update cache byprint_percent_progress computing most
					 * violated constraint explicitly for batch_size examples.
					 */
					viol_est = 0;
					progress = 0;
					svm_common.progress_n = progress;
					viol = compute_violation_of_constraint_in_cache(0);
					for (j = 0; (j < batch_size)
							|| ((j < n) && (viol - slack < sparm.epsilon)); j++) {
						if (svm_struct_common.struct_verbosity >= 1)
							svm_common.print_percent_progress(n, 10, ".");
						uptr = uptr % n;
						if (randmapping != null)
							i = randmapping[uptr];
						else
							i = uptr;
						/*
						 * find most violating fydelta=fy-fybar and rhs for
						 * example i
						 */
						/*
						 * String winfo=""; if(ex[i].x.doc!=null) { for(int
						 * l=0;l<ex[i].x.doc.fvec.words.length;l++) {
						 * winfo=winfo
						 * +ex[i].x.doc.fvec.words[l].wnum+":"+ex[i].x
						 * .doc.fvec.words[l].weight+" "; }
						 * logger.info("winfoii["+i+"]:"+winfo); }
						 */
						/*
						 * if(fydelta!=null) { winfo=""; for(int
						 * l=0;l<fydelta.words.length;l++) {
						 * winfo=winfo+fydelta.
						 * words[l].wnum+":"+fydelta.words[l].weight+" "; }
						 * logger.info("winfoiiii["+i+"]:"+winfo);
						 * 
						 * }
						 */
						// logger.info("call find find_most_violated_constraint and i="
						// + i);
					
						find_most_violated_constraint(ex[i], fycache[i], n, sm,
								sparm);
						
						// logger.info("fydelta in call yeah:"+fydelta_g.toString());

						/* add current fy-fybar and loss to cache */
						if (svm_struct_common.struct_verbosity >= 2)
							rt2 = svm_common.get_runtime();

						// logger.info("add_constraint i="+i);
						// logger.info("fydelta i="+i+" "+fydelta_g.toString());

						viol += add_constraint_to_constraint_cache(
								sm.svm_model, i, 0.0001 * sparm.epsilon / n,
								sparm.ccache_size);

						if (svm_struct_common.struct_verbosity >= 2)
							rt_cacheadd += Math.max(svm_common.get_runtime()
									- rt2, 0);
						viol_est += ccache_g.constlist[i].viol;
						// logger.info("viol_est:"+viol_est+" i="+i);
						uptr++;
					}
					if (j < n) {
						cached_constraint = 1;
					} else {
						cached_constraint = 0;
					}
					if (svm_struct_common.struct_verbosity >= 2)
						rt2 = svm_common.get_runtime();
					if (cached_constraint != 0) {
						// logger.info("cached_constraint is not 0");
						// System.out.println("cached_constraint is not 0");
						viol = find_most_violated_joint_constraint_in_cache(epsilon_est / 2);
						// logger.info(" cached_constraint lhs is :"+lhs.toString()+"  lhs_n is:"+lhs_n.toString());
					} else {
						// logger.info("cached_constraint is  0");
						// System.out.println("cached_constraint is  0");
						viol = find_most_violated_joint_constraint_in_cache(0);
						// logger.info(" no cached_constraint lhs is :"+lhs_g.toString());

					}
					if (svm_struct_common.struct_verbosity >= 2)
						rt_cacheconst += Math.max(svm_common.get_runtime()
								- rt2, 0);
					viol_est *= ((double) n / (double) j);
					logger.info("viol_est=" + viol_est + " slack=" + slack + "");
					epsilon_est = (1 - (double) j / (double) n) * epsilon_est
							+ (double) j / (double) n * (viol_est - slack);// epsilon_est璧嬪�
					logger.info("epsilon_est cal=" + epsilon_est);
					if ((svm_struct_common.struct_verbosity >= 1) && (j != n))
						logger.info("(upd=" + (100.0 * j / n) + ",eps^="
								+ (viol_est - slack) + ",eps*=" + epsilon_est
								+ ")");
				}

				lhsXw = rhs_g - viol;

				rt_total += Math.max(svm_common.get_runtime() - rt1, 0);
			} else {
				/* do not use constraint from cache */
				rt1 = svm_common.get_runtime();
				cached_constraint = 0;
				if (kparm.kernel_type == svm_common.LINEAR)
					svm_common.clear_nvector(lhs_n, sm.sizePsi);
				svm_common.progress_n = 0;
				rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

				for (i = 0; i < n; i++) {
					rt1 = svm_common.get_runtime();

					if (svm_struct_common.struct_verbosity >= 1)
						svm_common.print_percent_progress(n, 10, ".");

					/*
					 * compute most violating fydelta=fy-fybar and rhs for
					 * example i
					 */
					find_most_violated_constraint(ex[i], fycache[i], n, sm,
							sparm);
					/* add current fy-fybar to lhs of constraint */
					if (kparm.kernel_type == svm_common.LINEAR) {
						svm_common.add_list_n_ns(lhs_n, fydelta_g, 1.0);
					} else {
						svm_common.append_svector_list(fydelta_g, lhs_g);

						lhs_g = fydelta_g;
					}
					rhs_g += rhs_i_g; /* add loss to rhs */

					rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

				} /* end of example loop */

				rt1 = svm_common.get_runtime();

				/* create sparse vector from dense sum */
				// logger.info("kernel type is "+kparm.kernel_type);
				System.out.println("kernel type is " + kparm.kernel_type);
				if (kparm.kernel_type == svm_common.LINEAR) {
					// logger.info("kernel type is linear and lhs+n="+lhs_n.toString());
					lhs_g = svm_common.create_svector_n_r(lhs_n, sm.sizePsi,
							null, 1.0,
							svm_struct_common.COMPACT_ROUNDING_THRESH);
					// logger.info("kernel type is linear and lhs="+lhs_g.toString());
				}
				doc = svm_common.create_example(cset.m, 0, 1, 1, lhs_g);
				lhsXw = svm_common.classify_example(svmModel, doc);

				viol = rhs_g - lhsXw;

				rt_total += Math.max(svm_common.get_runtime() - rt1, 0);

			} /* end of finding most violated joint constraint */

			rt1 = svm_common.get_runtime();

			/**** if `error', then add constraint and recompute QP ****/
			if (slack > (rhs_g - lhsXw + 0.000001)) {
				logger.info("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
				logger.info(" set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
				logger.info("slack=" + slack + ", newslack=" + (rhs_g - lhsXw)
						+ "\n");

			}
			ceps = Math.max(0, rhs_g - lhsXw - slack);
			if ((ceps > sparm.epsilon) || cached_constraint != 0) {
				/**** resize constraint matrix and add new constraint ****/
				// cset.lhs=new DOC[cset.m+1];
				int ti = 0;
				ti = cset.m + 1;
				cset.lhs = svm_struct_api.reallocDOCS(cset.lhs, ti);
				// logger.info("lhs:" + lhs_g.toString());
				// logger.info("rhs_g:" + rhs_g);
				cset.lhs[cset.m] = svm_common.create_example(cset.m, 0, 1, 1,
						lhs_g);
				// logger.info("cset m="+cset.m+"  cset.lhs[cset.m] kernel id:"+cset.lhs[cset.m].kernelid);

				// cset.rhs = new double[cset.m + 1];
				cset.rhs = svm_struct_api
						.reallocDoubleArr(cset.rhs, cset.m + 1);
				cset.rhs[cset.m] = rhs_g;
				// alpha = new double[cset.m + 1];
				// logger.info("alpha_g:"+svm_struct_api.douarr2str(alpha_g));
				alpha_g = svm_struct_api.reallocDoubleArr(alpha_g, cset.m + 1);
				alpha_g[cset.m] = 0;
				// alphahist = new int[cset.m + 1];
				// logger.info("alphahist_g:"+svm_struct_api.intarr2str(alphahist_g));
				alphahist_g = svm_struct_api.reallocIntArr(alphahist_g,
						cset.m + 1);
				alphahist_g[cset.m] = optcount;
				// logger.info("optcount here:"+optcount);
				cset.m++;
				totconstraints++;
				if ((alg_type == ONESLACK_DUAL_ALG)
						|| (alg_type == ONESLACK_DUAL_CACHE_ALG)) {
					if (svm_struct_common.struct_verbosity >= 2)
						rt2 = svm_common.get_runtime();
					kparm.gram_matrix = update_kernel_matrix(kparm.gram_matrix,
							cset.m - 1, cset, kparm);
					if (svm_struct_common.struct_verbosity >= 2)
						rt_kernel += Math
								.max(svm_common.get_runtime() - rt2, 0);
				}

				/**** get new QP solution ****/
				if (svm_struct_common.struct_verbosity >= 1) {
					// logger.info("*");
				}
				if (svm_struct_common.struct_verbosity >= 2)
					rt2 = svm_common.get_runtime();
				/*
				 * set svm precision so that higher than eps of most violated
				 * constr
				 */
				if (cached_constraint != 0) {
					epsilon_cached = Math.min(epsilon_cached, ceps);
					lparm.epsilon_crit = epsilon_cached / 2;
				} else {
					epsilon = Math.min(epsilon, ceps); /* best eps so far */
					lparm.epsilon_crit = epsilon / 2;
					epsilon_cached = epsilon;
				}

				svmModel = new MODEL();
				/* Run the QP solver on cset. */
				kernel_type_org = kparm.kernel_type;
				if ((alg_type == svm_struct_common.ONESLACK_DUAL_ALG)
						|| (alg_type == svm_struct_common.ONESLACK_DUAL_CACHE_ALG))
					kparm.kernel_type = svm_common.GRAM; /*
														 * use kernel stored in
														 * kparm
														 */
				// logger.info("svm_learn_struct_joint call svm_learn_optimization in loop");
				sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi,
						lparm, kparm, null, svmModel, alpha_g);
				kparm.kernel_type = (short) kernel_type_org;

				svmModel.kernel_parm.kernel_type = (short) kernel_type_org;
				/*
				 * Always add weight vector, in case part of the kernel is
				 * linear. If not, ignore the weight vector since its content is
				 * bogus.
				 */
				svm_common.add_weight_vector_to_linear_model(svmModel);
				/***
				 * sm.svm_model = svmModel.copyMODEL();
				 ***/
				sm.svm_model = svmModel;
				/*************************
				 * sm.svm_model = svmModel; sm.w = svmModel.lin_weights;
				 **************************/

				sm.w = new double[svmModel.lin_weights.length];
				for (int iw = 0; iw < svmModel.lin_weights.length; iw++) {
					sm.w[iw] = svmModel.lin_weights[iw];
				}

				// logger.info("learned weights:"+svmModel.topWeights());
				/*
				 * for(int kk=0;kk<svmModel.lin_weights.length;kk++) {
				 * if(svmModel.lin_weights[kk]>0) {
				 * logger.info(kk+":"+svmModel.lin_weights[kk]+" "); } }
				 */
				optcount++;
				/*
				 * keep track of when each constraint was last active.
				 * constraints marked with -1 are not updated
				 */
				for (j = 0; j < cset.m; j++) {
					// logger.info("op j="+j+":");
					if ((alphahist_g[j] > -1) && (alpha_g[j] != 0)) {

						alphahist_g[j] = optcount;
						// logger.info("alphahist_g j="+j+":"+alphahist_g[j]);
					}
				}
				if (svm_struct_common.struct_verbosity >= 2)
					rt_opt += Math.max(svm_common.get_runtime() - rt2, 0);

				/*
				 * Check if some of the linear constraints have not been active
				 * in a while. Those constraints are then removed to avoid
				 * bloating the working set beyond necessity.
				 */
				if (svm_struct_common.struct_verbosity >= 3) {
					logger.info("Reducing working set...");
				}

				/************ 在这里要将某些限制去掉 **********/
				// logger.info("cset.m before:"+cset.m);
				remove_inactive_constraints(cset, optcount, 50);
				// logger.info("cset.m after:"+cset.m);
				if (svm_struct_common.struct_verbosity >= 3)
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

			rt_total += Math.max(svm_common.get_runtime() - rt1, 0);
            /*
			if (ceps < 0.1) {
				break;
			}
			*/
			
		} while (cached_constraint != 0
				|| (ceps > sparm.epsilon)
				|| svm_struct_api.finalize_iteration(ceps, cached_constraint,
						sample, sm, cset, alpha_g, sparm));

		if (svm_struct_common.struct_verbosity >= 1) {
			logger.info("Final epsilon on KKT-Conditions: "
					+ (Math.max(svmModel.maxdiff, ceps)) + "\n");

			slack = 0;
			for (j = 0; j < cset.m; j++)
				slack = Math.max(
						slack,
						cset.rhs[j]
								- svm_common.classify_example(svmModel,
										cset.lhs[j]));
			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha_g[i] * cset.rhs[i];
			if (kparm.kernel_type == svm_common.LINEAR)
				modellength = svm_common.model_length_n(svmModel);
			else
				modellength = svm_common.model_length_s(svmModel);
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
			if (svm_struct_common.struct_verbosity >= 2)
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
			else if (svm_struct_common.struct_verbosity == 1)
				logger.info("Runtime in cpu-seconds: " + (rt_total / 100.0)
						+ "\n");
		}

		if (ccache_g != null) {
			long cnum = 0;
			CCACHEELEM celem;
			for (i = 0; i < n; i++)
				for (celem = ccache_g.constlist[i]; celem != null; celem = celem.next)
					cnum++;
			logger.info("Final number of constraints in cache: " + cnum + "\n");
		}

		if (svm_struct_common.struct_verbosity >= 4)
			svm_struct_common.printW(sm.w, sizePsi, n, lparm.svm_c);

		if (svmModel != null) {
			if (svmModel.kernel_parm == null) {
				logger.info("svmModel kernel_parm is null");
			}

			logger.info("sv num there:" + svmModel.sv_num);
			// sm.svm_model = svm_common.copy_model(svmModel);
			sm.svm_model = svmModel;
			sm.w = sm.svm_model.lin_weights; /* short cut to weight vector */

		}
		/*
		 * if(sm.svm_model==null) { logger.info("the svm model is null"); }
		 * if(sm.svm_model.kernel_parm==null) {
		 * logger.info("the svm model kernel_parm is null"); }
		 */

		svm_struct_api.print_struct_learning_stats(sample, sm, cset, alpha_g,
				sparm);

	}

	public void remove_inactive_constraints(CONSTSET cset, int currentiter,
			int mininactive)
	/*
	 * removes the constraints from cset (and alpha) for which alphahist
	 * indicates that they have not been active for at least mininactive
	 * iterations
	 */

	{
		int i, m;
		/*
		 * double[] local_alpha=new double[alpha_g.length]; for(int
		 * lai=0;lai<local_alpha.length;lai++) { local_alpha[lai]=alpha_g[lai];
		 * }
		 * 
		 * int[] local_alphahist=new int[alphahist_g.length]; for(int
		 * lai=0;lai<local_alphahist.length;lai++) {
		 * local_alphahist[lai]=alphahist_g[lai]; }
		 */
		m = 0;
		for (i = 0; i < cset.m; i++) {
			if ((alphahist_g[i] < 0)
					|| ((currentiter - alphahist_g[i]) < mininactive)) {
				/*
				 * keep constraints that are marked as -1 or which have recently
				 * been active
				 */
				cset.lhs[m] = cset.lhs[i];
				cset.lhs[m].docnum = m;
				cset.rhs[m] = cset.rhs[i];
				alpha_g[m] = alpha_g[i];
				alphahist_g[m] = alphahist_g[i];
				// logger.info("m="+m+" i="+i+"  alphahist_g"+alphahist_g[m]+" \n");
				m++;
			} else {
			}
		}

		if (cset.m != m) {
			cset.m = m;
			svm_struct_api.realsmallloc_lhs(cset);
			svm_struct_api.realsmallloc_rhs(cset);

			alpha_g = svm_struct_api.reallocDoubleArr(alpha_g, cset.m);
			alphahist_g = svm_struct_api.reallocIntArr(alphahist_g, cset.m);

		}
	}

	public MATRIX init_kernel_matrix(CONSTSET cset, KERNEL_PARM kparm)
	/*
	 * assigns a kernelid to each constraint in cset and creates the
	 * corresponding kernel matrix.
	 */
	{
		int i, j;
		double kval;
		MATRIX matrix;

		/* assign kernel id to each new constraint */
		for (i = 0; i < cset.m; i++)
			cset.lhs[i].kernelid = i;

		/* allocate kernel matrix as necessary */
		matrix = svm_common.create_matrix(i + 50, i + 50);

		for (j = 0; j < cset.m; j++) {
			for (i = j; i < cset.m; i++) {
				kval = svm_common.kernel(kparm, cset.lhs[j], cset.lhs[i]);
				matrix.element[j][i] = kval;
				matrix.element[i][j] = kval;
			}
		}
		return (matrix);
	}

	public void create_constraint_cache(SAMPLE sample, STRUCT_LEARN_PARM sparm,
			STRUCTMODEL sm)
	/* create new constraint cache for training set */
	{
		int n = sample.n;
		EXAMPLE[] ex = sample.examples;
		// CCACHE ccache;
		int i;
		// logger.info("the train_n is "+n+" \n ");
		ccache_g = new CCACHE();
		ccache_g.n = n;
		ccache_g.sm = sm;
		ccache_g.constlist = new CCACHEELEM[n];
		ccache_g.avg_viol_gain = new double[n];
		ccache_g.changed = new int[n];
		for (i = 0; i < n; i++) {
			/* add constraint for ybar=y to cache */
			ccache_g.constlist[i] = new CCACHEELEM();
			ccache_g.constlist[i].fydelta = svm_common.create_svector_n(null,
					0, null, 1);
			ccache_g.constlist[i].rhs = ssa.loss(ex[i].y, ex[i].y,
					sparm) / n;
			ccache_g.constlist[i].viol = 0;
			ccache_g.constlist[i].next = null;
			ccache_g.avg_viol_gain[i] = 0;
			ccache_g.changed[i] = 0;
		}
		// return(ccache);
	}

	public double compute_violation_of_constraint_in_cache(double thresh)

	/*
	 * computes the violation of the most violated joint constraint in cache.
	 * assumes that update_constraint_cache_for_model has been run.
	 */
	/*
	 * NOTE: This function assumes that loss(y,y')>=0, and it is most efficient
	 * when loss(y,y)=0.
	 */
	{
		// logger.info("begin compute_violation_of_constraint_in_cache");
		double sumviol = 0;
		int i, n = ccache_g.n;

		/**** add all maximal violations ****/
		for (i = 0; i < n; i++) {
			// logger.info("cvocic:"+ccache.constlist[i].fydelta.toString());

			// logger.info("n="+n+"i="+i+"  and viol is "+(ccache_g.constlist[i].viol)+" and thresh="+thresh+"\n");
			if ((ccache_g.constlist[i].viol * n) > thresh) {
				// logger.info("maximal viol i="+i+" viol="+ccache_g.constlist[i].viol);
				sumviol += ccache_g.constlist[i].viol;
			}
		}
		// logger.info("end compute_violation_of_constraint_in_cache");
		return (sumviol);
	}

	/**
	 * lhs_n:瀵逛簬 linear 鐨勬儏鍐�
	 * 
	 * @param ccache
	 * @param thresh
	 * @param lhs_n
	 * @param lhs
	 * @param rhs
	 * @return
	 */
	public double find_most_violated_joint_constraint_in_cache(double thresh)
	/*
	 * constructs most violated joint constraint from cache. assumes that
	 * update_constraint_cache_for_model has been run.
	 */
	/*
	 * NOTE: For kernels, this function returns only a shallow copy of the Psi
	 * vectors in lhs. So, do not use a deep free, otherwise the case becomes
	 * invalid.
	 */
	/*
	 * NOTE: This function assumes that loss(y,y')>=0, and it is most efficient
	 * when loss(y,y)=0.
	 */
	{
		// logger.info("begin find_most_violated_joint_constraint_in_cache");
		double sumviol = 0;
		int i = 0;
		int n = ccache_g.n;
		SVECTOR fydelta;

		lhs_g = null;
		rhs_g = 0;
		if (lhs_n != null) { /* linear case? */
			// logger.info(" use the lhs_n \n");
			svm_common.clear_nvector(lhs_n, ccache_g.sm.sizePsi);
		}

		/**** add all maximally violated fydelta to joint constraint ****/
		for (i = 0; i < n; i++) {
			// logger.info("add all maximally violated fydelta to joint constraint i="+i);
			if ((thresh < 0) || (ccache_g.constlist[i].viol * n > thresh)) {
				/* get most violating fydelta=fy-fybar for example i from cache */
				fydelta = ccache_g.constlist[i].fydelta.copySVECTOR();

				// logger.info("fydelta in fmv :"+fydelta.toString());

				rhs_g += ccache_g.constlist[i].rhs;
				sumviol += ccache_g.constlist[i].viol;
				if (lhs_n != null) { /* linear case? */
					svm_common.add_list_n_ns(lhs_n, fydelta, 1.0); /*
																	 * add
																	 * fy-fybar
																	 * to sum
																	 */
				} else { /* add fy-fybar to vector list */
					fydelta = svm_common.copy_svector(fydelta);
					svm_common.append_svector_list(fydelta, lhs_g);
					lhs_g = fydelta;
				}
			}
		}

		/* create sparse vector from dense sum */
		if (lhs_n != null) /* linear case? */
		{
			lhs_g = svm_common.create_svector_n_r(lhs_n, ccache_g.sm.sizePsi,
					null, 1.0, svm_struct_common.COMPACT_ROUNDING_THRESH);
			// logger.info("lhs in fmv :" );
			// System.out.println("lhs in fmv :" );
			// logger.info("lhs_g word length:"+lhs_g.words.length);
			// logger.info(lhs_g.toString());
		}
		// logger.info("end find_most_violated_joint_constraint_in_cache");
		return (sumviol);
	}

	/**
	 * find_most_violated_constraint(&fydelta,&rhs_i,&ex[i],
	 * fycache[i],n,sm,sparm, &rt_viol,&rt_psi,&argmax_count);
	 * 
	 * @param fydelta
	 * @param rhs
	 * @param ex
	 * @param fycached
	 * @param n
	 * @param sm
	 * @param sparm
	 * @param rt_viol
	 * @param rt_psi
	 * @param argmax_count
	 */

	public void find_most_violated_constraint(EXAMPLE ex, SVECTOR fycached,
			int n, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm)
	/*
	 * returns fydelta=fy-fybar and rhs scalar value that correspond to the most
	 * violated constraint for example ex
	 */
	{
		// logger.info("begin find_most_violated_constraint");
		double rt2 = 0;
		LABEL ybar;
		SVECTOR fybar, fy;
		double factor, lossval;

		// logger.info("in the find_most_violated_constraint");

		if (svm_struct_common.struct_verbosity >= 2)
			rt2 = svm_common.get_runtime();
		argmax_count_g++;
		if (sparm.loss_type == SLACK_RESCALING) {
			// logger.info("call method find_most_violated_constraint_slackrescaling");
			ybar = ssa.find_most_violated_constraint_slackrescaling(
					ex.x, ex.y, sm, sparm);
		} else {
			// logger.info("call method find_most_violated_constraint_marginrescaling");
			ybar =ssa.find_most_violated_constraint_marginrescaling(ex.x, ex.y,
							sm, sparm);
		}
		if (svm_struct_common.struct_verbosity >= 2)
			rt_viol_g += Math.max(svm_common.get_runtime() - rt2, 0);

		if (ssa.empty_label(ybar)) {
			logger.info("ERROR: empty label was returned for example\n");
		}

		/**** get psi(x,y) and psi(x,ybar) ****/
		if (svm_struct_common.struct_verbosity >= 2)
			rt2 = svm_common.get_runtime();
		if (fycached != null)
			fy = svm_common.copy_svector(fycached);
		else
			fy = ssa.psi(ex.x, ex.y, sm, sparm);
		// logger.info("ybar label info:"+ybar.class_index);
		fybar = ssa.psi(ex.x, ybar, sm, sparm);
		// logger.info("fydelta find yeah:"+fybar.toString());
		if (svm_struct_common.struct_verbosity >= 2)
			rt_psi_g += Math.max(svm_common.get_runtime() - rt2, 0);
		lossval = ssa.loss(ex.y, ybar, sparm);

		/**** scale feature vector and margin by loss ****/
		if (sparm.loss_type == SLACK_RESCALING)
			factor = lossval / n;
		else
			/* do not rescale vector for */
			factor = 1.0 / n; /* margin rescaling loss type */
		svm_common.mult_svector_list(fy, factor);
		svm_common.mult_svector_list(fybar, -factor);
		svm_common.append_svector_list(fybar, fy); /* compute fy-fybar */
		/*
		 * String winfo=""; if(fybar!=null) { winfo=""; for(int
		 * l=0;l<fybar.words.length;l++) {
		 * winfo=winfo+fybar.words[l].wnum+":"+fybar.words[l].weight+" "; }
		 * logger.info("ybar info:"+winfo);
		 * 
		 * }
		 */
		fydelta_g = fybar;

		// logger.info("fydelta find year:" + fydelta_g.toString());
		//rhs_i_g = lossval / n;
		//rhs_i_g = lossval / n;
		rhs_i_g=WU.div(lossval, n, 20);
		// logger.info("rhs_i_g:" + rhs_i_g);
		// logger.info("end find_most_violated_constraint");
	}

	/**
	 * 杩欎釜鏂规硶鏈夐敊
	 * 
	 * @param ccache
	 * @param svmModel
	 * @param exnum
	 * @param gainthresh
	 * @param maxconst
	 * @return
	 */
	public double add_constraint_to_constraint_cache(MODEL svmModel, int exnum,
			double gainthresh, int maxconst)
	/*
	 * add new constraint fydelta*w>rhs for example exnum to cache, if it is
	 * more violated (by gainthresh) than the currently most violated constraint
	 * in cache. if this grows the number of cached constraints for this example
	 * beyond maxconst, then the least recently used constraint is deleted. the
	 * function assumes that update_constraint_cache_for_model has been run.
	 */
	{
		// logger.info("add constraint words:"+fydelta.toString());
		double viol, viol_gain, viol_gain_trunc;
		double dist_ydelta;
		DOC doc_fydelta;
		SVECTOR fydelta_new;
		CCACHEELEM celem;
		int cnum;
		double rt2 = 0;

		/* compute violation of new constraint */
		doc_fydelta = svm_common.create_example(1, 0, 1, 1, fydelta_g);
		dist_ydelta = svm_common.classify_example(svmModel, doc_fydelta);

		viol = rhs_i_g - dist_ydelta;
		viol_gain = viol - ccache_g.constlist[exnum].viol;
		viol_gain_trunc = viol - Math.max(ccache_g.constlist[exnum].viol, 0);
		ccache_g.avg_viol_gain[exnum] = viol_gain;

		/*
		 * check if violation of new constraint is larger than that of the best
		 * cache element
		 */
		if (viol_gain > gainthresh) {
			fydelta_new = fydelta_g;
			if (svm_struct_common.struct_verbosity >= 2)
				rt2 = svm_common.get_runtime();
			if (svmModel.kernel_parm.kernel_type == svm_common.LINEAR) {
				// logger.info("svm_struct_common.COMPACT_CACHED_VECTORS="+svm_struct_common.COMPACT_CACHED_VECTORS);
				if (svm_struct_common.COMPACT_CACHED_VECTORS == 1) {
					fydelta_new = svm_common.add_list_sort_ss_r(fydelta_g,
							svm_struct_common.COMPACT_ROUNDING_THRESH);
				} else if (svm_struct_common.COMPACT_CACHED_VECTORS == 2) {
					fydelta_new = svm_common.add_list_ss_r(fydelta_g,
							svm_struct_common.COMPACT_ROUNDING_THRESH);
				} else if (svm_struct_common.COMPACT_CACHED_VECTORS == 3) {
					fydelta_new = svm_common.add_list_ns_r(fydelta_g,
							svm_struct_common.COMPACT_ROUNDING_THRESH);
				}
			}
			if (svm_struct_common.struct_verbosity >= 2)
				rt_cachesum_g += Math.max(svm_common.get_runtime() - rt2, 0);
			celem = ccache_g.constlist[exnum];
			ccache_g.constlist[exnum] = new CCACHEELEM();
			ccache_g.constlist[exnum].next = celem;
			ccache_g.constlist[exnum].fydelta = fydelta_new;
			/***************************
			 * logger.info("exnum="+exnum); logger.info(fydelta_new.toString());
			 ******************/
			// logger.info("ccache.constlist:"+ccache_g.constlist[exnum].fydelta.toString());

			ccache_g.constlist[exnum].rhs = rhs_i_g;

			ccache_g.constlist[exnum].viol = viol;
			// logger.info("check add exnum="+exnum+" viol="+viol);
			ccache_g.changed[exnum] += 2;

			/* remove last constraint in list, if list is longer than maxconst */
			cnum = 2;
			for (celem = ccache_g.constlist[exnum]; (celem != null)
					&& (celem.next != null) && (celem.next.next != null); celem = celem.next)
				cnum++;
			if (cnum > maxconst) {
				celem.next = null;
			}
		} else {

		}
		return (viol_gain_trunc);
	}

	public MATRIX update_kernel_matrix(MATRIX matrix, int newpos,
			CONSTSET cset, KERNEL_PARM kparm)
	/*
	 * assigns new kernelid to constraint in position newpos and fills the
	 * corresponding part of the kernel matrix
	 */
	{
		int i, maxkernelid = 0, newid;
		double kval;
		double[] used;

		/* find free kernelid to assign to new constraint */
		for (i = 0; i < cset.m; i++) {
			// logger.info("i="+i+" newpos:"+newpos);
			/*
			 * if(cset.lhs!=null) {
			 * logger.info("cset.lhs.length:"+cset.lhs.length);
			 * logger.info("cset.lhs[i].kernelid all:"+cset.lhs[i].kernelid); }
			 * else { logger.info("cset.lhs is null"); }
			 */
			if (i != newpos) {
				// logger.info("cset.lhs[i].kernelid:"+cset.lhs[i].kernelid);
				maxkernelid = Math.max(maxkernelid, cset.lhs[i].kernelid);
			}
		}
		used = svm_common.create_nvector(maxkernelid + 2);
		svm_common.clear_nvector(used, maxkernelid + 2);
		for (i = 0; i < cset.m; i++)
			if (i != newpos)
				used[cset.lhs[i].kernelid] = 1;
		for (newid = 0; used[newid] != 0; newid++)
			;
		cset.lhs[newpos].kernelid = newid;

		/* extend kernel matrix if necessary */
		maxkernelid = Math.max(maxkernelid, newid);
		if ((matrix == null) || (maxkernelid >= matrix.m))
			matrix = svm_common.realloc_matrix(matrix, maxkernelid + 50,
					maxkernelid + 50);

		for (i = 0; i < cset.m; i++) {
			kval = svm_common.kernel(kparm, cset.lhs[newpos], cset.lhs[i]);
			matrix.element[newid][cset.lhs[i].kernelid] = kval;
			matrix.element[cset.lhs[i].kernelid][newid] = kval;
		}
		return (matrix);
	}

	public void svm_learn_struct_joint_custom(SAMPLE sample,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm,
			STRUCTMODEL sm) {

	}

	public void update_constraint_cache_for_model(MODEL svmModel)
	{
		int i;
		int progress = 0;
		double maxviol = 0;
		double dist_ydelta;
		DOC doc_fydelta;
		CCACHEELEM celem, prev, maxviol_celem, maxviol_prev;

		doc_fydelta = svm_common.create_example(1, 0, 1, 1, null);
		for (i = 0; i < ccache_g.n; i++) {
			/*** example loop ***/
			svm_common.progress_n = progress;
			if (svm_struct_common.struct_verbosity >= 3)
				svm_common.print_percent_progress(ccache_g.n, 10, "+");

			maxviol = 0;
			prev = null;
			maxviol_celem = null;
			maxviol_prev = null;
			for (celem = ccache_g.constlist[i]; celem != null; celem = celem.next) {
				doc_fydelta.fvec = celem.fydelta.copySVECTOR();
				dist_ydelta = svm_common
						.classify_example(svmModel, doc_fydelta);

				celem.viol = celem.rhs - dist_ydelta;

				// ccache_g.constlist[i].viol=ccache_g.constlist[i].rhs -
				// dist_ydelta;
				if ((celem.viol > maxviol) || (maxviol_celem == null)) {
					maxviol = celem.viol;
					maxviol_celem = celem;
					maxviol_prev = prev;
				}
				prev = celem;
			}

			ccache_g.changed[i] = 0;
			if (maxviol_prev != null) { /*
										 * move max violated constraint to the
										 * top of list
										 */
				maxviol_prev.next = maxviol_celem.next;
				maxviol_celem.next = ccache_g.constlist[i];
				ccache_g.constlist[i] = maxviol_celem;

				ccache_g.changed[i] = 1;
			}
		}
	
	}

}