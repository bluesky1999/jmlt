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
import org.click.classify.svm_struct.data.MostViolStruct;
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
		KERNEL_CACHE kcache = null;
		LABEL ybar;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;


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
			alpha = new double[cset.m];
			alphahist = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha[i] = 0;
				alphahist[i] = -1; // -1 makes sure these constraints are never removed
			}
		}

		//set initial model and slack variables
		svmModel = new MODEL();
		lparm.epsilon_crit = epsilon;
		Learn sl = new Learn();
		if (kparm.kernel_type !=  ModelConstant.LINEAR) {
			kcache = sl.kernel_cache_init(Math.max(cset.m, 1),
					lparm.kernel_cache_size);
		}

		logger.info("sizePsi:" + sizePsi + "   n:" + n);
		sl.svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi + n,
				lparm, kparm, kcache, svmModel, alpha);
		Common.addWeightVectorToLinearModel(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights;

		if ( ModelConstant.USE_FYCACHE != 0) {
			fycache = new SVECTOR[n];
			for (i = 0; i < n; i++) {
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);// temp
																		// point
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					diff = Common.addListSs(fy);
					fy = diff;
				}

			}

		}


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

						if ((use_shrinking == 0) || (opti[i] != opti_round)) {

							// if the example is not shrunk away, then see if it
							// is necessary to add a new constraint					
							argmax_count++;
							if (sparm.loss_type == SLACK_RESCALING) {
								ybar = ssa.findMostViolatedConstraintSlackrescaling(
												ex[i].x, ex[i].y, sm, sparm);
							} else {
								ybar = ssa.findMostViolatedConstraintMarginrescaling(
												ex[i].x, ex[i].y, sm, sparm);
							}


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
							if (fycache != null) {
								fy = Common.copySvector(fycache[i]);
							} else {
								fy = ssa.psi(ex[i].x, ex[i].y, sm,
										sparm);
							}

							fybar = ssa.psi(ex[i].x, ybar, sm, sparm);
;

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
							Common.appendSvectorList(fy, fybar);
							doc = Common.createExample(cset.m, 0, i + 1,
									1, fy);

							// compute slack for this example
							slack = 0;
							for (j = 0; j < cset.m; j++)
								if (cset.lhs[j].slackid == i + 1) {
									if (sparm.slack_norm == 2)
										slack = Math.max(slack,cset.rhs[j]- (Common.classifyExample(svmModel,cset.lhs[j]) - sm.w[sizePsi+ i]/ (Math.sqrt(2 * svmCnorm))));
									else
										slack = Math.max(slack,cset.rhs[j]- Common.classifyExample(svmModel,cset.lhs[j]));
								}

							//if `error' add constraint and recompute 
							dist = Common.classifyExample(svmModel, doc);
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
									diff = Common.addListSs(fy);
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = Common.createExample(cset.m - 1,0,i + 1,1,Common.copySvector(diff));
									else if (sparm.slack_norm == 2) {

										// add squared slack variable to feature vector
										slackv[0].wnum = sizePsi + i;
										slackv[0].weight = 1 / (Math
												.sqrt(2 * svmCnorm));
										slackv[1].wnum = 0; //terminator 
										slackvec = Common.createSvector(slackv, null, 1.0);
										cset.lhs[cset.m - 1] = Common.createExample(cset.m - 1, 0,i + 1, 1,Common.addSs(diff,slackvec));
									}
								} else {//kernel is used 
									if (sparm.slack_norm == 1)
										cset.lhs[cset.m - 1] = Common
												.createExample(cset.m - 1,0,
														i + 1,
														1,
														Common
																.copySvector(fy));
									else if (sparm.slack_norm == 2)
										System.exit(1);
								}
								Common.reallocRhs(cset);
								cset.rhs[cset.m - 1] = margin;

								alpha = Common.reallocAlpha(alpha,
										cset.m);
								alpha[cset.m - 1] = 0;
								alphahist = Common.reallocAlphaList(alphahist, cset.m);
								alphahist[cset.m - 1] = optcount;
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
									svmModel, alpha);
						
							// Always add weight vector, in case part of the
							//kernel is linear. If not, ignore the weight
							// vector since its content is bogus.
							Common.addWeightVectorToLinearModel(svmModel);
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
								if ((alphahist[j] > -1) && (alpha[j] != 0)) {
									alphahist[j] = optcount;
								}
							}

							if ((new_precision != 0)
									&& (epsilon <= sparm.epsilon))
								dont_stop = 1; // make sure we take one final pass
		
							new_precision = 0;
							newconstraints = 0;
						}

					}// exmample loop


					//if (svm_struct_common.struct_verbosity >= 1)
						logger.info("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");
						System.out.println("(NumConst=" + cset.m + ", SV="
								+ (svmModel.sv_num - 1) + ", CEps=" + ceps
								+ ", QPEps=" + svmModel.maxdiff + ")\n");

					if (CommonStruct.struct_verbosity >= 2)
						logger.info("Reducing working set...");

					remove_inactive_constraints(cset, alpha,optcount,alphahist,
							Math.max(50, optcount - lastoptcount));

					lastoptcount = optcount;
					if (CommonStruct.struct_verbosity >= 2)
						logger.info("done. (NumConst=" + cset.m + ")\n");


				} while ((use_shrinking != 0) && (activenum > 0));
			} while (((totconstraints - old_totconstraints) > tolerance)
					|| (dont_stop != 0));

		} while ((epsilon > sparm.epsilon)
				|| ssa.finalizeIteration(ceps, 0, sample, sm, cset,
						alpha, sparm)); // main_loop

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
									- Common.classifyExample(svmModel,
											cset.lhs[j]));
			} else if (sparm.slack_norm == 2) {
				for (j = 0; j < cset.m; j++)
					slacks[cset.lhs[j].slackid] = Math.max(
							slacks[cset.lhs[j].slackid],
							cset.rhs[j]
									- (Common.classifyExample(svmModel,
											cset.lhs[j]) - sm.w[sizePsi
											+ cset.lhs[j].slackid - 1]
											/ (Math.sqrt(2 * svmCnorm))));
			}
			slacksum = 0;
			for (i = 1; i <= n; i++)
				slacksum += slacks[i];

			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha[i] * cset.rhs[i];
			modellength = Common.modelLengthS(svmModel);
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

		ssa.printStructLearningStats(sample, sm, cset, alpha,
				sparm);

	}

	public void svm_learn_struct_joint(SAMPLE sample, STRUCT_LEARN_PARM sparm,
			LEARN_PARM lparm, KERNEL_PARM kparm, STRUCTMODEL sm, int alg_type) {
		int i, j;
		int numIt = 0;
		int totconstraints = 0;
		int kernel_type_org;
		double epsilon, epsilon_cached;
		double lhsXw;
		MostViolStruct violStruct=new MostViolStruct();
		 double rhs=0;
		double slack, ceps;
		double dualitygap, modellength, alphasum;
		int sizePsi;
		double[] alpha = null;
		 int[] alphahist = null;
		int optcount = 0;

		CONSTSET cset;
		SVECTOR diff = null;
		 double[] lhs_n=null;
		SVECTOR fy;
		SVECTOR[] fycache = null;
		 SVECTOR lhs;
		MODEL svmModel = null;
		DOC doc;

		int n = sample.n;
		EXAMPLE[] ex = sample.examples;
		int progress;


		int cached_constraint;
		double viol=0, viol_est, epsilon_est = 0;
		int uptr = 0;
		int[] randmapping = null;
		int batch_size = n;

		Learn sl = new Learn();

		if (sparm.batch_size < 100)
			batch_size = (int) ((sparm.batch_size * n) / 100.0);
		ssa.initStructModel(sample, sm, sparm, lparm, kparm);
		sizePsi = sm.sizePsi + 1; // sm must contain size of psi on return 

		if (sparm.slack_norm == 1) {
			lparm.svm_c = sparm.C; // set upper bound C
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
			alpha = new double[cset.m];
			alphahist = new int[cset.m];
			for (i = 0; i < cset.m; i++) {
				alpha[i] = 0;
				alphahist[i] = -1;
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
				kparm, null, svmModel, alpha);

		// logger.info("sl totwords:"+svmModel.totwords);
		Common.addWeightVectorToLinearModel(svmModel);
		sm.svm_model = svmModel;
		sm.w = svmModel.lin_weights; // short cut to weight vector

		// create a cache of the feature vectors for the correct labels
		fycache = new SVECTOR[n];
		for (i = 0; i < n; i++) {
			if ( ModelConstant.USE_FYCACHE != 0) {
				// logger.info("USE THE FYCACHE \n");
				fy = ssa.psi(ex[i].x, ex[i].y, sm, sparm);
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					diff = Common.addListSortSsR(fy,
							CommonStruct.COMPACT_ROUNDING_THRESH);
					fy = diff;
				}
			} else
				fy = null;
			fycache[i] = fy;
		}


		if (kparm.kernel_type ==  ModelConstant.LINEAR) {
			// logger.info("kernel type is LINEAR \n");
			lhs_n = Common.createNvector(sm.sizePsi);
		}
		// randomize order or training examples 
		if (batch_size < n)
			randmapping = Common.randomOrder(n);

	
		//=== main loop=======
		do { // iteratively find and add constraints to working set 

			if (CommonStruct.struct_verbosity >= 1) {
				logger.info("in loop Iter " + (++numIt) + ": ");
				System.out.println("in loop Iter " + (numIt) + ": ");
			}
			
        	// compute current slack 
			alphasum = 0;
			for (j = 0; (j < cset.m); j++)
			{
				alphasum += alpha[j];
			}
			for (j = 0, slack = -1; (j < cset.m) && (slack == -1); j++) {
				if (alpha[j] > alphasum / cset.m) {
					slack = Math.max(0,cset.rhs[j]- Common.classifyExample(svmModel,cset.lhs[j]));
				}
			}
			slack = Math.max(0, slack);
			//logge.info("slack val is :" + slack);

			// find a violated joint constraint 
			lhs = null;
			rhs = 0;
			
		
			if (alg_type == ONESLACK_DUAL_CACHE_ALG) {

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
					cached_constraint = 1;
				} else {
					logger.info("There is no violated constraint in cache");
	
					// There is no sufficiently violated constraint in cache, so
					// update cache byprint_percent_progress computing most
					// violated constraint explicitly for batch_size examples
					viol_est = 0;
					progress = 0;
					Common.progress_n = progress;

					for (j = 0; (j < batch_size)
							|| ((j < n) && (viol - slack < sparm.epsilon)); j++) {
						if (CommonStruct.struct_verbosity >= 1)
							Common.printPercentProgress(n, 10, ".");
						uptr = uptr % n;
						if (randmapping != null)
							i = randmapping[uptr];
						else
							i = uptr;
			
						//find most violating fydelta=fy-fybar and rhs for example i	
						find_most_violated_constraint(ex[i], fycache[i], n, sm,
								sparm,violStruct);
						

						uptr++;
					}
					if (j < n) {
						cached_constraint = 1;
					} else {
						cached_constraint = 0;
					}

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

				lhsXw = rhs - viol;

			} else {
				// do not use constraint from cache

				cached_constraint = 0;
				if (kparm.kernel_type ==  ModelConstant.LINEAR)
					Common.clearNvector(lhs_n, sm.sizePsi);
				Common.progress_n = 0;

				for (i = 0; i < n; i++) {

					if (CommonStruct.struct_verbosity >= 1)
						Common.printPercentProgress(n, 10, ".");


					// compute most violating fydelta=fy-fybar and rhs for example i
					find_most_violated_constraint(ex[i], fycache[i], n, sm,
							sparm,violStruct);
					// add current fy-fybar to lhs of constraint
					if (kparm.kernel_type ==  ModelConstant.LINEAR) {
						Common.addListNNS(lhs_n, violStruct.fydelta, 1.0);
					} else {
						Common.appendSvectorList(violStruct.fydelta, lhs);
						lhs=violStruct.fydelta;
					}
					rhs +=violStruct.rhs;// add loss to rhs 
				} // end of example loop 

				// create sparse vector from dense sum 
				System.out.println("kernel type is " + kparm.kernel_type);
				if (kparm.kernel_type ==  ModelConstant.LINEAR) {
					lhs = Common.createSvectorNR(lhs_n, sm.sizePsi,
							null, 1.0,
							CommonStruct.COMPACT_ROUNDING_THRESH);
				}
				doc = Common.createExample(cset.m, 0, 1, 1, lhs);
				lhsXw = Common.classifyExample(svmModel, doc);

				viol = rhs - lhsXw;

			} // end of finding most violated joint constraint

			// if `error', then add constraint and recompute QP 
			if (slack > (rhs - lhsXw + 0.000001)) {
				logger.info("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
				logger.info(" set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
				logger.info("slack=" + slack + ", newslack=" + (rhs - lhsXw)
						+ "\n");

			}
			ceps = Math.max(0, rhs - lhsXw - slack);
			if ((ceps > sparm.epsilon) || cached_constraint != 0) {
				// resize constraint matrix and add new constraint
				// cset.lhs=new DOC[cset.m+1];
				int ti = 0;
				ti = cset.m + 1;
				cset.lhs = Common.reallocDOCS(cset.lhs, ti);

				cset.lhs[cset.m] = Common.createExample(cset.m, 0, 1, 1,
						lhs);

				// cset.rhs = new double[cset.m + 1];
				cset.rhs = Common
						.reallocDoubleArr(cset.rhs, cset.m + 1);
				cset.rhs[cset.m] = rhs;
				// alpha = new double[cset.m + 1];
				alpha = Common.reallocDoubleArr(alpha, cset.m + 1);
				alpha[cset.m] = 0;
				// alphahist = new int[cset.m + 1];
				alphahist = Common.reallocIntArr(alphahist,
						cset.m + 1);
				alphahist[cset.m] = optcount;
				cset.m++;
				totconstraints++;
				if ((alg_type == ONESLACK_DUAL_ALG)
						|| (alg_type == ONESLACK_DUAL_CACHE_ALG)) {
					kparm.gram_matrix = update_kernel_matrix(kparm.gram_matrix,
							cset.m - 1, cset, kparm);
				}

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
						lparm, kparm, null, svmModel, alpha);
				kparm.kernel_type = (short) kernel_type_org;

				svmModel.kernel_parm.kernel_type = (short) kernel_type_org;

				// Always add weight vector, in case part of the kernel is
				// linear. If not, ignore the weight vector since its content is bogus.
				Common.addWeightVectorToLinearModel(svmModel);

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
					if ((alphahist[j] > -1) && (alpha[j] != 0)) {
						alphahist[j] = optcount;
					}
				}

				//Check if some of the linear constraints have not been active
				// in a while. Those constraints are then removed to avoid
				// bloating the working set beyond necessity.
				if (CommonStruct.struct_verbosity >= 3) {
					logger.info("Reducing working set...");
				}

			    // 在这里要将某些限制去掉
				// logger.info("cset.m before:"+cset.m);
				remove_inactive_constraints(cset,alpha, optcount,alphahist, 50);
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

			
		} while (cached_constraint != 0
				|| (ceps > sparm.epsilon)
				|| ssa.finalizeIteration(ceps, cached_constraint,
						sample, sm, cset, alpha, sparm));

		if (CommonStruct.struct_verbosity >= 1) {
			logger.info("Final epsilon on KKT-Conditions: "
					+ (Math.max(svmModel.maxdiff, ceps)) + "\n");

			slack = 0;
			for (j = 0; j < cset.m; j++)
				slack = Math.max(
						slack,
						cset.rhs[j]
								- Common.classifyExample(svmModel,
										cset.lhs[j]));
			alphasum = 0;
			for (i = 0; i < cset.m; i++)
				alphasum += alpha[i] * cset.rhs[i];
			if (kparm.kernel_type ==  ModelConstant.LINEAR)
				modellength = Common.modelLengthN(svmModel);
			else
				modellength = Common.modelLengthS(svmModel);
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
					+ violStruct.argmaxCount + "\n");
			logger.info("Number of SV: " + (svmModel.sv_num - 1) + " \n");
			logger.info("Norm of weight vector: |w|=" + modellength + "\n");
			logger.info("Value of slack variable (on working set): xi=" + slack
					+ "\n");
			logger.info("Value of slack variable (global): xi=" + viol + "\n");
			logger.info("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||="
					+ (sl.length_of_longest_document_vector(cset.lhs, cset.m,
							kparm)) + "\n");
			
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


		ssa.printStructLearningStats(sample, sm, cset, alpha,
				sparm);

	}

	/**
	 * removes the constraints from cset (and alpha) for which alphahist
	 * indicates that they have not been active for at least mininactive
	 * iterations
	 */
	public void remove_inactive_constraints(CONSTSET cset, double[] alpha,int currentiter,int[] alphahist,
			int mininactive)
	{
		int i, m;

		m = 0;
		for (i = 0; i < cset.m; i++) {
			if ((alphahist[i] < 0)
					|| ((currentiter - alphahist[i]) < mininactive)) {
			
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
			Common.realSmalllocLhs(cset);
			Common.realSmalllocRhs(cset);

			alpha = Common.reallocDoubleArr(alpha, cset.m);
			alphahist = Common.reallocIntArr(alphahist, cset.m);

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
		matrix = Common.createMatrix(i + 50, i + 50);

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
			int n, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm,MostViolStruct struct)
	{
		LABEL ybar;
		SVECTOR fybar, fy;
		double factor, lossval;


		struct.argmaxCount++;
		if (sparm.loss_type == SLACK_RESCALING) {
			ybar = ssa.findMostViolatedConstraintSlackrescaling(
					ex.x, ex.y, sm, sparm);
		} else {
			ybar =ssa.findMostViolatedConstraintMarginrescaling(ex.x, ex.y,
							sm, sparm);
		}

		if (ssa.emptyLabel(ybar)) {
			logger.info("ERROR: empty label was returned for example\n");
		}

		// get psi(x,y) and psi(x,ybar)
		if (fycached != null)
			fy = Common.copySvector(fycached);
		else
			fy = ssa.psi(ex.x, ex.y, sm, sparm);
		fybar = ssa.psi(ex.x, ybar, sm, sparm);

		lossval = ssa.loss(ex.y, ybar, sparm);

		// scale feature vector and margin by loss 
		if (sparm.loss_type == SLACK_RESCALING)
			factor = lossval / n;
		else
			factor = 1.0 / n; // do not rescale vector formargin rescaling loss type 
		Common.multSvectorList(fy, factor);
		Common.multSvectorList(fybar, -factor);
		Common.appendSvectorList(fybar, fy); // compute fy-fybar 

        struct.fydelta=fybar;
        struct.rhs=WU.div(lossval, n, 20);
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
		used = Common.createNvector(maxkernelid + 2);
		Common.clearNvector(used, maxkernelid + 2);
		for (i = 0; i < cset.m; i++)
			if (i != newpos)
				used[cset.lhs[i].kernelid] = 1;
		for (newid = 0; used[newid] != 0; newid++)
			;
		cset.lhs[newpos].kernelid = newid;

		// extend kernel matrix if necessary
		maxkernelid = Math.max(maxkernelid, newid);
		if ((matrix == null) || (maxkernelid >= matrix.m))
			matrix = Common.reallocMatrix(matrix, maxkernelid + 50,
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