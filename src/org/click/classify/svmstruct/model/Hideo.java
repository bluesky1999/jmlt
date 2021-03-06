package org.click.classify.svmstruct.model;

import java.io.FileWriter;
import java.io.PrintWriter;

import org.click.classify.svmstruct.data.LEARN_PARM;
import org.click.classify.svmstruct.data.QP;

import org.click.lib.math.SimFunc;

/**
 * The Hildreth and D'Espo solver specialized for SVMs. <br>
 * 解决下面的二次规划问题 minimize g0 * x + 1/2 x' * G * x s.t. ce*x - ce0 = 0 l <= x <= u <br>
 * ce 的元素只能取值 -1 或 1
 * 
 * @author lq
 */

public class Hideo {

	public static final int PRIMAL_OPTIMAL = 1;
	public static final int DUAL_OPTIMAL = 2;
	public static final int MAXITER_EXCEEDED = 3;
	public static final int NAN_SOLUTION = 4;
	public static final int ONLY_ONE_VARIABLE = 5;

	public static final int LARGEROUND = 0;
	public static final int SMALLROUND = 1;

	public static final double DEF_PRECISION = 1E-5;
	public static final int DEF_MAX_ITERATIONS = 200;
	public static final double DEF_LINDEP_SENSITIVITY = 1E-8;
	public static final double EPSILON_HIDEO = 1E-20;
	public static final double EPSILON_EQ = 1E-5;

	public double[] primal = null;

	public double[] dual = null;

	public long precision_violations = 0;
	public double opt_precision = DEF_PRECISION;
	public int maxiter = DEF_MAX_ITERATIONS;
	public double lindep_sensitivity = DEF_LINDEP_SENSITIVITY;
	public double[] buffer = null;
	public int[] nonoptimal = null;

	public static int smallroundcount = 0;

	public static int roundnumber = 0;

	public short verbosity = 0;

	public double progress;
	// private static Logger logger = Logger.getLogger(Hideo.class);

	public SimFunc<Integer> sf = new SimFunc<Integer>();

	public Hideo() {

	}

	/**
	 * the rank of g : &nbsp; 0 1 2 3 4 5 6 7 8 <br>
	 * &nbsp; <=> <br>
	 * &nbsp; 0 1 2 <br>
	 * &nbsp; 3 4 5 <br>
	 * &nbsp; 6 7 8 <br>
	 * 
	 * @param qp
	 * @param epsilon_crit
	 * @param nx
	 * @param threshold
	 * @param learn_param
	 * @return
	 */
	public double[] optimizeQp(QP qp, double epsilon_crit, int nx, double threshold, LEARN_PARM learn_param) {

		// System.err.println("epsilon_crit:"+epsilon_crit+" threshold:"+threshold);
		int i, j;
		int result;
		double eq;
		// svm_common.verbosity=5;
		roundnumber++;

		if (primal == null) {
			primal = new double[nx];

			// the dual variable includes 2*n dual variables for box constraints
			// and 2*1 dual variables for equality constraint
			dual = new double[2 * (nx + 1)];
			nonoptimal = new int[nx];
			buffer = new double[(nx + 1) * 2 * (nx + 1) * 2 + nx * nx + 2 * (nx + 1) * 2 + 2 * nx + 1 + 2 * nx + nx + nx + nx * nx];
			threshold = 0;
			for (i = 0; i < nx; i++) {
				primal[i] = 0;
			}
		}

		result = optimizeHildrethDespo(qp.opt_n, qp.opt_m, opt_precision, epsilon_crit, learn_param.epsilon_a, maxiter, 0, 0, lindep_sensitivity, qp.opt_g, qp.opt_g0, qp.opt_ce, qp.opt_ce0, qp.opt_low, qp.opt_up, primal, qp.opt_xinit, dual, nonoptimal, buffer);

		if (learn_param.totwords < learn_param.svm_maxqpsize) {

			learn_param.svm_maxqpsize = sf.max(learn_param.totwords, 2);

		}

		if (result == NAN_SOLUTION) {

			lindep_sensitivity *= 2;
			if (learn_param.svm_maxqpsize > 2) {
				learn_param.svm_maxqpsize--;
			}
			precision_violations++;
		}
		
	
		if ((result != PRIMAL_OPTIMAL) || (roundnumber % 31 == 0) || (progress <= 0)) {
			// System.err.println("result is not PRIMAL_OPTIMAL");
			smallroundcount++;

			//System.err.println("SMALLROUND:"+SMALLROUND);
			result = optimizeHildrethDespo(qp.opt_n, qp.opt_m, opt_precision, epsilon_crit, learn_param.epsilon_a, maxiter, PRIMAL_OPTIMAL, SMALLROUND, lindep_sensitivity, qp.opt_g, qp.opt_g0, qp.opt_ce, qp.opt_ce0, qp.opt_low, qp.opt_up, primal, qp.opt_xinit, dual, nonoptimal, buffer);

			if (result != PRIMAL_OPTIMAL) {
				if (result != ONLY_ONE_VARIABLE) {
					precision_violations++;
				}

				if (result == MAXITER_EXCEEDED) {
					maxiter += 100;
				}

				if (result == NAN_SOLUTION) {
					lindep_sensitivity *= 2;

					for (i = 0; i < qp.opt_n; i++) {
						primal[i] = qp.opt_xinit[i];
					}
				}
			}
		}

		if (precision_violations > 50) {
			precision_violations = 0;
			epsilon_crit = 10.0;
		}

		if ((qp.opt_m > 0) && (result != NAN_SOLUTION) && (!(Double.isNaN(dual[1] - dual[0])))) {
			threshold = dual[1] - dual[0];
		} else {
			threshold = 0;
		}

		return primal;
	}

	public int optimizeHildrethDespo(int n, int m, double precision, double epsilon_crit, double epsilon_a, int maxiter, int goal, int smallround, double lindep_sensitivity, double[] g, double[] g0, double[] ce, double[] ce0, double[] low, double[] up, double[] primal, double[] init, double[] dual, int[] lin_dependent, double[] buffer) {

		int i, j, k, from, to, n_indep;
		int changed;
		double sum, bmin = 0, bmax = 0;
		double[] d, d0, ig, dual_old, temp, start;
		double[] g0_new, g_new, ce_new, ce0_new, low_new, up_new;
		double add;
		//double t;
		int result;

		double obj_before, obj_after;
		int b1, b2;
		//double g0_b1 = 0, g0_b2 = 0, ce0_b;

		g0_new = new double[n];

		// d stands for d=B*ig*BT(what is B here?)
		d = new double[(n + m) * 2 * (n + m) * 2];

		// d0 stands for d0=c+B*ig*b,what is b and c
		// b stands for opt_g0
		// c stands for inequality right part that is B*x<=c
		d0 = new double[(n + m) * 2];
		ce_new = new double[n];
		ce0_new = new double[m];
		ig = new double[n * n];
		dual_old = new double[(n + m) * 2];
		low_new = new double[n];
		up_new = new double[n];
		start = new double[n];
		g_new = new double[n * n];
		temp = new double[n];

		b1 = -1;
		b2 = -1;

		for (i = 0; i < n; i++) {

			//note:sum=[g0[i]+(primal*g)]*ce[i]
			sum = g0[i];
			for (j = 0; j < n; j++) {
				sum += init[j] * g[i * n + j];
			}
			sum = sum * ce[i];

			if (((b1 == -1) || (sum < bmin)) && (!((init[i] <= (low[i] + epsilon_a)) && (ce[i] < 0.0))) && (!((init[i] >= (up[i] - epsilon_a)) && (ce[i] > 0.0)))) {
				bmin = sum;
				b1 = i;
			}
			if (((b2 == -1) || (sum >= bmax)) && (!((init[i] <= (low[i] + epsilon_a)) && (ce[i] > 0.0))) && (!((init[i] >= (up[i] - epsilon_a)) && (ce[i] < 0.0)))) {
				bmax = sum;
				b2 = i;
			}
		}

		if ((b1 == -1) || (b2 == -1)) {
			b1 = sf.max(b1, b2);
			b2 = sf.max(b1, b2);
		}

		for (i = 0; i < n; i++) {
			start[i] = init[i];
		}

		add = 0;
		changed = 0;

		//changed=maxMinDir(n, m, b1, b2, g, g0, start, ce, ce0, up, low, init);

		if ((m > 0) && ((Math.abs(g[b1 * n + b1]) < lindep_sensitivity) || (Math.abs(g[b2 * n + b2]) < lindep_sensitivity))) {

			add += 0.093274;
		}
		// in case both examples are linear dependent
		else if ((m > 0) && (g[b1 * n + b2] != 0 && g[b2 * n + b2] != 0) && (Math.abs(g[b1 * n + b1] / g[b1 * n + b2] - g[b1 * n + b2] / g[b2 * n + b2]) < lindep_sensitivity)) {

			add += 0.078274;
		}

	
		//linDepDir(n,m,b1,b2,g,g0,lindep_sensitivity,up,low,start);

		System.arraycopy(g, 0, d, 0, n * n);

		if ((m == 1) && (add > 0.0)) {
			for (j = 0; j < n; j++) {
				for (k = 0; k < n; k++) {
					d[j * n + k] += add * ce[j] * ce[k];
				}
			}
		} else {
			add = 0.0;
		}

		if (n > 2) { // switch, so that variables are better mixed
			switchrkMatrix(d, n, b1, 0);
			if (b2 == 0) {
				switchrkMatrix(d, n, b1, 1);
			} else {
				switchrkMatrix(d, n, b2, 1);
			}
		}

		//System.err.println("smallround:"+smallround);
		if (smallround == SMALLROUND) {
			for (i = 2; i < n; i++) {
				lin_dependent[i] = 1;
			}
			if (m > 0) { // for biased hyperplane, pick two variables
				lin_dependent[0] = 0;
				lin_dependent[1] = 0;
			} else { // for unbiased hyperplane, pick only one variable
				lin_dependent[0] = smallroundcount % 2;
				lin_dependent[1] = (smallroundcount + 1) % 2;
			}
		} else {
			for (i = 0; i < n; i++) {
				lin_dependent[i] = 0;
			}
		}

		invertMatrix(d, n, ig, lindep_sensitivity, lin_dependent);

		if (n > 2) { // now switch back
			if (b2 == 0) {
				switchrkMatrix(ig, n, b1, 1);
				i = lin_dependent[1];
				lin_dependent[1] = lin_dependent[b1];
				lin_dependent[b1] = i;
			} else {
				switchrkMatrix(ig, n, b2, 1);
				i = lin_dependent[1];
				lin_dependent[1] = lin_dependent[b2];
				lin_dependent[b2] = i;
			}
			switchrkMatrix(ig, n, b1, 0);
			i = lin_dependent[0];
			lin_dependent[0] = lin_dependent[b1];
			lin_dependent[b1] = i;
		}

		// copyMatrix(g, n, g_new); // restore g_new matrix

		System.arraycopy(g, 0, g_new, 0, n * n);
		if (add > 0)
			for (j = 0; j < n; j++) {
				for (k = 0; k < n; k++) {
					g_new[j * n + k] += add * ce[j] * ce[k];
				}
			}

		for (i = 0; i < n; i++) { // fix linear dependent vectors

			g0_new[i] = g0[i] + add * ce0[0] * ce[i];
		}
		if (m > 0)
			ce0_new[0] = -ce0[0];
		for (i = 0; i < n; i++) { // fix linear dependent vectors
			if (lin_dependent[i] > 0) {
				for (j = 0; j < n; j++) {
					if (lin_dependent[j] == 0) {

						g0_new[j] += start[i] * g_new[i * n + j];
					}
				}
				if (m > 0)
					ce0_new[0] -= (start[i] * ce[i]);
			}
		}

		from = 0; // remove linear dependent vectors
		to = 0;
		n_indep = 0;
		for (i = 0; i < n; i++) {
			if (lin_dependent[i] == 0) {
				g0_new[n_indep] = g0_new[i];
				ce_new[n_indep] = ce[i];
				low_new[n_indep] = low[i];
				up_new[n_indep] = up[i];
				primal[n_indep] = start[i];
				n_indep++;
			}
			for (j = 0; j < n; j++) {
				if ((lin_dependent[i] == 0) && (lin_dependent[j] == 0)) {
					ig[to] = ig[from];
					g_new[to] = g_new[from];
					to++;
				}
				from++;
			}
		}

		// cannot optimize with only one variable
		if ((n_indep <= 1) && (m > 0) && (changed == 0)) {
			for (i = n - 1; i >= 0; i--) {
				primal[i] = init[i];
			}
			return ((int) ONLY_ONE_VARIABLE);
		}

		if ((changed == 0) || (n_indep > 1)) {

			result = solveDual(n_indep, m, precision, epsilon_crit, maxiter, g_new, g0_new, ce_new, ce0_new, low_new, up_new, primal, d, d0, ig, dual, dual_old, temp, goal);
		} else {
			result = PRIMAL_OPTIMAL;
		}

		j = n_indep;
		for (i = n - 1; i >= 0; i--) {
			if (lin_dependent[i] == 0) {
				j--;
				primal[i] = primal[j];
			} else {
				//System.err.println("leave as it is");
				primal[i] = start[i]; // leave as is
			}
			temp[i] = primal[i];
		}

		obj_before = calculateQpObjective(n, g, g0, init);
		obj_after = calculateQpObjective(n, g, g0, primal);
		progress = obj_before - obj_after;
		verbosity = 5;

		return ((int) result);
	}

	public int maxMinDir(int n, int m, int b1, int b2, double[] g, double[] g0, double[] start, double[] ce, double[] ce0, double[] up, double[] low, double[] init) {

		int i, j, changed=0;
		double g0_b1 = 0, g0_b2 = 0, ce0_b, t;

		if ((b1 != b2) && (m == 1)) {
			for (i = 0; i < n; i++) { // fix other vectors
				if (i == b1)
					g0_b1 = g0[i];
				if (i == b2)
					g0_b2 = g0[i];
			}
			ce0_b = ce0[0];
			for (i = 0; i < n; i++) {
				if ((i != b1) && (i != b2)) {
					for (j = 0; j < n; j++) {
						if (j == b1)
							g0_b1 += start[i] * g[i * n + j];
						if (j == b2)
							g0_b2 += start[i] * g[i * n + j];
					}
					ce0_b -= (start[i] * ce[i]);
				}
			}
			if ((g[b1 * n + b2] == g[b1 * n + b1]) && (g[b1 * n + b2] == g[b2 * n + b2])) {

				System.err.println("ce[b1]=" + ce[b1] + "  ce[b2]=" + ce[b2]);
				if (ce[b1] == ce[b2]) {
					if (g0_b1 <= g0_b2) { // set b1 to upper bound

						changed = 1;
						t = up[b1] - init[b1];
						if ((init[b2] - low[b2]) < t) {
							t = init[b2] - low[b2];
						}
						start[b1] = init[b1] + t;
						start[b2] = init[b2] - t;
					} else if (g0_b1 > g0_b2) { // set b2 to upper bound

						changed = 1;
						t = up[b2] - init[b2];
						if ((init[b1] - low[b1]) < t) {
							t = init[b1] - low[b1];
						}
						start[b1] = init[b1] - t;
						start[b2] = init[b2] + t;
					}
				} else if (((g[b1 * n + b1] > 0) || (g[b2 * n + b2] > 0))) {
					t = ((ce[b2] / ce[b1]) * g0[b1] - g0[b2] + ce0[0] * (g[b1 * n + b1] * ce[b2] / ce[b1] - g[b1 * n + b2] / ce[b1])) / ((ce[b2] * ce[b2] / (ce[b1] * ce[b1])) * g[b1 * n + b1] + g[b2 * n + b2] - 2 * (g[b1 * n + b2] * ce[b2] / ce[b1])) - init[b2];
					changed = 1;
					if ((up[b2] - init[b2]) < t) {
						t = up[b2] - init[b2];
					}
					if ((init[b2] - low[b2]) < -t) {
						t = -(init[b2] - low[b2]);
					}
					if ((up[b1] - init[b1]) < t) {
						t = (up[b1] - init[b1]);
					}
					if ((init[b1] - low[b1]) < -t) {
						t = -(init[b1] - low[b1]);
					}
					start[b1] = init[b1] + t;
					start[b2] = init[b2] + t;
				}
			}
			if ((-g[b1 * n + b2] == g[b1 * n + b1]) && (-g[b1 * n + b2] == g[b2 * n + b2])) {

				if (ce[b1] != ce[b2]) {
					if ((g0_b1 + g0_b2) < 0) { // set b1 and b2 to upper bound

						changed = 1;
						t = up[b1] - init[b1];
						if ((up[b2] - init[b2]) < t) {
							t = up[b2] - init[b2];
						}
						start[b1] = init[b1] + t;
						start[b2] = init[b2] + t;
					} else if ((g0_b1 + g0_b2) >= 0) {
						changed = 1;
						t = init[b1] - low[b1];
						if ((init[b2] - low[b2]) < t) {
							t = init[b2] - low[b2];
						}
						start[b1] = init[b1] - t;
						start[b2] = init[b2] - t;
					}
				} else if (((g[b1 * n + b1] > 0) || (g[b2 * n + b2] > 0))) {

					t = ((ce[b2] / ce[b1]) * g0[b1] - g0[b2] + ce0[0] * (g[b1 * n + b1] * ce[b2] / ce[b1] - g[b1 * n + b2] / ce[b1])) / ((ce[b2] * ce[b2] / (ce[b1] * ce[b1])) * g[b1 * n + b1] + g[b2 * n + b2] - 2 * (g[b1 * n + b2] * ce[b2] / ce[b1])) - init[b2];
					changed = 1;
					if ((up[b2] - init[b2]) < t) {
						t = up[b2] - init[b2];
					}
					if ((init[b2] - low[b2]) < -t) {
						t = -(init[b2] - low[b2]);
					}
					if ((up[b1] - init[b1]) < -t) {
						t = -(up[b1] - init[b1]);
					}
					if ((init[b1] - low[b1]) < t) {
						t = init[b1] - low[b1];
					}
					start[b1] = init[b1] - t;
					start[b2] = init[b2] + t;
				}
			}
		}
		
		return changed;
	}

	/**
	 * special case for zero diagonal entry on unbiased hyperplane
	 * @param n
	 * @param m
	 * @param b1
	 * @param b2
	 * @param g
	 * @param g0
	 * @param lindep_sensitivity
	 * @param up
	 * @param low
	 * @param start
	 */
	public void linDepDir(int n,int m,int b1,int b2,double[] g,double[] g0,double lindep_sensitivity,double[] up,double[] low,double[] start)
	{
		int i,j;
		double g0_b1 = 0, g0_b2 = 0, ce0_b;
		
		if ((m == 0) && (b1 >= 0)) {
			if (Math.abs(g[b1 * n + b1]) < lindep_sensitivity) {

				for (i = 0; i < n; i++) { // fix other vectors
					if (i == b1)
						g0_b1 = g0[i];
				}
				for (i = 0; i < n; i++) {
					if (i != b1) {
						for (j = 0; j < n; j++) {
							if (j == b1)
								g0_b1 += start[i] * g[i * n + j];
						}
					}
				}
				if (g0_b1 < 0)
					start[b1] = up[b1];
				if (g0_b1 >= 0)
					start[b1] = low[b1];
			}
		}

		if ((m == 0) && (b2 >= 0)) {
			if (Math.abs(g[b2 * n + b2]) < lindep_sensitivity) {

				for (i = 0; i < n; i++) { // fix other vectors
					if (i == b2)
						g0_b2 = g0[i];
				}
				for (i = 0; i < n; i++) {
					if (i != b2) {
						for (j = 0; j < n; j++) {
							if (j == b2)
								g0_b2 += start[i] * g[i * n + j];
						}
					}
				}
				if (g0_b2 < 0)
					start[b2] = up[b2];
				if (g0_b2 >= 0)
					start[b2] = low[b2];
			}
		}
	}
	
	private class DualCheck {

		int retrain;

		double maxviol;

		int maxfaktor;

		int primal_optimal;

		double precision;

	}

	public int solveDual(int n, int m, double precision, double epsilon_crit, int maxiter, double[] g, double[] g0, double[] ce, double[] ce0, double[] low, double[] up, double[] primal, double[] d, double[] d0, double[] ig, double[] dual, double[] dual_old, double[] temp, int goal) {
		int i, j, k, iter;
		// double maxviol;
		double sum, w, viol, temp1, temp2, isnantest;
		double model_b, dist;
		// int retrain, maxfaktor;
		// int primal_optimal = 0;
		int at_bound, scalemaxiter;
		double epsilon_a = 1E-15, epsilon_hideo;
		// double eq;

		if ((m < 0) || (m > 1)) {
			System.err.println("SOLVE DUAL: inappropriate number of eq-constrains!");
		}

		for (i = 0; i < 2 * (n + m); i++) {
			dual[i] = 0;
			dual_old[i] = 0;
		}

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) { // dual hessian for box constraints
				d[i * 2 * (n + m) + j] = ig[i * n + j];
				d[(i + n) * 2 * (n + m) + j] = -ig[i * n + j];
				d[i * 2 * (n + m) + j + n] = -ig[i * n + j];
				d[(i + n) * 2 * (n + m) + j + n] = ig[i * n + j];
			}
			if (m > 0) {
				sum = 0; // dual hessian for eq constraints
				for (j = 0; j < n; j++) {
					sum += (ce[j] * ig[i * n + j]);
				}
				d[i * 2 * (n + m) + 2 * n] = sum;
				d[i * 2 * (n + m) + 2 * n + 1] = -sum;
				d[(n + i) * 2 * (n + m) + 2 * n] = -sum;
				d[(n + i) * 2 * (n + m) + 2 * n + 1] = sum;
				d[(n + n) * 2 * (n + m) + i] = sum;
				d[(n + n + 1) * 2 * (n + m) + i] = -sum;
				d[(n + n) * 2 * (n + m) + (n + i)] = -sum;
				d[(n + n + 1) * 2 * (n + m) + (n + i)] = sum;

				sum = 0;
				for (j = 0; j < n; j++) {
					for (k = 0; k < n; k++) {
						sum += (ce[k] * ce[j] * ig[j * n + k]);
					}
				}
				d[(n + n) * 2 * (n + m) + 2 * n] = sum;
				d[(n + n) * 2 * (n + m) + 2 * n + 1] = -sum;
				d[(n + n + 1) * 2 * (n + m) + 2 * n] = -sum;
				d[(n + n + 1) * 2 * (n + m) + 2 * n + 1] = sum;
			}
		}

		for (i = 0; i < n; i++) { // dual linear component for the box
									// constraints
			w = 0;
			for (j = 0; j < n; j++) {
				w += (ig[i * n + j] * g0[j]);
			}
			d0[i] = up[i] + w;
			d0[i + n] = -low[i] - w;
		}

		if (m > 0) {
			sum = 0; // dual linear component for eq constraints
			for (j = 0; j < n; j++) {
				for (k = 0; k < n; k++) {
					sum += (ce[k] * ig[k * n + j] * g0[j]);
				}
			}
			d0[2 * n] = ce0[0] + sum;
			d0[2 * n + 1] = -ce0[0] - sum;
		}

		// maxviol = 999999;
		iter = 0;
		// retrain = 1;
		// maxfaktor = 1;
		scalemaxiter = maxiter / 5;

		// ========= main loop ===================

		DualCheck dc = new DualCheck();
		dc.retrain = 1;
		dc.maxviol = 999999;
		dc.maxfaktor = 1;
		dc.precision = precision;
		dc.primal_optimal = 0;

		while ((dc.retrain > 0) && (dc.maxviol > 0) && (iter < (scalemaxiter * dc.maxfaktor))) {
			iter++;

			// Hildreth and D'Espo route here
			while ((dc.maxviol > precision) && (iter < (scalemaxiter * dc.maxfaktor))) {
				iter++;
				dc.maxviol = 0;

				// the main update step here ,s.t.
				// d0(i)+sum(j)_1^m{d(ij)*dual(j)}=0
				for (i = 0; i < 2 * (n + m); i++) {
					sum = d0[i];
					for (j = 0; j < 2 * (n + m); j++) {
						sum += d[i * 2 * (n + m) + j] * dual_old[j];
					}

					// remove the dual[i] self effect so sum is
					// d0(i)-sum(j|j!=i)_1^m{d(ij)} here
					// Reference: introduction to nonlinear optimization D.A.
					// Wismer R. chattergy Hildreth and D'Espo part
					sum -= d[i * 2 * (n + m) + i] * dual_old[i];
					dual[i] = -sum / d[i * 2 * (n + m) + i];

					// not less than zero
					if (dual[i] < 0)
						dual[i] = 0;

					viol = Math.abs(dual[i] - dual_old[i]);

					// test whether the dual solution is converged
					if (viol > dc.maxviol)
						dc.maxviol = viol;
					dual_old[i] = dual[i];
				}

			}// end of Hildreth and D'Espo

			if (m > 0) {
				for (i = 0; i < n; i++) {
					// dual[i],dual[i+n] stands for dual variable for up ,low
					// constraint ,dual[n+n],dual[n+n+1]
					// stands for dual variable for equality constraint,b stands
					// for g0
					temp[i] = dual[i] - dual[i + n] + ce[i] * (dual[n + n] - dual[n + n + 1]) + g0[i];
				}
			} else {
				for (i = 0; i < n; i++) {
					temp[i] = dual[i] - dual[i + n] + g0[i];// dual[i]一个关于上界,一个关于下界
				}
			}
			for (i = 0; i < n; i++) {
				primal[i] = 0; // calc value of primal variables

				// x(*)=-ig*(B^T*dual(*)+b), the element of B is either +1 or -1
				for (j = 0; j < n; j++) {
					primal[i] += ig[i * n + j] * temp[j];
				}
				primal[i] *= -1.0;
				if (primal[i] <= (low[i])) { // clip conservatively
					primal[i] = low[i];
				} else if (primal[i] >= (up[i])) {
					primal[i] = up[i];
				}
			}

			if (m > 0)
				model_b = dual[n + n + 1] - dual[n + n];
			else
				model_b = 0;

			// checkDual(n, m, model_b, ce, ce0, primal, g, g0, up, low,
			// epsilon_crit, epsilon_a, goal, dc);

		}// end of outer loop

		if (dc.primal_optimal == 0) {
			for (i = 0; i < n; i++) {
				primal[i] = 0; // calc value of primal variables
				for (j = 0; j < n; j++) {
					primal[i] += ig[i * n + j] * temp[j];
				}
				primal[i] *= -1.0;
				if (primal[i] <= (low[i] + epsilon_a)) { // clip conservatively
					primal[i] = low[i];
				} else if (primal[i] >= (up[i] - epsilon_a)) {
					primal[i] = up[i];
				}
			}
		}

		isnantest = 0;
		for (i = 0; i < n; i++) { // check for isnan
			isnantest += primal[i];
		}

		if (m > 0) {
			temp1 = dual[n + n + 1]; // copy the dual variables for the eq
			temp2 = dual[n + n]; // constraints to a handier location
			for (i = n + n + 1; i >= 2; i--) {
				dual[i] = dual[i - 2];
			}
			dual[0] = temp2;
			dual[1] = temp1;
			isnantest += temp1 + temp2;
		}

		if (Double.isNaN(isnantest)) {
			return ((int) NAN_SOLUTION);
		} else if (dc.primal_optimal > 0) {

			return ((int) PRIMAL_OPTIMAL);
		} else if (dc.maxviol == 0.0) {
			return ((int) DUAL_OPTIMAL);
		} else {
			return ((int) MAXITER_EXCEEDED);
		}

	}

	public void checkDual(int n, int m, double model_b, double[] ce, double[] ce0, double[] primal, double[] g, double[] g0, double[] up, double[] low, double epsilon_crit, double epsilon_a, int goal, DualCheck dc) {
		int at_bound = 0;
		double dist = 0, epsilon_hideo = 0, eq;
		dc.primal_optimal = 1;
		at_bound = 0;

		int i, j;

		epsilon_hideo = EPSILON_HIDEO;

		// the following are two different termination criteria for the
		// algorithm
		// first is checking the alphas(which is derived from primal KKT
		// conditions)
		// second is checking the primal KKT conditions
		// reference:Making Large-scale SVM Learning practical (Thorsten
		// Joachims)
		// g stands for y*k ,primal stands for alphas,the y is all one here
		//

		for (i = 0; i < n; i++) { // check precision of alphas
			dist = -model_b * ce[i];
			dist += (g0[i] + 1.0);
			for (j = 0; j < i; j++) {
				dist += (primal[j] * g[j * n + i]);
			}
			for (j = i; j < n; j++) {
				dist += (primal[j] * g[i * n + j]);
			}
			if ((primal[i] < (up[i] - epsilon_hideo)) && (dist < (1.0 - epsilon_crit))) {
				epsilon_hideo = (up[i] - primal[i]) * 2.0;
			} else if ((primal[i] > (low[i] + epsilon_hideo)) && (dist > (1.0 + epsilon_crit))) {
				epsilon_hideo = (primal[i] - low[i]) * 2.0;
			}
		}

		for (i = 0; i < n; i++) { // clip alphas to bounds
			if (primal[i] <= (low[i] + epsilon_hideo)) {
				primal[i] = low[i];
			} else if (primal[i] >= (up[i] - epsilon_hideo)) {
				primal[i] = up[i];
			}
		}

		dc.retrain = 0;

		for (i = 0; (i < n); i++) { // check primal KT-Conditions
			dist = -model_b * ce[i];
			dist += (g0[i] + 1.0);
			for (j = 0; j < i; j++) {
				dist += (primal[j] * g[j * n + i]);
			}
			for (j = i; j < n; j++) {
				dist += (primal[j] * g[i * n + j]);
			}

			if ((primal[i] < (up[i] - epsilon_a)) && (dist < (1.0 - epsilon_crit))) {
				dc.retrain = 1;
				dc.primal_optimal = 0;
			} else if ((primal[i] > (low[i] + epsilon_a)) && (dist > (1.0 + epsilon_crit))) {
				dc.retrain = 1;
				dc.primal_optimal = 0;
			}
			if ((primal[i] <= (low[i] + epsilon_a)) || (primal[i] >= (up[i] - epsilon_a))) {
				at_bound++;
			}

		}
		if (m > 0) {
			eq = -ce0[0]; // check precision of eq-constraint
			for (i = 0; i < n; i++) {
				eq += (ce[i] * primal[i]);
			}
			if ((EPSILON_EQ < Math.abs(eq))) {
				dc.retrain = 1;
				dc.primal_optimal = 0;
			}

		}

		if (dc.retrain > 0) {
			dc.precision /= 10;
			if (((goal == PRIMAL_OPTIMAL) && (dc.maxfaktor < 50000)) || (dc.maxfaktor < 5)) {
				dc.maxfaktor++;
			}
		}
	}

	public void switchrkMatrix(double[] matrix, int depth, int rk1, int rk2) {
		int i;
		double temp;
		for (i = 0; i < depth; i++) {
			temp = matrix[rk1 * depth + i];
			matrix[rk1 * depth + i] = matrix[rk2 * depth + i];
			matrix[rk2 * depth + i] = temp;
		}

		for (i = 0; i < depth; i++) {
			temp = matrix[i * depth + rk1];
			matrix[i * depth + rk1] = matrix[i * depth + rk2];
			matrix[i * depth + rk2] = temp;
		}
	}

	public void invertMatrix(double[] matrix, int depth, double[] inverse, double lindep_sensitivity, int[] lin_dependent) {
		int i, j, k;
		double factor;
		for (i = 0; i < depth; i++) {
			// lin_dependent[i]=0;
			for (j = 0; j < depth; j++) {
				inverse[i * depth + j] = 0.0;
			}
			inverse[i * depth + i] = 1.0;
		}
		for (i = 0; i < depth; i++) {
			if ((lin_dependent[i] > 0) || (Math.abs(matrix[i * depth + i]) < lindep_sensitivity)) {
				lin_dependent[i] = 1;
			} else {
				for (j = i + 1; j < depth; j++) {
					factor = matrix[j * depth + i] / matrix[i * depth + i];
					for (k = i; k < depth; k++) {
						matrix[j * depth + k] -= (factor * matrix[i * depth + k]);
					}
					for (k = 0; k < depth; k++) {
						inverse[j * depth + k] -= (factor * inverse[i * depth + k]);
					}
				}
			}
		}
		for (i = depth - 1; i >= 0; i--) {
			if (lin_dependent[i] == 0) {
				factor = 1 / matrix[i * depth + i];
				for (k = 0; k < depth; k++) {
					inverse[i * depth + k] *= factor;
				}
				matrix[i * depth + i] = 1;
				for (j = i - 1; j >= 0; j--) {
					factor = matrix[j * depth + i];
					matrix[j * depth + i] = 0;
					for (k = 0; k < depth; k++) {
						inverse[j * depth + k] -= (factor * inverse[i * depth + k]);
					}
				}
			}
		}

	}

	public double calculateQpObjective(int opt_n, double[] opt_g, double[] opt_g0, double[] alpha) {
		double obj;
		int i, j;
		obj = 0;

		// calculate objective
		for (i = 0; i < opt_n; i++) {
			obj += opt_g0[i] * alpha[i];
			obj += 0.5 * alpha[i] * alpha[i] * opt_g[i * opt_n + i];
			for (j = 0; j < i; j++) {
				obj += alpha[j] * alpha[i] * opt_g[j * opt_n + i];
			}
		}
		return (obj);

	}

	public static void main(String[] args) {

		/*
		double[] opt_low = { 0, 0 };
		double[] opt_up = { 1, 1 };

		double[] opt_g = { 1, 0, 0, 1 };
		double[] opt_g0 = { -2, -2 };

		double[] opt_xinit = { 0, 0 };
		double[] opt_ce = { 0, 0 };
		double[] opt_ce0 = { 0 };

		int opt_m = 0;
		int opt_n = 2;

		QP qp = new QP();
		qp.opt_up = opt_up;
		qp.opt_low = opt_low;
		qp.opt_g = opt_g;
		qp.opt_g0 = opt_g0;
		qp.opt_n = opt_n;
		qp.opt_m = opt_m;
		qp.opt_xinit = opt_xinit;
		qp.opt_ce = opt_ce;
		qp.opt_ce0 = opt_ce0;

		Hideo hd = new Hideo();

		LEARN_PARM lp = new LEARN_PARM();
		double[] res = hd.optimizeQp(qp, 50, 2, 0, lp);

		for (int i = 0; i < res.length; i++) {
			System.err.println(i + ":" + res[i]);
		}
		
		*/

		double[] opt_low = { 0, 0 };
		double[] opt_up = { 100, 100 };

		double[] opt_g = { 2, -1, -1, 1 };
		double[] opt_g0 = { -1, 0 };

		double[] opt_xinit = { 0, 0 };
		double[] opt_ce = { 3, 2 };
		double[] opt_ce0 = { -5 };

		int opt_m = 1;
		int opt_n = 2;

		QP qp = new QP(2);
		qp.opt_up = opt_up;
		qp.opt_low = opt_low;
		qp.opt_g = opt_g;
		qp.opt_g0 = opt_g0;
		qp.opt_n = opt_n;
		qp.opt_m = opt_m;
		qp.opt_xinit = opt_xinit;
		qp.opt_ce = opt_ce;
		qp.opt_ce0 = opt_ce0;

		Hideo hd = new Hideo();

		LEARN_PARM lp = new LEARN_PARM();
		double[] res = hd.optimizeQp(qp, 50, 2, 0, lp);

		for (int i = 0; i < res.length; i++) {
			System.err.println(i + ":" + res[i]);
		}

	}

}
