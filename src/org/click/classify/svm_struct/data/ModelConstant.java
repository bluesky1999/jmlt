package org.click.classify.svm_struct.data;

public class ModelConstant {

	public static final String VERSION = "V6.20";
	public static final String VERSION_DATE = "14.08.08";

	public static final short LINEAR = 0;
	public static final short POLY = 1;
	public static final short RBF = 2;
	public static final short SIGMOID = 3;
	public static final short CUSTOM = 4;
	public static final short GRAM = 5;

	public static final short CLASSIFICATION = 1;
	public static final short REGRESSION = 2;
	public static final short RANKING = 3;
	public static final short OPTIMIZATION = 4;

	public static final int MAXSHRINK = 50000;

	public static final String INST_NAME = "Multi-Class SVM";

	public static final String INST_VERSION = "V2.20";

	public static final String INST_VERSION_DATE = "14.08.08";

	/**
	 * default precision for solving the optimization problem
	 */
	public static final double c = 0.1;

	/**
	 * default loss rescaling method: 1=slack_rescaling, 2=margin_rescaling
	 */
	public static final short DEFAULT_RESCALING = 2;

	/**
	 * default loss function:
	 */
	public static final int DEFAULT_LOSS_FCT = 0;

	/**
	 * default optimization algorithm to use:
	 */
	public static final int DEFAULT_ALG_TYPE = 4;

	/**
	 * store Psi(x,y) once instead of recomputing it every time:
	 */
	public static final int USE_FYCACHE = 0;

	/**
	 * decide whether to evaluate sum before storing vectors in constraint
	 * cache: 0 = NO, 1 = YES (best, if sparse vectors and long vector lists), 2
	 * = YES (best, if short vector lists), 3 = YES (best, if dense vectors and
	 * long vector lists)
	 */
	public static final short COMPACT_CACHED_VECTORS = 2;

	/**
	 * minimum absolute value below which values in sparse vectors are rounded
	 * to zero. Values are stored in the FVAL type defined in svm_common.h
	 * RECOMMENDATION: assuming you use FVAL=float, use 10E-15 if
	 * COMPACT_CACHED_VECTORS is 1 10E-10 if COMPACT_CACHED_VECTORS is 2 or 3
	 */
	public static final double COMPACT_ROUNDING_THRESH = 10E-15;
	
	public static final int ZEROONE=  0;
	public static final int FONE     =    1;
	public static final int ERRORRATE  =  2;
	public static final int PRBEP     =   3;
	public static final int PREC_K    =   4;
	public static final int REC_K     =   5;
	public static final int SWAPPEDPAIRS= 10;
	public static final int AVGPREC    =  11;
}
