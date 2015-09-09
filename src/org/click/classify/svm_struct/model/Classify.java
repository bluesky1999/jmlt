package org.click.classify.svm_struct.model;

import org.apache.log4j.Logger;
import org.click.classify.svm_struct.data.DOC;
import org.click.classify.svm_struct.data.MODEL;
import org.click.classify.svm_struct.data.ModelConstant;
import org.click.classify.svm_struct.data.ReadStruct;
import org.click.classify.svm_struct.data.ReadSummary;
import org.click.classify.svm_struct.data.WORD;
import org.jmlp.file.utils.FileToArray;
import org.jmlp.str.basic.SSO;

public class Classify {

	public static String docfile;
	public static String modelfile;
	public static String predictionsfile;
	public static int verbosity;
	public static int pred_format;
	private static Logger logger = Logger.getLogger(Classify.class);  

	public static void main(String[] args) throws Exception {
		DOC doc;
		WORD[] words;
		int max_docs, max_words_doc, lld;
		int totdoc = 0, queryid, slackid;
		int correct = 0, incorrect = 0, no_accuracy = 0;
		int res_a = 0, res_b = 0, res_c = 0, res_d = 0, wnum, pred_format;

		int j;
		double t1, runtime = 0;
		double dist, doc_label, costfactor;
		String line, comment;
		MODEL model;

		read_input_parameters(args.length + 1, args);
		
		ReadSummary summary=null;
		summary=Common.nol_ll(docfile);
		
		max_docs = summary.read_max_docs;
		max_words_doc = summary.read_max_words_doc;
		max_words_doc += 2;
		model = Common.read_model(modelfile);

		if (model.kernel_parm.kernel_type == 0) { /* linear kernel */
			/* compute weight vector */
			Common.add_weight_vector_to_linear_model(model);
		}

		String[] test_samples = FileToArray.fileToDimArr(docfile);
        WORD[] read_words;
		for (int i = 0; i < test_samples.length; i++) {
			line = test_samples[i];
			
			ReadStruct rs=new ReadStruct();
			read_words=Common.parse_document(line, max_words_doc,rs);
			doc_label = rs.read_doc_label;
			queryid = rs.read_queryid;
			slackid = rs.read_slackid;
			costfactor = rs.read_costfactor;
			comment = rs.read_comment;
			//words = svm_common.read_words;
			words = read_words;
			doc = Common.createExample(-1, 0, 0, 0.0,
					Common.createSvector(words, comment, 1.0));
			if (model.kernel_parm.kernel_type == ModelConstant.LINEAR) { 
			    logger.info("kernel type is linear aa");
				dist = Common.classify_example_linear(model, doc);
			} else { /* non-linear kernel */
				logger.info("kernel type is nonlinear");
				dist = Common.classify_example(model, doc);
			}

			if (dist > 0) {
				totdoc++;
				if (doc_label > 0)
					correct++;
				else
					incorrect++;
				if (doc_label > 0)
					res_a++;
				else
					res_b++;
			} else {
				totdoc++;
				if (doc_label < 0)
					correct++;
				else
					incorrect++;
				if (doc_label > 0)
					res_c++;
				else
					res_d++;
			}
		}
		
	
		System.out.println("Accuracy on test set:"+(float)(correct)*100.0/totdoc+" ("+correct+" correct, "+incorrect+" incorrect,"+totdoc+" total)\n");
			   
			  

	}

	public static void read_input_parameters(int argc, String[] argv) {
		int i;

		/* set default */
		modelfile = "svm_model";
		predictionsfile = "svm_predictions";
		verbosity = 2;
		pred_format = 1;

		for (i = 1; (i < argc) && ((argv[i]).charAt(0) == '-'); i++) {
			switch ((argv[i]).charAt(1)) {
			case 'h':
				print_help();
				System.exit(0);
			case 'v':
				i++;
				verbosity = Integer.parseInt(argv[i]);
				break;
			case 'f':
				i++;
				pred_format = Integer.parseInt(argv[i]);
				break;
			default:
				System.out.println("\nUnrecognized option !" + argv[i]);
				print_help();
				System.exit(0);
			}
		}
		if ((i + 1) >= argc) {
			System.out.println("\nNot enough input parameters!\n\n");
			print_help();
			System.exit(0);
		}
		i=0;
		docfile = argv[i];
		modelfile = argv[i + 1];
		if ((i + 2) < argc) {
			predictionsfile = argv[i + 2];
		}
		if (((pred_format) != 0) && ((pred_format) != 1)) {
			System.out
					.println("\nOutput format can only take the values 0 or 1!\n\n");
			print_help();
			System.exit(0);
		}
	}

	public static void print_help() {

	}
}
