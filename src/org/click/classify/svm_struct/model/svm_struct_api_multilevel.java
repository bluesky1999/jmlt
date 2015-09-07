package org.click.classify.svm_struct.model;

import java.io.InputStream;
import java.util.ArrayList;

import org.click.classify.svm_struct.data.KERNEL_PARM;
import org.click.classify.svm_struct.data.LABEL;
import org.click.classify.svm_struct.data.LEARN_PARM;
import org.click.classify.svm_struct.data.PATTERN;
import org.click.classify.svm_struct.data.SAMPLE;
import org.click.classify.svm_struct.data.STRUCTMODEL;
import org.click.classify.svm_struct.data.STRUCT_LEARN_PARM;
import org.click.classify.svm_struct.data.SVECTOR;

/**
 * 多层次分类实现
 * @author zkyz
 *
 */
public class svm_struct_api_multilevel extends svm_struct_api{

	@Override
	public void init_struct_model(SAMPLE sample, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm, LEARN_PARM lparm, KERNEL_PARM kparm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public SVECTOR psi(PATTERN x, LABEL y, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL find_most_violated_constraint_slackrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL find_most_violated_constraint_marginrescaling(PATTERN x,
			LABEL y, STRUCTMODEL sm, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean empty_label(LABEL y) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public SAMPLE read_struct_examples(String file, STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SAMPLE read_struct_examples_from_stream(InputStream is,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SAMPLE read_struct_examples_from_arraylist(ArrayList<String> list,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LABEL classify_struct_example(PATTERN x, STRUCTMODEL sm,
			STRUCT_LEARN_PARM sparm) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public PATTERN sample2pattern(String words) {
		// TODO Auto-generated method stub
		return null;
	}

}
