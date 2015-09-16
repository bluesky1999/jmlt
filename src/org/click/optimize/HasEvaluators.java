package org.click.optimize;

/**
 * Indicates that an minimizer supports evaluation periodically
 *
 * @author Angel Chang
 */
public interface HasEvaluators {
  public void setEvaluators(int iters, Evaluator[] evaluators);
}