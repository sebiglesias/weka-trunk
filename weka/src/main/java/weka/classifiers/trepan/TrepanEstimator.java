package weka.classifiers.trepan;

import java.util.*;
import java.io.*;

public interface TrepanEstimator extends Serializable {
    /**
     * Add a new data value to the current estimator
     *
     * @param data new data value
     * @param weight weight assigned to the data value
     */
    public void addValue(double data, double weight);

    /**
     * Get a probability estimate for a value
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    public double getProbability(double data);

    /**
     * Generates a random value according to the estimator distribution
     * @param seed random seed
     * @return the value generated
     */
    public double newValue(long seed);
}