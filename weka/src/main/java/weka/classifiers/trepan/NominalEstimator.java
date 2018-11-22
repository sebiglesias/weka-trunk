package weka.classifiers.trepan;

import java.util.*;
import weka.core.*;

/**
 * Simple symbolic probability estimator based on symbol counts
  */


public class NominalEstimator implements TrepanEstimator {
    /** Hold the counts */
    private double [] counts;
    /** Hold the sum of counts */
    private double sumOfCounts;
    /** Hold the accumulated probabilities */
    private double [] accumulatedCounts;

    /**
     * Constructor
     *
     * @param numSymbols the number of possible symbols (including 0)
     * @param laplace if true, counts will be initialised to 1
     */
    public NominalEstimator(int numSymbols, boolean laplace) {
        counts = new double [numSymbols];
        accumulatedCounts = new double [numSymbols];
        sumOfCounts = 0;
        if (laplace) {
            for (int i = 0; i < numSymbols; i++) {
                counts[i] = 1;
            }
            sumOfCounts = (double)numSymbols;
        }
    }

    /**
     * Add a new data value to the current estimator
     * @param data new data value
     * @param weight weight assigned to the data value
     */
    public void addValue(double data, double weight) {
        counts[(int)data] += weight;
        sumOfCounts += weight;
    }

    /**
     * Get a Propability estimate for a value
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    public double getProbability(double data) {
        return (sumOfCounts == 0) ? 0 : counts[(int)data] / sumOfCounts;
    }

    /**
     * Gets the number of symbols this estimator operates with
     * @return the number of estimator symbols
     */
    public int getNumSymbols() {
        return (counts == null) ? 0 : counts.length;
    }

    /**
     * Generates a random value according to the estimator distribution
     * @param seed random seed
     * @return the value generated
     */
    public double newValue(long seed) {
        // If no values then return missing value
        if (sumOfCounts == 0) {
            return Double.NaN;
        }

        Random random = new Random(seed);
        double r;
        double sum = 0;
        int value = 0;

        //Accumulate probabilities
        for (int i = 0; i < accumulatedCounts.length; i++) {
            sum = sum + getProbability((double)i);
            accumulatedCounts[i] = sum;
        }

        // Sum of probabilities = 1
        accumulatedCounts[accumulatedCounts.length - 1] = 1;

        // Generate a random value to choose the value to be returned
        r = random.nextDouble();
        int j = 0;
        boolean found = false;
        while ( (!found) && (j < counts.length)) {
            if (accumulatedCounts[j] >= r) {
                value = j;
                found = true;
            }
            j++;
        }
        return (double)value;
    }

    /**
     * Display a representation of this estimator
     * @return a string representation of this estimator
     */
    public String toString() {
        StringBuilder result = new StringBuilder("Discrete Estimator. Counts = ");
        if (sumOfCounts > 1) {
            for (double m_Count : counts) {
                result.append(" ").append(Utils.doubleToString(m_Count, 2));
            }
            result.append(" (Total = ").append(Utils.doubleToString(sumOfCounts, 2)).append(")\n");
        } else {
            for (double m_Count : counts) {
                result.append(" ").append(m_Count);
            }
            result.append(" (Total = ").append(sumOfCounts).append(")\n");
        }
        return result.toString();
    }

    /**
     * Main method for testing this class
     * @param argv should contain a sequence of integers which will be treated as symbolic
     */
    public static void main(String [] argv) {
        try {
            if (argv.length == 0) {
                System.out.println("Please specify a set of instances.");
                return;
            }
            int current = Integer.parseInt(argv[0]);
            int max = current;
            for (String anArgv : argv) {
                current = Integer.parseInt(anArgv);
                if (current > max) {
                    max = current;
                }
            }
            NominalEstimator newEst = new NominalEstimator(max + 1, true);
            for (String anArgv : argv) {
                current = Integer.parseInt(anArgv);
                System.out.println(newEst);
                System.out.println("Prediction for " + current + " = " + newEst.getProbability(current));
                newEst.addValue(current, 1);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}