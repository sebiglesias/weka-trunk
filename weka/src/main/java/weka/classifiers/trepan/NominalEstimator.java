package weka.classifiers.trepan;

import java.util.*;
import weka.core.*;

/**
 * Simple symbolic probability estimator based on symbol counts
  */


public class NominalEstimator implements TrepanEstimator {
    /** Hold the counts */
    private double [] m_Counts;
    /** Hold the sum of counts */
    private double m_SumOfCunts;
    /** Hold the accumulated probabilities */
    private double [] m_AcumCounts;

    /**
     * Constructor
     *
     * @param numSymbols the number of possible symbols
     * @param laplace if true, counts will be initialised to 1
     */
    public NominalEstimator(int numSymbols, boolean laplace) {
        m_Counts = new double [numSymbols];
        m_AcumCounts = new double [numSymbols];
        m_SumOfCunts = 0;
        if (laplace) {
            for (int i = 0; i < numSymbols; i++) {
                m_Counts[i] = 1;
            }
            m_SumOfCunts = (double)numSymbols;
        }
    }

    // Add a new data value to the current estimator
    // data = new data value
    // weight = weight assigned to the data value
    public void addValue(double data, double weight) {
        m_Counts[(int)data] += weight;
        m_SumOfCunts += weight;
    }

    // get a probability estimate for a value
    // data the value to estimate the probability of
    // the estimated probability of the supplied value
    public double getProbability(double data) {
        if (m_SumOfCunts == 0) {
            return 0;
        }
        return (double)m_Counts[(int)data] / m_SumOfCunts;
    }

    // gets the number of symbols this estimator operates with
    // return the number of estimator symbols
    public int getNumSymbols() {
        return (m_Counts == null) ? 0 : m_Counts.length;
    }

    // generates a random value according to the estimator distribution
    // seed = random seed
    // retun the value generated
    public double newValue(long seed) {
        // If no values then return missing value
        if (m_SumOfCunts == 0) {
            return Double.NaN;
        }

        Random random = new Random(seed);
        double r;
        double sum = 0;
        int value = 0;

        //Accumulate probabilities
        for (int i =0; i < m_AcumCounts.length; i++) {
            sum = sum + getProbability((double)i);
            m_AcumCounts[i] = sum;
        }

        // Sum of probabilities = 1
        m_AcumCounts[m_AcumCounts.length - 1] = 1;

        // Genrate a random value to choose the value to be returned
        r = random.nextDouble();
        int j = 0;
        boolean found = false;
        while ( (found == false) && (j < m_Counts.length)) {
            if (m_AcumCounts[j] >= r) {
                value = j;
                found = true;
            }
            j++;
        }
        return (double)value;
    }

    // Display a representation of this estimator
    public String toString() {
        String result = "Discrete Estimator. Counts = ";
        if (m_SumOfCunts > 1) {
            for (int i = 0; i < m_Counts.length; i++) {
                result += " " + Utils.doubleToString(m_Counts[i], 2);
            }
            result += " (Total = " + Utils.doubleToString(m_SumOfCunts, 2) + ")\n";
        } else {
            for (int i =0; i < m_Counts.length; i++) {
                result += " " + m_Counts[i];
            }
            result += " (Total = " + m_SumOfCunts + ")\n";
        }
        return result;
    }

    // Main method for testing this class
    // argv should contain a sequence of integers which will be treated as symbolic
    public static void main(String [] argv) {
        try {
            if (argv.length == 0) {
                System.out.println("Please specify a set of instances.");
                return;
            }
            int current = Integer.parseInt(argv[0]);
            int max = current;
            for (int i =0; i < argv.length; i++) {
                current = Integer.parseInt(argv[i]);
                if (current > max) {
                    max = current;
                }
            }
            NominalEstimator newEst = new NominalEstimator(max + 1, true);
            for (int i = 0; i < argv.length; i++) {
                current = Integer.parseInt(argv[i]);
                System.out.println(newEst);
                System.out.println("Prediction for " + current + " = " + newEst.getProbability(current));
                newEst.addValue(current, 1);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}