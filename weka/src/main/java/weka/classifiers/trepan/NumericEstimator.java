// NumericEstimator
package weka.classifiers.trepan;
import java.util.*;
import weka.core.*;

/**
 * Simple kernel density estimator. Uses one gaussian kernel per observed data value
 */

public class NumericEstimator implements TrepanEstimator {

    /** Vector containing all of the values seen */
    private double [] values;

    /** Vector containing the associated weights */
    private double [] weights;

    /** Number of values stored in weights and values so far */
    private int valuesCount;

    /** The sum of the weights so far */
    private double sumOfWeights;

    /** The standard deviation */
    private double standardDeviation;

    /** The precision of data values */
    private double precision;

    /** Whether we can optimise the kernel summation */
    private boolean allWeightsOne;

    /** Maximum percentage error permitted in probability calculations */
    private static double MAX_ERROR = 0.01;

    /**
     * Execute a binary search to locate the nearest data value
     * @param key data value to locate
     * @return the index of the nearest data value
     */
    private int findNearestValue(double key) {
        int low = 0;
        int high = valuesCount;
        int middle = 0;
        while (low < high) {
            middle = (low + high) / 2;
            double current = values[middle];
            if ( current == key) {
                return middle;
            }
            if ( current > key) {
                high = middle;
            } else if (current < key) {
                low = middle + 1;
            }
        }
        return low;
    }

    /**
     * Round a data value using the defined precision for this estimator
     * @param data the value to round
     * @return the rounded data value
     */
    private double round(double data) {
        return Math.rint(data / precision) * precision;
    }

    /**
     * Constructor that takes a precision argument
     * @param precision to which numeric values are given. If the precision is states to be 0.1,
     *                  the values in the interval (0.25, 0.35] are all treated as 0.3
     */
    public NumericEstimator(double precision) {
        this.values = new double [50];
        this.weights = new double [50];
        this.valuesCount = 0;
        this.sumOfWeights = 0;
        this.allWeightsOne = true;
        this.precision = precision;
        this.standardDeviation = this.precision / (2 * 3);
    }

    /**
     * Add a new data value to the current estimator
     * @param data new data value
     * @param weight weight assigned to the data value
     */
    public void addValue(double data, double weight) {
        if (weight == 0) {
            return;
        }
        data = round(data);
        int insertIndex = findNearestValue(data);
        if ((valuesCount <= insertIndex) || (values[insertIndex] !=data)) {
            if (valuesCount < values.length) {
                int left = valuesCount - insertIndex;
                System.arraycopy(values, insertIndex, values, insertIndex + 1, left);
                System.arraycopy(weights, insertIndex, weights, insertIndex + 1, left);
                values[insertIndex] = data;
                weights[insertIndex] = weight;
                valuesCount++;
            } else {
                double [] newValues = new double [values.length * 2];
                double [] newWeights = new double [values.length * 2];
                int left = valuesCount - insertIndex;
                System.arraycopy(values, 0, newValues, 0, insertIndex);
                System.arraycopy(weights, 0, newWeights, 0, insertIndex);
                newValues[insertIndex] = data;
                newWeights[insertIndex] = weight;
                System.arraycopy(values, insertIndex, newValues, insertIndex + 1, left);
                System.arraycopy(weights, insertIndex, newWeights, insertIndex + 1, left);
                valuesCount++;
                values = newValues;
                weights = newWeights;
            }
            if (weight != 1) {
                allWeightsOne = false;
            }
        } else {
            weights[insertIndex] += weight;
            allWeightsOne = false;
        }
        sumOfWeights += weight;
        double range = values[valuesCount - 1] - values[0];
        if (range > 0) {
            // allow at most 3 standardDeviations within one interval
            standardDeviation = Math.max(range / Math.sqrt(sumOfWeights), precision / (2 * 3));
        }
    }

    /**
     * Generates a random value according to the estimator distribution
     * @param seed random seed
     * @return the value generated
     */
    public double newValue(long seed) {
        // If no values then return missing value
        if (valuesCount == 0) {
            return Double.NaN;
        }

        // Accumulate probabilities
        double sumProb = 0;
        double[] m_Prob = new double[valuesCount];
        for (int k = 0; k < valuesCount; k++) {
            sumProb = sumProb + getProbability(values[k]);
            m_Prob[k] = sumProb;
        }

        // Generate a random value to choose the value to be returned
        Random random = new Random(seed);
        double r = random.nextDouble();
        while (r > sumProb) {
            r = random.nextDouble();
        }

        // Find interval containing random value
        int interval = 0;
        while (m_Prob[interval] < r) {
            interval ++;
        }

        // Choose the sign of the desviation
        double sign;
        r = random.nextDouble();
        if (Math.abs(r) < 0.5) {
            sign = 1;
        } else {
            sign = -1;
        }

        // Generates value X = mu + sign.random.deviation ( with mu = interval )
        double value;
        r = random.nextDouble();
        value = (values)[interval] + (sign * r * standardDeviation);
        return value;
    }

    /**
     * Get a probability estimate for a value
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    public double getProbability(double data) {
        double delta = 0, sum = 0, currentProb = 0;
        double zLower = 0, zUpper = 0;
        if (valuesCount == 0) {
            zLower = (data - (precision / 2)) / standardDeviation;
            zUpper = (data + (precision / 2)) / standardDeviation;
            return (Statistics.normalProbability(zUpper) - Statistics.normalProbability(zLower));
        }
        double weightSum = 0;
        int start = findNearestValue(data);
        for (int i = start; i < valuesCount; i++) {
            delta = values[i] - data;
            zLower = (delta - (precision / 2)) / standardDeviation;
            zUpper = (delta + (precision / 2)) / standardDeviation;
            currentProb = (Statistics.normalProbability(zUpper) - Statistics.normalProbability(zLower));
            sum += currentProb * weights[i];
            weightSum += weights[i];
            if (currentProb * (sumOfWeights - weightSum) < sum * MAX_ERROR) {
                break;
            }
        }
        for (int i = start; i >= 0; i--) {
            delta = values[i] - data;
            zLower = (delta - (precision / 2)) / standardDeviation;
            zUpper = (delta + (precision / 2)) / standardDeviation;
            currentProb = (Statistics.normalProbability(zUpper) - Statistics.normalProbability(zLower));
            sum += currentProb * weights[i];
            weightSum += weights[i];
            if (currentProb * (sumOfWeights - weightSum) < sum * MAX_ERROR) {
                break;
            }
        }
        return sum / sumOfWeights;
    }

    /**
     * Display a representation of this estimator
     * @return string representation of Numeric Estimator
     */
    public String toString() {
        StringBuilder result = new StringBuilder(valuesCount + " Normal Kernels. \n StandardDev = " +
                Utils.doubleToString(standardDeviation, 6, 4) + " Precision = " + precision);
        if (valuesCount == 0) {
            result.append(" \nMean = 0");
        } else {
            result.append(" \nMeans =");
            for (int i = 0; i < valuesCount; i++) {
                result.append(" ").append(values[i]);
            }
            if (!allWeightsOne) {
                result.append("\nWeights = ");
                for (int i = 0; i < valuesCount; i++) {
                    result.append(" ").append(weights[i]);
                }
            }
        }
        return result + "\n";
    }

    /**
     * Main method for testing this class
     * @param argv should contain a sequence of numeric values
     */
    public static void main(String [] argv) {
        try {
            if (argv.length < 2) {
                System.out.println("Please specify a set of instances.");
                return;
            }
            NumericEstimator newEst = new NumericEstimator(0.01);
            for (int i = 0; i < argv.length - 3; i += 2) {
                newEst.addValue(Double.valueOf(argv[i]), Double.valueOf(argv[i + 1]));
            }
            System.out.println(newEst);

            double start = Double.valueOf(argv[argv.length - 2]);
            double finish = Double.valueOf(argv[argv.length - 1]);
            for (double current = start; current < finish; current += (finish - start) / 50) {
                System.out.println("Data: " + current + " " + newEst.getProbability(current));
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}