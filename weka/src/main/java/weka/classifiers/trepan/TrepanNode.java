// TrepanNode
package weka.classifiers.trepan;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.classifiers.*;
import weka.estimators.Estimator;
import java.lang.*;

public class TrepanNode {
    //subset of training examples that reach the node
    private Instances trainExamples;

    //set of query instances to be created
    private Instances queryExamples;

    // number of training instances and query instances
    private int m_CountTrainEx = 0;
    private int m_CountQueryEx = 0;

    // Attribute's distributions
    private TrepanEstimator [] m_Distributions;

    // Class value if node is leaf
    private double m_ClassLabel;

    // attribute used for splitting
    private Attribute m_Attribute;

    // Class distribution at node
    private double[] m_ClassDistribution;

    // Node's children
    private TrepanNode [] m_Children;

    // Num of children nodes
    private int m_CountChildren = 0;

    // Node's parent
    private TrepanNode m_Parent;

    // Node Type - Values R: Root, L: Leaf, N: Node
    private String m_NodeType;

    // Num of attributes of the instances
    private int m_NumAttr;

    // Num of classes in the dataset
    private int m_NumClass;

    // Candidate values for splitting
    private double[] attrValues;

    // Prop of instances that reach the node
    private double m_Reach;

    // Fidelity of the extracted tree
    private double m_Fidelity;

    // BF Value for the node
    private double m_Best_First;

    // ID NODE
    private int m_IdNodo;

    // random value
    private Random random;

    // class constructor
    // data = data reaching the node
    public TrepanNode(Instances data) {
        m_CountTrainEx = data.numInstances();

        setTrainExamples(data);

        m_NumAttr = data.numAttributes();
        m_NumClass = data.numClasses();

        // Inicialize random variable
        random = new Random(0);

        // Inicialize class distributions
        m_ClassDistribution = new double[m_NumClass];
    }

    // Generates new instances to evaluate the node
    // minSamples = min num of instances required to evaluate the node
    // oracle = to classify new instances
    // exception = if there is an error
    public void drawSample(int minSamples, Classifier oracle) throws Exception {
        // Num of instances to be created
        m_CountQueryEx = minSamples - m_CountTrainEx;

        if (m_CountQueryEx > 0) {
            // Distribution
            m_Distributions = new TrepanEstimator[m_NumAttr];

            Enumeration enumAtt = trainExamples.enumerateAttributes();
            while (enumAtt.hasMoreElements()) {
                Attribute attr = (Attribute) enumAtt.nextElement();
                if (attr.isNominal()) {
                    m_Distributions[attr.index()] = new NominalEstimator(attr.numValues(), false);
                } else {
                    m_Distributions[attr.index()] = new NumericEstimator(0.01);
                }
            }

            // Generate distribution for each attribute
            Enumeration enumInst = trainExamples.enumerateInstances();
            while (enumInst.hasMoreElements()) {
                Instance inst = (Instance) enumInst.nextElement();
                enumAtt = inst.enumerateAttributes();
                while (enumAtt.hasMoreElements()) {
                    Attribute attr = (Attribute) enumAtt.nextElement();
                    m_Distributions[attr.index()].addValue(inst.value(attr), 1);
                }
            }

            // Create new instances
            queryExamples = new Instances(trainExamples, m_CountQueryEx);
            queryExamples.delete();
            for (int j = 0; j < m_CountQueryEx; j++) {
                // Create new instance
                Instance inst = drawInstance();
                // Classify new instance
                double dClass = oracle.classifyInstance(inst);
                // Add instance in dataset
                inst.setClassValue(dClass);
                queryExamples.add(inst);
            }
        } else {
            m_CountQueryEx = 0;
        }
    }

    // Generates a new instance according to the distributions
    // return new instance
    private Instance drawInstance() {
        // Create a new instance
        Instance newInst = new DenseInstance(m_NumAttr);
        newInst.setDataset(queryExamples);

        // Get values for each attribute
        Enumeration enumAtt = newInst.enumerateAttributes();
        while (enumAtt.hasMoreElements()) {
            Attribute attr = (Attribute) enumAtt.nextElement();
            newInst.setValue(attr, setAttrValues(attr));
        }
        return newInst;
    }

    // Generate a random value for an attribute
    // return random value
    private double setAttrValues(Attribute attr) {
        long randomSeed = random.nextLong();
        return m_Distributions[attr.index()].newValue(randomSeed);
    }

    // get class label/ Generates class distribution at the node. Computes fidelity and bestfirst value
    public void setClassLabelNodo() {
        Enumeration instEnum = trainExamples.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            m_ClassDistribution[(int) inst.classValue()]++;
        }
        if (m_CountQueryEx > 0) {
            Enumeration instQEnum = queryExamples.enumerateInstances();
            while (instQEnum.hasMoreElements()) {
                Instance inst = (Instance) instQEnum.nextElement();
                m_ClassDistribution[(int) inst.classValue()]++;
            }
        }

        // Get main class at node
        Utils.normalize(m_ClassDistribution);
        m_ClassLabel = Utils.maxIndex(m_ClassDistribution);

        // Fidelity
        m_Fidelity = m_ClassDistribution[(int) m_ClassLabel];

        // Best First
        m_Best_First = m_Reach * (1 - m_Fidelity);
    }

    // Split node
    // Propc = prop instances that make a node to be a leaf
    // exception = if there is an error
    public void constructTest(double PropC) throws Exception {
        // If prop > PropC then make a leaf
        double prop = 1;

        if (m_CountTrainEx + m_CountQueryEx > 0) {
            prop = m_ClassDistribution[(int)m_ClassLabel] / (double) (m_CountQueryEx + m_CountTrainEx);
        }

        if (prop > PropC) {
            // Node -> Leaf
            makeALeaf();
        } else {
            // Internal node -> Split Node
            // Join Training Examples + QueryExamples to calculate values
            Instances data;
            if (m_CountQueryEx > 0) {
                data = new Instances(trainExamples);
                Enumeration enumInst = queryExamples.enumerateInstances();
                while (enumInst.hasMoreElements()) {
                    Instance inst = (Instance) enumInst.nextElement();
                    data.add(inst);
                }
            } else {
                data = trainExamples;
            }

            // compute attribute with maximun information gain
            double[] infoGains = new double[m_NumAttr];

            // Split values for numeric attributes
            attrValues = new double[m_NumAttr];

            // compute information gain for each attribute
            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeInfoGain(data, att);
            }

            // select best attribute for split
            m_Attribute = data.attribute(Utils.maxIndex(infoGains));

            // make leaf if information gain is zero
            // otherwise create successors
            if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
                makeALeaf();
            } else {
                splitNode(m_Attribute);
            }
        }
    }

    // split node generating children
    // att split attribute
    // exception if there is an error
    private void splitNode(Attribute attr) throws Exception {
        int numVal = 0;
        if (attr.isNominal()) {
            numVal = attr.numValues();
        } else {
            numVal = 2;
        }

        // Split data using selected attribute
        Instances[] splitData = splitData(trainExamples, attr, attrValues[attr.index()]);

        // Split query instances to compute Reach
        Instances[] splitQueryData = new Instances[numVal];
        if (m_CountQueryEx > 0) {
            splitQueryData = splitData(queryExamples, attr, attrValues[attr.index()]);
        }

        // Create child nodes
        m_Children = new TrepanNode[numVal];
        m_CountChildren = numVal;
        double reach;
        int n = 0;
        for (int j = 0; j < m_CountChildren; j++) {
            m_Children[j] = new TrepanNode(splitData[j]);
            m_Children[j].setParent(this);
            if (splitData[j].numInstances() > 0) {
                // Node is not a leaf
                m_Children[j].setNodeType("N");
                // Compute Reach
                if (m_CountQueryEx > 0) {
                    reach = m_Reach * ((splitData[j].numInstances() + splitQueryData[j].numInstances()) / (double) (m_CountTrainEx + m_CountQueryEx));
                } else {
                    reach = m_Reach * (splitData[j].numInstances() / (double) m_CountTrainEx);
                }
                m_Children[j].setReach(reach);
                n++;
            } else {
                // Node is leaf
                m_Children[j].setReach(0);
                m_Children[j].setFidelity(0);
                m_Children[j].setBestFirst(0);
                // Class label = parent's class label
                m_Children[j].setClassLabel(m_ClassLabel);
                double [] m_ChildDistribution = new double[m_NumClass];
                m_ChildDistribution[(int)m_ClassLabel] = 1;
                m_Children[j].setDistribution(m_ChildDistribution);
                m_Children[j].makeALeaf();
            }
        }

        // if only one child then make a leaf
        if (n <= 1) {
            makeALeaf();
        }
    }

    // Given an attribute compute information gain
    // data = the data for which information gain is to be computed
    // att = the attribute to bu used for splitting
    // return information gain
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = 0;
        if (att.isNumeric()) {
            // Information gain for numeric attributes
            infoGain = getMaxInfoGain(data, att);
        } else {
            // Compute entropy
            infoGain = computeEntropy(data);
            Instances[] splitData = splitData(data, att, 0);
            int k = att.numValues();

            // Compute Information Gain
            for (int j =0; j < k; j++) {
                int numInst = splitData[j].numInstances();
                if (numInst > 0) {
                    infoGain -= ((double) numInst / (double) data.numInstances()) * computeEntropy(splitData[j]);
                }
            }
        }
        return infoGain;
    }

    // Computes the entropy of a dataset
    // data = the data for which entropy is to be computed
    // return the entropy of the data's class distribution
    private double computeEntropy(Instances data) {
        double [] classCounts = new double[m_NumClass];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < m_NumClass; j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }

        int numInst = data.numInstances();

        entropy /= (double) numInst;
        return entropy + Utils.log2(numInst);
    }

    // Splits a dataset according to the values of a nominal or numeric attribute
    // data = the data which is to be split
    // att = the attribute to be used for splitting
    // value = is the split value for numeric attributes
    // return the sets of instances produced by the split
    private Instances[] splitData(Instances data, Attribute att, double value) {
        int k = 2;
        if (att.isNominal()) {
            k = att.numValues();
        }

        // Contains splits
        Instances[] splitData = new Instances[k];
        for (int j = 0; j < k; j++) {
            splitData[j] = new Instances(data, data.numInstances());
            splitData[j].delete();
        }

        if (att.isNominal()) {
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                splitData[(int) inst.value(att)].add(inst);
            }
        }

        if (att.isNumeric()) {
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                if (inst.value(att) <= value) {
                    splitData[0].add(inst);
                } else {
                    splitData[1].add(inst);
                }
            }
        }
        return splitData;
    }

    // Given a numeric attribute compute Information Gain
    // instances = the data for which information gain is to be computed
    // att = the attribute to be used for splitting
    // return information gain
    private double getMaxInfoGain(Instances instances, Attribute att) throws Exception {
        int i = 0;
        double actual = 0; // Attribute value
        double value = 0; // Split value
        double maxInfoGain = 0;
        double infoGain;
        double infoGainData;

        // Sort instances by attr
        Instances data = new Instances(instances);
        int numInst = data.numInstances();
        data.sort(att);

        // Split data for each attribute value
        Instances[] splitData;

        // Contains attribute values
        double[] attValues = new double[numInst];
        attValues = data.attributeToDoubleArray(att.index());
        // Contains class values
        double[] classValues = new double[numInst];
        classValues = data.attributeToDoubleArray(data.classAttribute().index());

        // Candidate values
        double[] candValues = new double[numInst];
        int k = 0; // Counter
        int pos = 0; // Counter

        while (k < attValues.length - 2) {
            // Find values where val(k) <> val(k+1) and class(k) <> class(k+1)
            if ((attValues[k] != attValues[k+1]) && (classValues[k] != classValues[k+1])) {
                candValues[pos] = (attValues[k] + attValues[k+1]) / 2;
                pos++;
            }
            k++;
        }

        infoGainData = computeEntropy(data);

        while (i < pos) {
            actual = candValues[i];

            // Split data using actual value
            splitData = splitData(data, att, actual);

            // Compute Information Gain
            infoGain = infoGainData;
            for (int j = 0; j < 2; j++) {
                if (splitData[j].numInstances() > 0) {
                    infoGain -= ((double) splitData[j].numInstances() / (double) numInst) * computeEntropy(splitData[j]);
                }
            }

            // Select max info gain
            if (infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                value = actual;
            }
            i++;
        }
        attrValues[att.index()] = value;
        return maxInfoGain;
    }

    // Make a node to be a leaf
    public void makeALeaf() {
        m_Attribute = null;
        m_NodeType = "L";
        m_CountChildren = 0;
        m_Children = null;
    }

    // Set and get methods

    // i = num of children
    public void setCountChildren(int i) {
        m_CountChildren = i;
    }

    // return num of children
    public int getCountChildren() {
        return m_CountChildren;
    }

    // set training examples
    public void setTrainExamples(Instances inst) {
        trainExamples = inst;
    }

    // return train examples
    public Instances getTrainExamples() {
        return trainExamples;
    }

    // query examples
    public void setQueryExamples(Instances inst) {
        queryExamples = inst;
    }

    public Instances getQueryExamples() {
        return queryExamples;
    }

    // class distribution
    public void setDistribution(double[] e) {
        m_ClassDistribution = e;
    }

    public double[] getDistribution() {
        return m_ClassDistribution;
    }

    // node's children
    public TrepanNode [] getChildren() {
        return m_Children;
    }

    public TrepanNode getChild(int i) {
        return m_Children[i];
    }

    // node type
    public void setNodeType(String s) {
        m_NodeType = s;
    }

    public String getNodeType() {
        return m_NodeType;
    }

    // parent node
    public void setParent(TrepanNode n) {
        m_Parent = n;
    }

    public TrepanNode getParent() {
        return m_Parent;
    }

    // split attribute
    public Attribute getSplitAttribute() {
        return m_Attribute;
    }

    // class label
    public double getClassLabel() {
        return m_ClassLabel;
    }

    public void setClassLabel(double c) {
        m_ClassLabel = c;
    }

    // return split value, att = Attribute
    public double getSplitValue(Attribute att) {
        if (att.isNumeric()) {
            return attrValues[att.index()];
        } else {
            return 0;
        }
    }

    // return fidelity
    public double getFidelity() {
        return m_Fidelity;
    }

    public void setFidelity(double d) {
        m_Fidelity = d;
    }

    // reach
    public double getReach() {
        return m_Reach;
    }

    public void setReach(double d) {
        m_Reach = d;
    }

    // bestfirst value
    public double getBestFirst() {
        return m_Best_First;
    }

    public void setBestFirst(double d) {
        m_Best_First = d;
    }

    // node id
    public int getIdNodo() {
        return m_IdNodo;
    }

    public void setIdNodo(int i) {
        m_IdNodo = i;
    }
}