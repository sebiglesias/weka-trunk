// Trepan

package weka.classifiers.trepan;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.classifiers.*;
import weka.core.Queue;
import weka.estimators.*;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.filters.*;

// class for running Trepan classifier

public class Trepan extends DistributionClassifier implements OptionHandler {
    // class atribute of dataset
    private Attribute m_ClassAttribute;
    // Used to set the max number of nodes allowed in the tree
    private int m_maxNodes = 0;
    // Used to set the min number of examples required to test a node
    private int m_minSamples = 0;
    // Proportion of instances belonging to the same class that makes the node a leaf
    private double m_propInst = 0.95;
    // Oracle used to classify instances
    protected Classifier m_Oracle = new weka.classifiers.neural.NeuralNetwork();
    // Instances classified by the Oracle
    private Instances newData;
    // Queue of leaves
    private Queue m_Queue;
    // Construct the tree using the best nodes
    private boolean m_BestFirst = false;
    // Prune the tree
    private boolean m_Pruning = false;
    // Root node
    private TrepanNode m_RootNode;
    // Filter to replace missing values
    private Filter m_Filter;
    // Counters
    private int m_CountNodes = 0;
    private int m_CountLeaves = 0;
    // Tree fidelity
    private double m_Fidelity;

    // Builds Trepan decision tree classifier
    // data = training data
    // exception if classififer can't be built succesfully
    public void buildClassifier(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            throw new IllegalArgumentException("No training instances.");
        }
        if (!data.classAttribute().isNominal()) {
            throw new Exception("Trepan: nominal class please");
        }
        if (data.checkForStringAttributes()) {
            throw new Exception("Trepan: no string attributes please");
        }
        data = new Instances(data);

        // Class attribute
        m_ClassAttribute = data.classAttribute();
        // Build Oracle's Classifier
        m_Oracle.buildClassifier(data);

        // Replace missing values
        m_Filter = new ReplaceMissingValuesFilter();
        m_Filter.inputFormat(data);
        Instances. filteredData = Filter.useFilter(data, m_Filter);
        data = new Instances(filteredData);
        // Classify data with the Oracle
        newData = new Instances(classifyWithOracle(data));
        // Make Trepan Tree with new data
        makeTree(newData);
    }

    // Check missing values in data
    // data = dataset
    // return true if data contains missing values
    private boolean withMissingValues(Instances data) {
        boolean missing = false;
        Enumeration enumAtt = data.enumerateAttributes();
        while (enumAtt.hasMoreElements() && !missing) {
            Attribute attr = (Attribute) enumAtt.nextElement();
            Enumeration enumeration = data.enumerateInstances();
            while (enumeration.hasMoreElements() && !missing) {
                if (((Instance) enumeration.nextElement()).isMissing(attr)) {
                    missing = true;
                }
            }
        }
        return missing;
    }

    // Replace the instance's classes with the class predicted by the oracle
    // data = training data
    // exception = if there is an error
    // return data classified with the oracle
    private Instances classifyWithOracle(Instances data) throws Exception {
        Instance newInstance;
        double newClassValue;
        Instances newData = new Instances(data, data.numInstances());

        Enumeration enumeration = data.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            newInstance = (Instance) enumeration.nextElement();
            newClassValue = m_Oracle.classifyInstance(newInstance);
            newInstance.setClassValue(newClassValue);
            newData.add(newInstance);
        }
        return newData;
    }

    // Build trepan tree
    // data = training data
    // exception if decision tree can't be built successfully
    private void makeTree(Instances data) throws Exception {
        // Inicialize the root node
        m_RootNode = new TrepanNode(data);
        final String r = "R";
        m_RootNode.setNodeType(r);
        m_RootNode.setReach(1);
        // Create new instances
        m_RootNode.drawSample(m_minSamples, m_Oracle);

        // Set node's main class
        m_RootNode.setClassLabelNodo();

        // Inicialize queue inserting the root node
        Queue m_Queue = new Queue();
        m_Queue.push(m_RootNode);

        // Make the Trepan Tree
        TrepanNode m_Node;

        // Num of nodes in the tree
        int m_NumNodes = 0;

        // Max nodes in the tree
        int m_LongTree = 0;
        // If Best First -> we need to make the tree and then prune it
        // Else we generate the tree pruned
        if (!m_BestFirst && m_maxNodes > 0) {
            m_LongTree = m_maxNodes;
        } else {
            m_LongTree = 99999;
        }

        //Expand root node to make decision Tree
        while ((!m_Queue.empty()) && m_NumNodes <= m_LongTree) {
            // Get node from queue
            m_Node = (TrepanNode) m_Queue.pop();
            // Set node ID
            m_Node.setIdNodo(m_NumNodes);
            m_NumNodes ++;
            // Split node
            m_Node.constructTest(m_propInst);
            // Get node's children
            TrepanNode [] m_ChildNode = m_Node.getChildren();
            for (int j =0; j < m_Node.getCountChildren(); j++) {
                if (!(localStopCriteria(m_ChildNode[j]))) {
                    // Node is not a leaf, add children in queue
                    m_ChildNode[j].drawSample(m_minSamples, m_Oracle);
                    m_ChildNode[j].setClassLabelNodo();
                    m_Queue.push(m_ChildNode[j]);
                } else {
                    // Node is leaf
                    m_ChildNode[j].setIdNodo(m_NumNodes);
                    m_NumNodes++;
                }
            }
        }

        // if max nodes reached then make leafs with remaining nodes
        int i = m_NumNodes;
        while (!m_Queue.empty()) {
            m_Node = (TrepanNode) m_Queue.pop();
            m_Node.setIdNodo(i);
            m_Node.makeALeaf();
            i++;
        }

        // If best first then choose the best nodes
        if (m_maxNodes > 0) {
            if (m_NumNodes > m_maxNodes) {
                if (m_BestFirst) {
                    // Make BestFirst tree
                    GetBestTree(m_NumNodes);
                }
            }
        }

        // Prune the tree to eliminate redundant nodes
        if (m_Pruning) {
            pruneTree();
        }

        // Compute Fidelity between the tree and the oracle
        m_Fidelity = computerFidelity(newData);
    }

    // Indicates if node must be expanded
    // node = node to be evaluated
    // return true if node is a leaf
    private boolean localStopCriteria(TrepanNode node) {
        if (node.getNodeType() == "L") {
            return true;
        } else {
            return false
        }
    }

    // Make a new pruned tree using the best nodes
    // m_NumNodes = num of nodes of the original tree
    private void GetBestTree(int m_NumNodes) {
        // Vector containing all the tree's nodes
        TrepanNode[] vNodos = new TrepanNode[m_NumNodes];
        cargarVectorNodos(m_RootNode, vNodos);

        // Vector containing the best nodes
        int[] vValidos = new int[m_maxNodes];
        // Vector containing BestFirst values
        double[] vBF = new double[m_NumNodes];
        // ID valid node
        int ID;
        // Counts best nodes
        int i = 0;
        // Aux variable
        int IdNodo;

        // Inicialize vector BF
        for (int j =0; j < m_NumNodes; j++) {
            vBF[j] = 0;
        }

        // Root node is always a Best Node
        ID = m_RootNode.getIdNodo();
        vValidos[i] = ID;
        i++;

        while (i < m_maxNodes) {
            // Insert child nodes into vNodos
            for (int k = 0; k < vNodos[ID].getCountChildren(); k++) {
                IdNodo = vNodos[ID].getChild(k).getIdNodo();
                vBF[IdNodo] = vNodos[ID].getChild(k).getBestFirst();
            }

            // Choose the Best Node
            ID = Utils.maxIndex(vBF);
            vValidos[i] = ID;
            i++;

            // Eliminate the node selected
            vBF[ID] = -1;
        }

        // Make leafs with the remaining nodes
        Queue m_Queue = new Queue();
        m_Queue.push(m_RootNode);

        TrepanNode m_Node;
        while (!m_Queue.empty()) {
            m_Node = (TrepanNode) m_Queue.pop();
            // Check if the node is a Best Node
            if (NodoEncontrado(m_Node.getIdNodo(), vValidos) == false) {
                // If node is not a best node then make a leaf
                m_Node.makeALeaf();
            } else {
                // Is a BF Node. Insert children into queue
                if (m_Node.getNodeType() != "L") {
                    TrepanNode [] m_ChildNode = m_Node.getChildren();
                    for (int m = 0; m < m_Node.getCountChildren(); m++) {
                        m_Queue.push(m_ChildNode[m]);
                    }
                }
            }
        }
    }

    // Check if a node is found in a vector
    // idNodo = node to be found
    // return true if the node is found
    private boolean NodoEncontrado(int idNodo, int[] vect) {
        boolean bfound = false;
        int i = 0;
        while (!bfound && i < m_maxNodes) {
            if (vect[i] == idNodo) {
                bfound = true;
            }
            i++;
        }
        return bfound;
    }

    // Insert all the tree's nodes into a vector (recursive)
    // m_Node node to be inserted
    // vNodos vector
    private void cargarVectorNodos(TrepanNode m_Node, TrepanNode[] vNodos) {
        vNodos[m_Node.getIdNodo()] = m_Node;
        if (m_Node.getNodeType() != "L") {
            TrepanNode [] m_ChildNode = m_Node.getChildren();
            for (int j = 0; j < m_Node.getCountChildren(); j++) {
                cargarVectorNodos(m_ChildNode[j], vNodos);
            }
        }
    }

    // Prune the tree
    private void pruneTree() {
        boolean again = true;
        Queue m_Queue = new Queue();

        TrepanNode m_Node;
        double Class;
        boolean IdemClass;
        boolean AllLeaves;

        while (again) {
            again = false;
            m_Queue.push(m_RootNode);

            while (!m_Queue.empty()) {
                m_Node = (TrepanNode) m_Queue.pop();

                if (m_Node.getNodeType() != "L") {
                    // Get node's children
                    TrepanNode [] m_ChildNode = m_Node.getChildren();

                    // Check if all the children are leafs
                    AllLeaves = true;
                    for (int i = 0; i < m_Node.getCountChildren(); m++) {
                        if (m_ChildNode[m].getClassLabel() != Class) {
                            IdemClass = false;
                        }
                    }
                    if (IdemClass) {
                        // Make parent node a leaf
                        m_Node.setClassLabel(Class);
                        m_Node.makeALeaf();
                        again = true;
                    }
                } else {
                    for (int m = 0; m < m_Node.getCountChildren(); m++) {
                        m_Queue.push(m_ChildNode[m]);
                    }
                }
            }
        }
    }

    // Method computing the fidelity of the classifier with the Oracle
    // data = training data
    // exception if there is an error
    // return the fidelity
    public double computerFidelity(Instances data) throws Exception {
        Instance instance;
        double classValue;
        double classValueOracle;
        double countFidelity = 0;

        Enumeration enumeration = data.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            instance = (Instance) enumeration.nextElement();
            // Clasify instance with Tree
            classValue = classifyInstance(instance);
            // Classify instance with ORacle
            classValueOracle = m_Oracle.classifyInstance(instance);
            if (classValue == classValueOracle) {
                countFidelity++;
            }
        }
        // Compute Fidelity
        return (double)((double)countFidelity / (double) data.numInstances());
    }

    // Classifies a given test instance using the decision tree
    // instance = instance to be classified
    // return the classification
    public double classifyInstance(Instance instance) throws Exception {
        // Replace missing values
        m_Filter.input(instance);
        m_Filter.batchFinished();
        instance = m_Filter.output();
        return classifyTrepanInstance(m_RootNode, instance);
    }

    // Classifies a given test instance using the TREPAN tree
    // node = root node
    // instance = instance to be classified
    // return the classification
    private double classifyTrepanInstance(TrepanNode node, Instance instance) {
        Attribute m_SplitAttribute = node.getSplitAttribute();
        if (node.getSplitAttribute() == null) {
            return node.getClassLabel();
        } else {
            // Get the child node according to attribute value
            double dValue = instance.value(m_SplitAttribute);
            TrepanNode childNode;
            if (m_SplitAttribute.isNominal()) {
                childNode = node.getChild((int) dValue);
            } else {
                double splitValue = node.getSplitValue(m_SplitAttribute);
                if (dValue <= splitValue) {
                    childNode = node.getChild(0);
                } else {
                    childNode = node.getChild(1);
                }
            }
            return classifyTrepanInstance(childNode, instance);
        }
    }

    // Computes class distribution for instance using decision tree
    // instance = instance for which distribution is to be computed
    // return the class distribution for the given instance
    public double[] distributionForInstance(Instance instance) throws Exception {
        // Replace missing values
        m_Filter.input(instance);
        m_Filter.batchFinished();
        instance = m_Filter.output();

        return distributionForTrepanInstance(m_RootNode, instance);
    }

    // Computes class distribution for instance using TREPAN tree
    // node = root node
    // instance = instance for which distribution is to be computed
    // return the class distribution for the given instance
    private double[] distributionForTrepanInstance(TrepanNode node, Instance instance) {
        Attribute m_Attribute = node.getSplitAttribute();
        if (m_Attribute == null) {
            double[] classMembership = node.getDistribution();
            return node.getDistribution();
        } else {
            // Get the child node according to attribute value
            double dValue = instance.value(m_Attribute);
            TrepanNode childNode;
            if (m_Attribute.isNominal()) {
                childNode = node.getChild((int) dValue);
            } else {
                double splitValue = node.getSplitValue(m_Attribute);
                if (dValue <= splitValue) {
                    childNode = node.getChild(0);
                } else {
                    childNode = node.getChild(1);
                }
            }
            return distributionForTrepanInstance(childNode, instance);
        }
    }

    // prints the decision tree using the private toString method below
    // return textual description of the classifier
    public String toString() {
        m_CountLeaves = 0;
        m_CountNodes = 0;

        if ((m_RootNode.getDistribution() == null) && (m_RootNode.getChildren() == null)) {
            return "Trepan: No model built yet";
        }

        StringBuffer text = new StringBuffer();
        text.append("Trepan\n\n" + toString(m_RootNode, 0));

        text.append("\n\nNumber of Leaves: \t" + getCountLeaves() + "\n");
        text.append("\nSize of the tree: \t" + getCountNodes() + "\n");

        return text.toString();
    }

    private String toString(TrepanNode node, int level) {
        StringBuffer text = new StringBuffer();
        Attribute m_Attribute = node.getSplitAttribute();
        double dClass = node.getClassLabel();

        if (m_Attribute == null) {
            m_CountLeaves ++;
            if (Instance.isMissingValue(dClass)) {
                text.append(": null");
            } else {
                text.append(": " + m_ClassAttribute.value((int) dClass));
            }
        } else {
            m_CountNodes++;
            for (int j =0; j < node.getCountChildren(); j++) {
                text.append("\n");
                for (int i =0; i < level; i++) {
                    text.append("| ");
                }
                if (m_Attribute.isNominal()) {
                    text.append(m_Attribute.name() + " = " + m_Attribute.value(j)
                            + " (R= " + Utils.roundDouble(node.getChild(j).getReach(), 3) + ","
                            + " (F= " + Utils.roundDouble(node.getChild(j).getFidelity(), 3) + ","
                            + " (B= " + Utils.roundDouble(node.getChild(j).getBestFirst(), 3) + ")");
                } else {
                    if (j == 0) {
                        text.append(m_Attribute.name() + "\t\t\t<=\t\t\t"
                                + Utils.roundDouble(node.getSplitValue(m_Attribute), 5)
                                + " (R= " + Utils.roundDouble(node.getChild(j).getReach(), 3) + ","
                                + " (F= " + Utils.roundDouble(node.getChild(j).getFidelity(), 3) + ","
                                + " (B= " + Utils.roundDouble(node.getChild(j).getBestFirst(), 3) + ")");
                    } else {
                        text.append(m_Attribute.name() + "\t\t\t>\t\t\t"
                                + Utils.roundDouble(node.getSplitValue(m_Attribute), 5)
                                + " (R= " + Utils.roundDouble(node.getChild(j).getReach(), 3) + ","
                                + " (F= " + Utils.roundDouble(node.getChild(j).getFidelity(), 3) + ","
                                + " (B= " + Utils.roundDouble(node.getChild(j).getBestFirst(), 3) + ")");
                    }
                }
                TrepanNode childNode = node.getChild();
                text.append(toString(childNode, level + 1));
            }
        }
        return text.toString();
    }

    // Set & Get Methods

    // Max number of nodes in the tree
    public void setMaxNodes(int i) {
        if (i >= 0) {
            m_maxNodes = i;
        }
    }

    // return max number of nodes in the tree
    public int getMaxNodes() {
        return m_maxNodes;
    }

    // min number of examples to test a node
    public void setMinSamples(int i) {
        if (i >= 0 ) {
            m_minSamples = i;
        } else {
            m_minSamples = 0;
        }
    }

    // return min number of examples to test a node
    public int getMinSamples() {
        return m_minSamples;
    }

    // return data classified with oracle
    public Instances getDataNN() {
        return newData;
    }

    //return num of leaves of the tree
    public int getCountLeaves() {
        return m_CountLeaves;
    }

    // return num of nodes of the tree
    public int getCountNodes() {
        return m_CountNodes;
    }

    // return trepan's oracle
    public Classifier getOracle() {
        return m_Oracle;
    }

    // classifier used as the oracle
    public void setOracle(Classifier c) {
        m_Oracle = c;
    }

    // best first value
    public void setBestFirst(boolean b) {
        m_BestFirst = b;
    }

    // return best first value
    public boolean getBestFirst() {
        return m_BestFirst;
    }

    //pruning value
    public void setPrunning(boolean b) {
        m_Pruning = b;
    }

    public boolean getPrunning(){
        return m_Pruning;
    }

    // prop of instances to make a leaf
    public void setPropInstances(double i) {
        m_propInst = i;
    }

    // prop of instances to make a leaf
    public double getPropInstances() {
        return m_propInst;
    }

    // return fidelity
    public double getFidelity() {
        return m_Fidelity;
    }

    // Option handler

    // Returns an enumeration describing the available options
    // return an enumeration of all the available options
    public Enumeration listOptions() {
        Vector newVector = new Vector(6);
        newVector.addElement(new Option("\tMin number of samples to test a node.\n" + "\t(Default = 100).", "m", 1, "-m <number of samples>"));
        newVector.addElement(new Option("\tMax number of nodes in tree.\n" + "\t(Default=10).", "M", 1, "-M <number of nodes>"));
        newVector.addElement(new Option("\tBest tree.", "B", 0, "-B"));
        newVector.addElement(new Option("\tPrunning.", "P", 0, "-P"));
        newVector.addElement(new Option("\tProp. Inst.", "I", 0, "-I"));
        newVector.addElement(new Option("\tFull name of classifier to use as oracle." + "\tNot Optional." + "\teg: weka.classifiers.neural.NeuralNetwork", "W", 1, "-W <classifier class name>"));

        if ((m_Oracle != null) && (m_Oracle instanceof OptionHandler)) {
            newVector.addElement(new Option("", "", 0, "\nOptions specific to sub-classifier" + m_Oracle.getClass().getName() + ":\n(use -- to signal start of sub-classifier options)"));
            Enumeration enumeration = ((OptionHandler)m_Oracle).listOptions();
            while (enumeration.hasMoreElements()) {
                newVector.addElement(enumeration.nextElement());
            }
        }
        return newVector.elements();
    }

    // Parse a given list of options
    // options = the list of options as an array of strings
    // exception = if an option is not supported
    public void setOptions(String[] options) throws Exception {
        String minSampleString = Utils.getOption('m', options);
        if (minSampleString.length() != 0) {
            setMinSamples(Integer.parseInt(minSampleString));
        } else {
            setMinSamples(0);
        }
        String maxNodesString = Utils.getOption('M', options);
        if (maxNodesString.length() != 0) {
            setMaxNodes(Integer.parseInt(maxNodesString));
        } else {
            setMaxNodes(0);
        }

        m_BestFirst = Utils.getFlag('B', options);
        m_Pruning = Utils.getFlag('P', options);

        String propInstances = Utils.getOption('I', options);
        if (propInstances.length() != 0) {
            double prop = (new Double(propInstances)).doubleValue();
            if (prop <= 0 || prop > 1) {
                setPropInstances(0.95);
            } else {
                setPropInstances(prop);
            }
        } else {
            setPropInstances(0.95);
        }

        String classifierName = Utils.getOption('W', options);
        if (classifierName.length() == 0) {
            throw new Exception("A classifier must be specified with the -W option");
        }

        setOracle(Classifier.forName(classifierName, Utils.partitionOptions(options)));
    }

    // Gets the current settings of trepan
    // return an array of strings suitable for passing to setOptions()
    public String[] getOptions() {
        String [] classifierOptions = new String[0];
        if ((m_Oracle != null) && (m_Oracle instanceof OptionHandler)) {
            classifierOptions = ((OptionHandler) m_Oracle).getOptions();
        }

        String[] options = new String[classifierOptions.length + 11];
        int current = 0;
        options[current++] = "-m";
        options[current++] = "" + getMinSamples();
        options[current++] = "-M";
        options[current++] = "" + getMaxNodes();
        if (m_BestFirst) {
            options[current++] = "-B"
        }
        if (m_Pruning) {
            options[current++] = "-P";
        }
        options[current++] = "-I";
        options[current++] = "" + getPropInstances();
        options[current++] = "-W";
        options[current++] = getOracle().getClass().getName();
        options[current++] = "--";

        System.arraycopy(classifierOptions, 0, options, current, classifierOptions.length);
        current += classifierOptions.length;

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    //main method
    public static void main(String[] args) {
        try {
            System.out.println(Evaluation.evaluateModel(new Trepan(), args));
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
