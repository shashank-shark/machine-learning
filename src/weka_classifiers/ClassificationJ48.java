package weka_classifiers;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.util.Random;

public class ClassificationJ48 {
    public static void main (String[] args) {
        try {
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("data/zoo.arff");
            Instances data = dataSource.getDataSet();

            // here we are about to remove the animal attribute, which is the first attribute
            Remove remove = new Remove(); String[] options = new String[] {"-R", "1"};
            remove.setOptions(options); remove.setInputFormat(data);

            data = Filter.useFilter(data, remove);

            // get the best attributes (features) for our model training
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker rankerSearch = new Ranker();

            AttributeSelection attributeSelection = new AttributeSelection();
            attributeSelection.setEvaluator(eval);
            attributeSelection.setSearch(rankerSearch);
            attributeSelection.SelectAttributes(data);

            int[] indices = attributeSelection.selectedAttributes();
            Instances finalData = data;
//            Arrays.stream(indices).forEach(x -> System.out.println(finalData.attribute(x)));

            // now after knowing the ranks of attributes that need to be selected, we will go for J48 tree impl
            J48 decisionTree = new J48();
            String[] treeOptions = new String[]{"-U"};
            decisionTree.setOptions(treeOptions);

            // initialize the learning process
            decisionTree.buildClassifier(data);
//            System.out.println(decisionTree.toString());

            TreeVisualizer tv = new TreeVisualizer(null, decisionTree.graph(), new PlaceNode2());
            JFrame frame = new JFrame("Tree Visualizer");
            frame.setSize(800, 500);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.getContentPane().add(tv);
            frame.setVisible(true);
            tv.fitToScreen();

            // now let's classify a sea-horse
            double[] vals = new double[data.numAttributes()];
            vals[0] = 1.0; //hair {false, true}
            vals[1] = 0.0;  //feathers {false, true}
            vals[2] = 0.0;  //eggs {false, true}
            vals[3] = 1.0;  //milk {false, true}
            vals[4] = 0.0;  //airborne {false, true}
            vals[5] = 0.0;  //aquatic {false, true}
            vals[6] = 0.0;  //predator {false, true}
            vals[7] = 1.0;  //toothed {false, true}
            vals[8] = 1.0;  //backbone {false, true}
            vals[9] = 1.0;  //breathes {false, true}
            vals[10] = 1.0;  //venomous {false, true}
            vals[11] = 0.0;  //fins {false, true}
            vals[12] = 4.0;  //legs INTEGER [0,9]
            vals[13] = 1.0;  //tail {false, true}
            vals[14] = 1.0;  //domestic {false, true}
            vals[15] = 0.0;  //catsize {false, true}
            DenseInstance myUnicorn = new DenseInstance(1.0, vals);
            myUnicorn.setDataset(data);

            double result = decisionTree.classifyInstance(myUnicorn);
            System.out.println(data.classAttribute().value((int) result));

            // now lets check the accuracy of our model -
            Classifier classifier = new J48();
            Evaluation classifierEvaluation = new Evaluation(data);
            classifierEvaluation.crossValidateModel(classifier, data, 10, new Random(1), new Object[]{});
            System.out.println(classifierEvaluation.toSummaryString());

            // Now let's see where the misclassification is made using confusion matrix
            double[][] classifierConfusionMatrix = classifierEvaluation.confusionMatrix();
            System.out.println(classifierEvaluation.toMatrixString());

        } catch (Exception exception) {
            exception.printStackTrace();
        }
    }
}
