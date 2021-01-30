package weka_classifiers;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class ClassificationRandomTree {
    public static void main (String[] args) {

        try {
            // load the data
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("data/zoo.arff");

            // create and import the instances
            Instances data = dataSource.getDataSet();

            Remove remove = new Remove();
            String[] removeOptions = new String[] {"-R", "1"};
            remove.setOptions(removeOptions);
            remove.setInputFormat(data);

            data = Filter.useFilter(data, remove);
//            System.out.println(data.toString());

            InfoGainAttributeEval gainAttributeEval = new InfoGainAttributeEval();
            Ranker rankerSearch = new Ranker();

            AttributeSelection attributeSelection = new AttributeSelection();
            attributeSelection.setEvaluator(gainAttributeEval);
            attributeSelection.setSearch(rankerSearch);
            attributeSelection.SelectAttributes(data);

            // RandomTree
            RandomForest randomTree = new RandomForest();
            String[] randomTreeOptions = new String[]{"-U"};
            randomTree.setOptions(randomTreeOptions);

            System.out.println(randomTree.getTechnicalInformation().toBibTex());

            // Initiate the learning process
            randomTree.buildClassifier(data);

            // lets visualize
//            TreeVisualizer tv = new TreeVisualizer(null, randomTree.toString(), new PlaceNode2());
//            JFrame frame = new JFrame("Random Tree Visualizer");
//            frame.setSize(1500, 1500);
//            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//            frame.getContentPane().add(tv);
//            frame.setVisible(true);
//            tv.fitToScreen();

            // RandomForest
            System.out.println("RandomForest Classification : ");
            System.out.println(randomTree.toString());
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

            DenseInstance unicornData = new DenseInstance(1.0, vals);
            unicornData.setDataset(data);
            double result = randomTree.classifyInstance(unicornData);
            System.out.println("With given data the classification obtained was : " + data.classAttribute().value((int) result));

            Classifier randomTreeClassifier = new RandomForest();
            Evaluation randomTreeEvaluation = new Evaluation(data);
            randomTreeEvaluation.crossValidateModel(randomTreeClassifier, data, 10, new Random(1), new Object[]{});
            System.out.println("RandomTree Classification Details : ");
            System.out.println(randomTreeEvaluation.toSummaryString());

            System.out.println("Confusion Matrix : ");
            randomTreeEvaluation.confusionMatrix();
            System.out.println(randomTreeEvaluation.toMatrixString());

        } catch (Exception exception) {
            exception.printStackTrace();
        }
    }
}
