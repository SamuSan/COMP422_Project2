package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class BackPropNeuralNet
{
    private Instances train = null;
    private Instances test = null;
    private MultilayerPerceptron mlp = null;
    private static final String[] options = {
            "-N","10000", "-L", "0.2","-H","3"
    };
    private static String optionsPrint ="";
    public BackPropNeuralNet( String trainFile, String testFile )
    {
        mlp = new MultilayerPerceptron();

        createTrainingInstances( trainFile );
        createTestInstances( testFile );
        try
        {
            optionsPrint = getCurrentOptions();
            mlp.setOptions( options );
            mlp.buildClassifier( train );
            System.out.println( "Done building Network" );
        }
        catch ( Exception e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private void createTrainingInstances( String f )
    {
        try
        {
            train = new Instances(
                    new BufferedReader(
                            new InputStreamReader(
                                    ClassLoader.getSystemResourceAsStream( f ) ) ) );
        }
        catch ( IOException e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        train.setClassIndex( train.numAttributes() - 1 );
    }

    private void createTestInstances( String f )
    {
        try
        {
            test = new Instances(
                    new BufferedReader(
                            new InputStreamReader(
                                    ClassLoader.getSystemResourceAsStream( f ) ) ) );
        }
        catch ( IOException e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        test.setClassIndex( train.numAttributes() - 1 );
    }

    public double testClassifier()
    {
        double total = 500;
        double correct = 0;
        try
        {
            for ( Instance i : test )
            {
                double pred = mlp.classifyInstance( i );
                if ( pred == i.classValue() )
                {
                    correct++;
                }
            }
        }
        catch ( Exception e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return ( correct / total );
    }
    private String getCurrentOptions(){
        String opt = "Epochs : " + options[1]+ "\n"
                    +"Learning rate : " + options[3]+ "\n";
        return opt;
    }
    
    public String opts(){
        return optionsPrint;
    }

}
