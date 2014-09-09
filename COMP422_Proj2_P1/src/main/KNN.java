package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KNN
{
    private Instances train = null;
    private Instances test = null;
    private IBk nn = null;

    public KNN( String trainFile, String testFile )
    {
        nn = new IBk();
        createTrainingInstances( trainFile );
        createTestInstances( testFile );
        try
        {
            nn.buildClassifier( train );
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

    private void classify() throws Exception
    {
        nn.buildClassifier( train );
    }
    private void testClassifier(){
        try
        {
            for ( Instance i : test)
            {
            double pred  = nn.classifyInstance( i );
            System.out.println(pred);
            }
        }
        catch ( Exception e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
