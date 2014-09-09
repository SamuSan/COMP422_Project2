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

    public void setOptions( int set )
    {
        int k = 1;
        switch ( set )
        {
            case 0 :
                break;
            case 2 :
                k = 3;
                break;
            case 4 :
                k = 4;
                break;
            case 6 :
                k = 5;
                break;
        }
                nn.setKNN( k );
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

    public void classify() throws Exception
    {
        nn.buildClassifier( train );
    }

    public double testClassifier()
    {
        double total = 500;
        double correct = 0;
        try
        {
            for ( Instance i : test )
            {
                double pred = nn.classifyInstance( i );
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
}
