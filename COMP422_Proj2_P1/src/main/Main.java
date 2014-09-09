package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class Main
{
    private static final String[] tasks = { "data/digits_00", "data/digits_15", "data/digits_30", "data/digits_60" };

    public static void main( String[] args )
    {
        Instances train = null;
        try
        {
            train = new Instances(
                    new BufferedReader(
                            new InputStreamReader(
                                    ClassLoader.getSystemResourceAsStream( tasks[0] ) ) ) );
            train.setClassIndex( train.numAttributes() - 1 );
            
            IBk nn = new IBk();
            
            nn.buildClassifier( train );
            System.out.println( train.numAttributes() );
            System.out.println( train.classIndex() );
        }
        catch ( IOException e1 )
        {
            // TODO Auto-generated catch block
            e1.printStackTrace();
        }
        catch ( Exception e )
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }




        System.out.println( "Done" );
    }

}
