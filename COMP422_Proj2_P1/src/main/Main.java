package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class Main
{
    private static final String[] tasks = { "data/digits_00_train", "data/digits_00_test", 
                                            "data/digits_15_train", "data/digits_15_test", 
                                            "data/digits_30_train", "data/digits_30_test", 
                                            "data/digits_60_train", "data/digits_60_test" };

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
            

            System.out.println( train.numAttributes() );
            System.out.println( train.classIndex() );
            
            IBk nn = new IBk();
            
            nn.buildClassifier( train );
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
