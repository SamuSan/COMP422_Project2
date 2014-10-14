package main;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class Main
{
    private static final String[] tasks = { "data/digits_00_train", "data/digits_00_test",
            "data/digits_15_train", "data/digits_15_test",
            "data/digits_30_train", "data/digits_30_test",
            "data/digits_60_train", "data/digits_60_test" };
    private static final int[] kVals = {};

    public static void main( String[] args ) throws Exception
    {
 
                for ( int i = 0; i < 10; i++ )
                {
                    doKNN();
                    //doMLP();
                }
    }

    private static void doMLP() throws Exception
    {
        ArrayList<Double> results = new ArrayList<Double>();
        String params = "";
        int num = 1;
        for ( int i = 0; i < tasks.length; i += 2 )
        {
            BackPropNeuralNet bpNN = new BackPropNeuralNet( tasks[i], tasks[i + 1] );
            results.add( bpNN.testClassifier() );
            if( params.length() == 0 ){
                params = bpNN.opts();
            }

            System.out.println(params);
        }
        for ( Double res : results )
        {
            System.out.println( res );
        }
        recordResults( results, "MLP", params );
        System.out.println( "NN Done" );

    }

    private static void doKNN() throws Exception
    {
        ArrayList<Double> results = new ArrayList<Double>();
        for ( int i = 0; i < tasks.length; i += 2 )
        {
            KNN knn = new KNN( tasks[i], tasks[i + 1] );
            knn.setOptions( i );
            results.add( knn.testClassifier() );

        }
        for ( Double res : results )
        {
            System.out.println( res );
        }
        System.out.println( "KNN Done" );
    }

    private static void recordResults( ArrayList<Double> results, String clazzifier, String params ) throws IOException
    {
        DateFormat dateFormat = new SimpleDateFormat( "yyyy/MM/dd HH:mm:ss" );
        Date date = new Date();
        System.out.println( dateFormat.format( date ) ); //2014/08/06 15:59:48
        FileWriter write = new FileWriter( "output.txt", true );
        write.append( "\n" + clazzifier + "\n" );
        write.append( "\nRun Date / Time\n" + dateFormat.format( date ) );
        write.append( "\nPercentage \t Accuracy\n" );
        write.append( "0% Noise \t\t" + results.get( 0 ).toString() + "\n" );
        write.append( "15% Noise \t\t" + results.get( 1 ).toString() + "\n" );
        write.append( "30% Noise \t\t" + results.get( 2 ).toString() + "\n" );
        write.append( "60% Noise \t\t" + results.get( 3 ).toString() + "\n" );
        write.append("Parameters :  \n");
        write.append( params );
        write.close();
    }
}
