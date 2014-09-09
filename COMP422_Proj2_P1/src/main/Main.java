package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class Main
{
    private static final String[] tasks = { "data/digits_00_train", "data/digits_00_test",
            "data/digits_15_train", "data/digits_15_test",
            "data/digits_30_train", "data/digits_30_test",
            "data/digits_60_train", "data/digits_60_test" };
    private static final int[] kVals = {};

    public static void main( String[] args ) throws Exception
    {
//        doKNN();
        doMLP();
    }

    private static void doMLP() throws Exception
    {
        ArrayList<Double> results = new ArrayList<Double>();
        int num =1;
        for ( int i = 0; i < tasks.length; i += 2 )
        {
            BackPropNeuralNet bpNN = new BackPropNeuralNet( tasks[i], tasks[i + 1] );
            results.add( bpNN.testClassifier() );

        }
        for ( Double res : results )
        {
            System.out.println( res );
        }
        System.out.println( "NN Done" );

    }

    private static void doKNN() throws Exception
    {
        ArrayList<Double> results = new ArrayList<Double>();
        for ( int i = 0; i < tasks.length; i += 2 )
        {
            KNN knn = new KNN( tasks[i], tasks[i + 1] );
            knn.setOptions( i );
//            knn.classify();
            results.add( knn.testClassifier() );

        }
        for ( Double res : results )
        {
            System.out.println( res );
        }
        System.out.println( "KNN Done" );
    }

}
