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
        KNN knn = new KNN(tasks[0], tasks[1]);



        System.out.println( "Done" );
    }

}
