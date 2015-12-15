package multilayerperceptron;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Main {
    
    static final int dataSetSize = 440;
    
    static float[][] data = new float[dataSetSize][16];

    static float[][] testSet_1 = new float[88][16];
    static float[][] trainSet_1 = new float[352][16];
    
    static float[][] testSet_2 = new float[88][16];
    static float[][] trainSet_2 = new float[352][16];
    
    static float[][] testSet_3 = new float[88][16];
    static float[][] trainSet_3 = new float[352][16];
    
    static float[][] testSet_4 = new float[88][16];
    static float[][] trainSet_4 = new float[352][16];
    
    static float[][] testSet_5 = new float[88][16];
    static float[][] trainSet_5 = new float[352][16];
    
    static float[] Test_SSE = new float[5];
    static float[] Test_RMSE = new float[5];
    static float Average_Test_SSE, Average_Test_RMSE;
    
    static ArrayList<float[][]> testList = new ArrayList<>();
    static ArrayList<float[][]> trainList = new ArrayList<>();
    
    public static void main(String[] args) throws IOException {
        
        initData("forest.txt");
        initTestAndTrainSets();       
        
        for(int X=0; X<5; X++){

            MultiLayerPerceptron MLP = new MultiLayerPerceptron(testList.get(X), trainList.get(X));

            for(int epoch=0; epoch<10000; epoch++){
                //Train our network.
                MLP.forwardPass();
                MLP.backPropagation();
                MLP.epoch++;
            }

            MLP.testForwardPass();
            Test_SSE[X] = MLP.Test_SSE;
            Test_RMSE[X] = MLP.Test_RMSE;
            
            //System.out.printf("%d - %d\n", X, MLP.bestEpoch);
            //MLP.printResults();
            
            MLP.printBestWeights();
            System.out.println("\n\n\n\n");
            
        }

        Average_Test_SSE = (Test_SSE[0] + Test_SSE[1] + Test_SSE[2] + Test_SSE[3] + Test_SSE[4]) / 5;
        Average_Test_RMSE = (Test_RMSE[0] + Test_RMSE[1] + Test_RMSE[2] + Test_RMSE[3] + Test_RMSE[4]) / 5;



//        System.out.println("---------------------------------------------------------------------------------");
//        System.out.printf("SSEs:\t%f \t %f \t %f \t %f \t %f \t", Test_SSE[0],Test_SSE[1],Test_SSE[2],Test_SSE[3],Test_SSE[4]);
//        System.out.printf("\nRMSEs:\t%f \t %f \t %f \t %f \t %f \t", Test_RMSE[0],Test_RMSE[1],Test_RMSE[2],Test_RMSE[3],Test_RMSE[4]);
//        System.out.println("\n---------------------------------------------------------------------------------");
//        System.out.printf("Avg.SSE: %f\tAvg.RMSE:%f\n", Average_Test_SSE, Average_Test_RMSE);
//        System.out.println("---------------------------------------------------------------------------------\n\n");


    }
    
    public static void initData(String path) throws FileNotFoundException, IOException {
        FileInputStream fstream = new FileInputStream(path);
        BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

        String strLine = null;
        float[] buffer2;

        int counter = 0;
        for (int i = 0; i < dataSetSize; i++) {
            strLine = br.readLine();
            String[] buffer = strLine.split(",");

            buffer2 = new float[16];

            for (int j = 0; j < 16; j++) {
                buffer2[j] = Float.parseFloat(buffer[j]);
            }

            data[counter] = buffer2;
            counter++;
        }
    }
    
    public static void initTestAndTrainSets(){
        
        //CLASS 1
        for(int i=0; i<88; i++) testSet_1[i] = data[i];
        for(int i=88; i<440; i++) trainSet_1[i-88] = data[i];                
        
        //CLASS 2
        for(int i=0; i<88; i++) trainSet_2[i] = data[i];
        for(int i=88; i<176; i++) testSet_2[i-88] = data[i];
        for(int i=176; i<440; i++) trainSet_2[i-176] = data[i];
        
        //CLASS 3
        for(int i=0; i<176; i++) trainSet_3[i] = data[i];
        for(int i=176; i<264; i++) testSet_3[i-176] = data[i];
        for(int i=264; i<440; i++) trainSet_3[i-264] = data[i];
        
        //CLASS 4
        for(int i=0; i<264; i++) trainSet_4[i] = data[i];
        for(int i=264; i<352; i++) testSet_4[i-264] = data[i];
        for(int i=352; i<440; i++) trainSet_4[i-352] = data[i];
        
        //CLASS 5
        for(int i=0; i<352; i++) trainSet_5[i] = data[i];
        for(int i=352; i<440; i++) testSet_5[i-352] = data[i];
        
        trainList.add(trainSet_1);
        trainList.add(trainSet_2);
        trainList.add(trainSet_3);
        trainList.add(trainSet_4);
        trainList.add(trainSet_5);
        
        testList.add(testSet_1);
        testList.add(testSet_2);
        testList.add(testSet_3);
        testList.add(testSet_4);
        testList.add(testSet_5);
        
    }
}
