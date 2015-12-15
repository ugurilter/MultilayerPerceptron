package multilayerperceptron;

import java.util.Random;

public class MultiLayerPerceptron {
    
    Random RNG;
    float learningRate = 0.45F, momentum = 0.60F;
    int inputNodeCount = 15, hiddenNodeCount = 13;
        
    InputNode[] inputLayer = new InputNode[inputNodeCount];
    HiddenNode[] hiddenLayer = new HiddenNode[hiddenNodeCount];
    OutputNode outputNode = new OutputNode(0);
    
    float[][] testSet = new float[88][16];
    float[][] trainSet = new float[352][16];
    float[] trainOutputs = new float[352];
    float[] testOutputs = new float[88];
    
    float[][] I_H_weights, BEST_I_H_weights = new float[inputNodeCount][hiddenNodeCount];  
    float[] H_O_weights, BEST_H_O_weights = new float[hiddenNodeCount];  
    
    float Test_SSE, Train_SSE, BEST_Train_SSE=999999999;
    float Test_RMSE, Train_RMSE, BEST_Train_RMSE=999999999;
    int bestEpoch=0, epoch=0;
    
    float randVal = 0.50F;
    float biasVal = 4.35F;


    public MultiLayerPerceptron(float[][] test, float train[][]) {
        this.testSet = test;
        this.trainSet = train;
        RNG = new Random(100);      //GIVING SEED TO RNG TO GET SAME RANDOM VALUES WHEN WE RUN THE PROGRAM EACH TIME. (IN ORDER TO GET RID OF CONFUSION)
        initInputLayer();
        initHiddenLayer();
        initIH_Weights();
        initHO_Weights();
        initBiases();
    }
    
    public void initBiases()    {
        for(int i=0; i<hiddenNodeCount; i++){
            hiddenLayer[i].setBias(biasVal);       //INITIAL HIDDEN LAYER BIAS VALUES := 0.3
        }
        outputNode.setBias(biasVal);               //INITIAL OUTPUT NODE BIAS VALUE := 0.2
    }

    public void initInputLayer(){
        inputLayer = new InputNode[inputNodeCount];
        for(int i=0; i<inputNodeCount; i++) inputLayer[i] = new InputNode(0);
    }
    
    public void initHiddenLayer(){
        hiddenLayer = new HiddenNode[hiddenNodeCount];
        for(int i=0; i<hiddenNodeCount; i++) hiddenLayer[i] = new HiddenNode(0);
    }
    
    public void initIH_Weights(){
        I_H_weights = new float[inputNodeCount][hiddenNodeCount];
        
        for(int i=0; i<inputNodeCount; i++){
            for(int j=0; j<hiddenNodeCount; j++){
                I_H_weights[i][j] = (RNG.nextFloat() - randVal);    //INITIAL INPUT-HIDDEN LAYER WEIGHTS (0.5, -0.5)
            }
        }

    }
    
    public void initHO_Weights(){
        H_O_weights = new float[hiddenNodeCount];
        
        for(int j=0; j<hiddenNodeCount; j++){
                H_O_weights[j] = (RNG.nextFloat() - randVal);       //INITIAL HIDDEN-OUTPUT LAYER WEIGHTS (0.5, -0.5)
        }
    }
   
    public void forwardPass(){
        
        for(int i=0; i<352; i++){
            
            //TAKE ONE ROW OF INPUTS FROM "trainSet[][]".
            takeInput(i);
            
            //CALCULATE OUTPUTS OF HIDDEN LAYER NODES.
            calculateHiddenOutputs();
            
            //CALCULATE OUTPUT AND STORE IT.            
            outputNode.setValue(calculateOutput());
            trainOutputs[i] = outputNode.getValue();
            
        }
        
        //SET CURRENT SSE & RMSE.
        Train_SSE = Train_SSE();
        Train_RMSE = Train_RMSE();
        
        //IF BEST RMSE (FOR TRAINING!) IS BIGGER THAN CURRENT (TRAININING!) RMSE; (MEANS IT'S WORSE), SAVE WEIGHTS AND SSE-RMSE VALUES.
        if(BEST_Train_RMSE > Train_RMSE){
            BEST_Train_SSE = Train_SSE;
            BEST_Train_RMSE = Train_RMSE();
            saveBestWeights();
            bestEpoch = epoch;
        }
        
    }
    
    public void saveBestWeights(){
        //SAVE INPUT-HIDDEN WEIGHTS.
        for(int i=0; i<inputNodeCount; i++)
            System.arraycopy(I_H_weights[i], 0, BEST_I_H_weights[i], 0, hiddenNodeCount);  
        
        //SAVE HIDDEN-OUTPUT WEIGHTS.
        System.arraycopy(H_O_weights, 0, BEST_H_O_weights, 0, hiddenNodeCount);
        
        //SAVE BIASES.
        for(int i=0; i<hiddenNodeCount; i++) hiddenLayer[i].setBestBias(hiddenLayer[i].getBias());        
        outputNode.setBestBias(outputNode.getBias());
    }
    
    public float Train_SSE(){
        //SUM OF SQUARED ERRORS FOR TRAINING.
        float result = 0;
        float buffer = 0;
        for(int i=0; i<352; i++){
            buffer = (denormalize(trainSet[i][15]) - denormalize(trainOutputs[i]));
            buffer = (float) Math.pow(buffer, 2);
            result += buffer;
        }
        return result;
    }
    
    public float Train_RMSE(){
        //ROOT MEAN SQUARED ERROR FOR TRAINING.
        float result = 0;
        float buffer = 0;
        for(int i=0; i<352; i++){
            buffer = (denormalize(trainSet[i][15]) - denormalize(trainOutputs[i]));
            buffer = (float) Math.pow(buffer, 2);
            result += buffer;
        }
        return (float) Math.sqrt(result / 352); 
    }
    
    public void takeInput(int row){
        //TAKES ONE INPUT ROW FROM TRAINSET ARRAY. (FIRST 15 ARE INPUTS, 16TH IS THE ACTUAL OUTPUT!)
        for(int j=0 ;j<15; j++){
                inputLayer[j].setValue(trainSet[row][j]);
        }
    }
    
    public void calculateHiddenOutputs(){
        //CALCULATE HIDDEN LAYER OUTPUT VALUES & USE SIGMOID ACTIVATION FUNCTION.
        for(int m=0; m<hiddenNodeCount; m++){
                float buff=0;
                for(int n=0; n<inputNodeCount; n++){
                    buff += (inputLayer[n].getValue() * I_H_weights[n][m]);                    
                }
                buff += hiddenLayer[m].getBias();
                buff = sigmoid(buff);
                hiddenLayer[m].setValue(buff);
        }
    }
    
    public float calculateOutput(){
        //CALCULATE OUTPUT NODE VALUE & USE SIGMOID ACTIVATION FUNCTION.
        float buffer=0;
        for(int m=0; m<hiddenNodeCount; m++){
            buffer += (hiddenLayer[m].getValue() * H_O_weights[m]);                
        }
        buffer += outputNode.getBias();
        buffer = sigmoid(buffer);
        return buffer;
    }
    
    public float sigmoid(float x) {
        //SIGMOID ACTIVATION FUNCTION.
        return (float) (1.0f / (1.0f + Math.exp((double) (-x))));
    }
    
    public void backPropagation(){
        
        //ERROR BACK-PROPAGATION ALGORITHM.
        int random = RNG.nextInt(trainSet.length-1);        
        outputNode.setValue(trainOutputs[random]);     

        //SET INPUT LAYER TO THE RANDOMLY SELECTED DATA ROW.
        takeInput(random);
        
        //CALCULATE OUTPUT AND HIDDEN NODE ERRORS.
        calculateNodeErrors(random);
        
        //UPDATE HIDDEN-OUTPUT WEIGHTS.
        for(int i=0; i<hiddenNodeCount; i++){
            float delta = (learningRate * momentum * outputNode.getError() * hiddenLayer[i].getValue());
            H_O_weights[i] += delta;
        }
        
        //UPDATE INPUT-HIDDEN WEIGHTS.
        for(int i=0; i<inputNodeCount; i++){
            for(int j=0; j<hiddenNodeCount; j++){
                float delta = (learningRate * momentum * hiddenLayer[j].getError() * inputLayer[i].getValue());
                I_H_weights[i][j] += delta;
            }
        }
        
        //OUTPUT NODE BIAS UPDATE.
        float buffer = outputNode.getBias() + (learningRate * momentum * outputNode.getError() * outputNode.getBias());
        outputNode.setBias(buffer);
        
        //HIDDEN NODES BIAS UPDATES.
        for(int i=0; i<hiddenNodeCount; i++){
            buffer = hiddenLayer[i].getBias() + (learningRate * momentum * hiddenLayer[i].getError() * hiddenLayer[i].getBias());
            hiddenLayer[i].setBias(buffer);
        }
    }    
    
    public void calculateNodeErrors(int r){
        float act = trainSet[r][15];
        float pred = trainOutputs[r];        
        float error = pred * (1 - pred) * (act - pred);     // output * (1-output) * (actual - output) 
        
        outputNode.setError(error);                         
        
        for(int i=0; i<hiddenNodeCount; i++){
            float a = hiddenLayer[i].getValue();
            hiddenLayer[i].setError(a * (1-a) * error);
        }        
    }
    
    public void printBestWeights(){
        //UTILITY FUNCTION USED WHILE DEVELOPING. (NOT IMPORTANT!)
        System.out.println("BEST IH WEIGHTS:");
        for(int i=0; i<inputNodeCount; i++){
            for(int j=0; j<hiddenNodeCount; j++){
                System.out.printf("%f\t", BEST_I_H_weights[i][j]);
            }
            System.out.println("");
        }
        
        System.out.println("\n\nBEST HO WEIGHTS:");
        for(int j=0; j<hiddenNodeCount; j++){
                System.out.printf("%f\t", BEST_H_O_weights[j]);
        }
    }
    
    public void testForwardPass(){
        
        //5-FOLD CROSS VALIDATION TEST.
        for(int i=0; i<88; i++){
            takeTestInput(i);
            
            //(TEST) CALCULATE HIDDEN NODE OUTPUTS WITH BEST WEIGHTS.
            for(int m=0; m<hiddenNodeCount; m++){
                float buff=0;
                for(int n=0; n<inputNodeCount; n++){
                    buff += (inputLayer[n].getValue() * BEST_I_H_weights[n][m]);     //USING BEST WEIGHTS HERE !!!               
                }
                buff += hiddenLayer[m].getBestBias();                                //USING BEST BIAS VALUE HERE !!!
                buff = sigmoid(buff);
                hiddenLayer[m].setValue(buff);
            }
            
            //(TEST) CALCULATE OUTPUT WITH BEST WEIGHTS.
            float buffer=0;
            for(int m=0; m<hiddenNodeCount; m++){
                buffer += (hiddenLayer[m].getValue() * BEST_H_O_weights[m]);         //USING BEST WEIGHTS HERE !!!       
            }
            buffer += outputNode.getBestBias();                                      //USING BEST BIAS VALUE HERE !!!
            buffer = sigmoid(buffer);
            
            outputNode.setValue(buffer);
            testOutputs[i] = outputNode.getValue();            
        }
        
        Test_SSE = Test_SSE();
        Test_RMSE = Test_RMSE();
        
    }
    
    public void takeTestInput(int row){
        //TAKES ONE INPUT ROW FROM TEST DATA SET. (FIRST 15 ARE INPUT, 16TH IS THE ACTUAL OUTPUT)
        for(int j=0 ;j<15; j++){
                inputLayer[j].setValue(testSet[row][j]);
        }
    }

    public float Test_SSE(){
        float result = 0;
        float buffer = 0;
        for(int i=0; i<88; i++){
            buffer = (denormalize(testSet[i][15]) - denormalize(testOutputs[i]));
            buffer = (float) Math.pow(buffer, 2);
            result += buffer;
        }
        return result;
    }
    
    public float Test_RMSE(){
        float result = 0;
        float buffer = 0;
        for(int i=0; i<88; i++){
            buffer = (denormalize(testSet[i][15]) - denormalize(testOutputs[i]));
            buffer = (float) Math.pow(buffer, 2);
            result += buffer;
        }
        return (float) Math.sqrt(result / 352); 
    }
    
    public void printResults(){
        
        for(int i=0; i<88; i++){            
            System.out.println("----------------------------------------------------");
            System.out.println("ACTUAL: " + denormalize(testSet[i][15]) + "\t\tPREDICTION: " + denormalize(testOutputs[i]) + "\t\tERROR: " + denormalize((testSet[i][15] - testOutputs[i])));     
        }
        
    }
    
    public float denormalize(float x){
        //DENORMALIZATION FUNCTION FOR OUTPUT. (MAX = 1090.84, MIN=0)
        return (1090.84F * x);
    }
}
