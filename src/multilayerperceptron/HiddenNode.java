package multilayerperceptron;

public class HiddenNode extends Node {

    private float bias;
    private float bestBias;
    private float error;
    
    public HiddenNode(float value) {
        super(value);
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public float getBestBias() {
        return bestBias;
    }

    public void setBestBias(float bestBias) {
        this.bestBias = bestBias;
    }

    public float getError() {
        return error;
    }

    public void setError(float error) {
        this.error = error;
    }
    
}
