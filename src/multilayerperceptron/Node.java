package multilayerperceptron;

public abstract class Node {
    
    private float value;

    public Node(float value) {
        this.value = value;
    }

    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }
    
}
