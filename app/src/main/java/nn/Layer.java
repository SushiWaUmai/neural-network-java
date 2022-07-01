package nn;

public abstract class Layer {
    public abstract float[] forward(float[] prev);
    public abstract float[] backward(float[] loss, float learningRate);
}

