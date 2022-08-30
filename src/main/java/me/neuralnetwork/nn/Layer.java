package me.neuralnetwork.nn;
import java.io.Serializable;

public abstract class Layer implements Serializable {
    public abstract float[] forward(float[] prev);
    public abstract float[] backward(float[] loss, float learningRate);
}

