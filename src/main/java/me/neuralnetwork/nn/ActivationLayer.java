package me.neuralnetwork.nn;

import me.neuralnetwork.nn.activations.*;

public class ActivationLayer extends Layer {
    public float[] inputData; 
    public ActivationFunction activationFunction;

    public ActivationLayer(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public float[] forward(float[] prev) {
        this.inputData = prev;
        float[] next = activationFunction.activation(inputData);
        return next;
    }

    public float[] backward(float[] outputErr, float learningRate) {
        float[] inputErr = new float[outputErr.length];
        inputErr = activationFunction.activationPrime(inputData);

        for (int i = 0; i < outputErr.length; i++) {
            inputErr[i] *= outputErr[i];
        }

        return inputErr;
    }

    public String toString() {
        return "ActivationLayer";
    }
}
