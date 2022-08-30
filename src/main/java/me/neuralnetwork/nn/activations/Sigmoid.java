package me.neuralnetwork.nn.activations;

public class Sigmoid extends ActivationFunction {
    public float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    public float sigmoidPrime(float x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public float[] activation(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = sigmoid(x[i]);
        }
        return y;
    }

    public float[] activationPrime(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = sigmoidPrime(x[i]);
        }
        return y;
    }
}
