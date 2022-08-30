package me.neuralnetwork.nn.activations;

public class Softmax extends ActivationFunction {
    public float[] activation(float[] x) {
        assert x.length > 1 : "Softmax activation function requires at least 2 inputs";
        float[] y = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) Math.exp(x[i]);
            sum += y[i];
        }
        for (int i = 0; i < x.length; i++) {
            y[i] /= sum;
        }
        return y;
    }

    public float[] activationPrime(float[] x) {
        float[] y = new float[x.length];
        float[] a = activation(x);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += a[j] * ((i == j ? 1 : 0) - a[i]);
            }
        }
        return y;
    }
}
