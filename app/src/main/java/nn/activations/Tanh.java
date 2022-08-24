package nn.activations;

public class Tanh extends ActivationFunction {
    public static float tanh(float value) {
        return (float)Math.tanh(value);
    }

    public static float tanhPrime(float value) {
        return 1 - (float)Math.pow(Math.tanh(value), 2);
    }

    public float[] activation(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = tanh(x[i]);
        }
        return y;
    }

    public float[] activationPrime(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = tanhPrime(x[i]);
        }
        return y;
    }
}
