package nn.activations;

public class ReLU extends ActivationFunction {
    public static float relu(float value) {
        return Math.max(0, value);
    }

    public static float reluPrime(float value) {
        return value > 0 ? 1 : 0;
    }

    public float[] activation(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = relu(x[i]);
        }
        return y;
    }

    public float[] activationPrime(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = reluPrime(x[i]);
        }
        return y;
    }
}
