package nn.activations;

public class Softmax extends ActivationFunction {
    public float[] activation(float[] x) {
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
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                if (i == j) {
                    y[i] += x[j] * (1 - x[j]);
                } else {
                    y[i] -= x[j] * x[j];
                }
            }
        }
        return y;
    }
}
