package nn.activations;

public abstract class ActivationFunction {
    public abstract float[] activation(float[] x);
    public abstract float[] activationPrime(float[] x);
}
