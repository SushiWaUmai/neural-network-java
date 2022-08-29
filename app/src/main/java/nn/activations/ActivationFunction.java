package nn.activations;
import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {
    public abstract float[] activation(float[] x);
    public abstract float[] activationPrime(float[] x);
}
