package nn;
import java.util.function.*;

public class ActivationLayer extends Layer {
    public float[] inputData; 
    public Function<Float, Float> activation;
    public Function<Float, Float> activationPrime;

    public ActivationLayer(Function<Float, Float> activation, Function<Float, Float> activationPrime) {
        this.activation = activation;
        this.activationPrime = activationPrime;
    }
    
    public float[] forward(float[] prev) {
        this.inputData = prev;
        float[] next = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            next[i] = activation.apply(inputData[i]);
        }

        return next;
    }

    public float[] backward(float[] outputErr, float learningRate) {
        float[] inputErr = new float[outputErr.length];

        for (int i = 0; i < outputErr.length; i++) {
            inputErr[i] = activationPrime.apply(inputData[i]) * outputErr[i];
        }

        return inputErr;
    }

    public static float sigmoid(float value) {
        return 1f / (1f + (float)Math.pow(Math.E, -value));
    }

    public static float sigmoidPrime(float value) {
        return sigmoid(value) * (1 - sigmoid(value));
    }

    public static float tanh(float value) {
        return (float)Math.tanh(value);
    }

    public static float tanhPrime(float value) {
        return 1 - (float)Math.pow(Math.tanh(value), 2);
    }

    public String toString() {
        return "ActivationLayer";
    }
}
