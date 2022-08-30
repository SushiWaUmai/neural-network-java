package me.neuralnetwork.nn;

public class FCLayer extends Layer {
    public int inputSize;
    public int outputSize;
    public float[][] weights;
    public float[] bias;

    public float[] inputData; 

    public FCLayer(int inputs, int outputs) {
        this.inputSize = inputs;
        this.outputSize = outputs;
        this.weights = new float[inputs][outputs];
        this.bias = new float[outputs];

        for (int o = 0; o < outputs; o++) {
            for (int i = 0; i < inputs; i++) {
                this.weights[i][o] = 2 * (float)Math.random() - 1;
            }
            this.bias[o] = 2 * (float)Math.random() - 1;
        }
    }

    public float[] forward(float[] prev) {
        this.inputData = prev;
        float[] next = new float[outputSize];

        for (int o = 0; o < outputSize; o++) {
            float sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += prev[i] * weights[i][o];
            }
            next[o] = sum + bias[o];
        }
        return next;
    }

    public float[] backward(float[] outputErr, float learningRate) {
        float[] inputErr = new float[inputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int o = 0; o < outputSize; o++) {
                inputErr[i] += outputErr[o] * weights[i][o];
            }
        }

        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[i][o] -= inputData[i] * outputErr[o] * learningRate;
            }
            bias[o] -= outputErr[o] * learningRate;
        }
        return inputErr;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("FCLayer(");
        sb.append(inputSize);
        sb.append(", ");
        sb.append(outputSize);
        sb.append(")");
        return sb.toString();
    }
}
