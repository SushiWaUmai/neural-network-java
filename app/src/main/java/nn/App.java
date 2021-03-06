/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package nn;
import java.io.IOException;

import mnist.*;

public class App {
    public static void main(String[] args) {
        // xor();
        trainMnist();
    }

    public static void xor() {
        int epochs = 1000;
        float learningRate = 0.1f;
        float[][] trainX = new float[][]
        {
            new float[] { 1, 0 },
            new float[] { 0, 1 },
            new float[] { 0, 0 },
            new float[] { 1, 1 },
        };
        float[][] trainY = new float[][] 
        {
            new float[] { 1 },
            new float[] { 1 },
            new float[] { 0 },
            new float[] { 0 },
        };
        
        Model model = new Model(
            new Layer[] {
                new FCLayer(2, 3),
                new ActivationLayer(x -> ActivationLayer.tanh(x), x -> ActivationLayer.tanhPrime(x)),
                new FCLayer(3, 1),
                new ActivationLayer(x -> ActivationLayer.tanh(x), x -> ActivationLayer.tanhPrime(x)),
            }
        );

        model.fit(trainX, trainY, learningRate, epochs, 1);
        
        for (int i = 0; i < 4; i++) {
            float[] y = model.forward(trainX[i]);
            System.out.println("Value: " + arrayToString(trainX[i]));
            System.out.println("Pred: " + y[0]);
        }
    }

    public static void trainMnist() {
        MnistMatrix[] mnistTrain;
        MnistMatrix[] mnistTest;
        try {
            mnistTrain = new MnistDataReader().readData("./src/main/resources/train-images.idx3-ubyte", "./src/main/resources/train-labels.idx1-ubyte");
            mnistTest = new MnistDataReader().readData("./src/main/resources/t10k-images.idx3-ubyte", "./src/main/resources/t10k-labels.idx1-ubyte");
        }
        catch (IOException e) {
            System.out.println("Failed to load file: " + e.getMessage());
            return;
        }

        int epochs = 35;
        float learningRate = 0.05f;
        float[][] trainX = new float[mnistTrain.length][28 * 28];
        float[][] trainY = new float[mnistTrain.length][10];
        toFloatArray(mnistTrain, trainX, trainY);

        Model model = new Model(
            new Layer[] {
                new FCLayer(28 * 28, 16),
                new ActivationLayer(x -> ActivationLayer.tanh(x), x -> ActivationLayer.tanhPrime(x)),
                new FCLayer(16, 16),
                new ActivationLayer(x -> ActivationLayer.tanh(x), x -> ActivationLayer.tanhPrime(x)),
                new FCLayer(16, 10),
                new ActivationLayer(x -> ActivationLayer.tanh(x), x -> ActivationLayer.tanhPrime(x)),
            }
        );
    
        model.fit(trainX, trainY, learningRate, epochs, 32);
        
        float[][] testX = new float[mnistTest.length][28 * 28];
        float[][] testY = new float[mnistTest.length][10];
        toFloatArray(mnistTest, testX, testY);

        int correct = 0;
        for (int i = 0; i < mnistTest.length; i++) {
            float[] y = model.forward(trainX[i]);
            int pred = argmax(y);
            if (pred == mnistTest[i].getLabel()) {
                correct++;
            }
        }

        float accuracy = (float)correct / mnistTest.length;
        System.out.println("The accuracy of this model is: " + (accuracy * 100) + "%.");
    }

    public static int argmax(float[] arr) {
        int result = -1;
        float max = -Float.MAX_VALUE;

        for (int i = 0; i < arr.length; i++) {
            if (max < arr[i]) {
                max = arr[i];
                result = i;
            }
        }

        return result;
    }

    public static void toFloatArray(MnistMatrix[] matrix, float[][] images, float[][] labels) {
        for (int i = 0; i < matrix.length; i++) {
            labels[i][matrix[i].getLabel()] = 1;
            
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    images[i][x * 28 + y] = matrix[i].getValue(x, y) / 255f;
                }
            }
        }
    }

    public static void printArray(float[] arr) {
        System.out.println(arrayToString(arr));
    }

    public static String arrayToString(float[] arr) {
        String result = "[";

        for (int i = 0; i < arr.length - 1; i++) {
            result += arr[i] + ", ";
        }
        result += arr[arr.length - 1] + "]";

        return result;
    }
}
