/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package me.neuralnetwork.nn;
import me.neuralnetwork.nn.activations.*;
import java.io.IOException;

import me.neuralnetwork.mnist.*;

public class App {
    public static void main(String[] args) {
        // xor();
        // trainMnist();
        testMnist();
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
            new float[] { 1, 0 },
            new float[] { 1, 0 },
            new float[] { 0, 1 },
            new float[] { 0, 1 },
        };

        float[][] testX = new float[][]
        {
            new float[] { 1, 0 },
            new float[] { 0, 1 },
            new float[] { 0, 0 },
            new float[] { 1, 1 },
        };
      

        Model model = new Model(
            new Layer[] {
                new FCLayer(2, 3),
                new ActivationLayer(new Sigmoid()),
                new FCLayer(3, 2),
                new ActivationLayer(new Tanh()),
            }
        );

        model.fit(trainX, trainY, learningRate, epochs, 1);
        
        for (int i = 0; i < 4; i++) {
            float[] y = model.forward(testX[i]);
            System.out.println("Value: " + arrayToString(testX[i]));
            System.out.println("Pred: " + arrayToString(y));
        }
    }

    public static void trainMnist() {
        MnistMatrix[] mnistTrain;
        MnistMatrix[] mnistTest;
        try {
            mnistTrain = new MnistDataReader().readData("./src/main/resources/train-images.idx3-ubyte", "./src/main/resources/train-labels.idx1-ubyte");
            mnistTest = new MnistDataReader().readData("./src/main/resources/t10k-images.idx3-ubyte", "./src/main/resources/t10k-labels.idx1-ubyte");
            System.out.println("Successfully read MNIST data");
        }
        catch (IOException e) {
            System.out.println("Failed to load file: " + e.getMessage());
            return;
        }

        int epochs = 35;
        float learningRate = 0.01f;
        float[][] trainX = new float[mnistTrain.length][28 * 28];
        float[][] trainY = new float[mnistTrain.length][10];
        toFloatArray(mnistTrain, trainX, trainY);

        Model model;
        try {
            model = Model.load("./src/main/resources/mnistmodel.bin");
        }
        catch (IOException e) {
            System.out.println("Failed to load model: " + e.getMessage());
            System.out.println("Creating new model...");
            
            model = new Model(
                new Layer[] {
                    new FCLayer(28 * 28, 16),
                    new ActivationLayer(new Tanh()),
                    new FCLayer(16, 16),
                    new ActivationLayer(new Tanh()),
                    new FCLayer(16, 10),
                    new ActivationLayer(new Sigmoid()),
                }
            );
        }

        model.fit(trainX, trainY, learningRate, epochs, 32);
        
        float[][] testX = new float[mnistTest.length][28 * 28];
        float[][] testY = new float[mnistTest.length][10];
        toFloatArray(mnistTest, testX, testY);

        int correct = 0;
        for (int i = 0; i < mnistTest.length; i++) {
            float[] y = model.forward(testX[i]);
            int pred = argmax(y);
            if (pred == mnistTest[i].getLabel()) {
                correct++;
            }
        }

        float accuracy = (float)correct / mnistTest.length;
        System.out.println("The accuracy of this model is: " + (accuracy * 100) + "%.");

        try {
            Model.save("./src/main/resources/mnistmodel.bin", model);
            System.out.println("Successfully saved model.");
        }
        catch (IOException e) {
            System.out.println("Failed to save model: " + e.getMessage());
        }
    }
    
    public static void testMnist() {
        MnistMatrix[] mnistTest;
        try {
            mnistTest = new MnistDataReader().readData("./src/main/resources/t10k-images.idx3-ubyte", "./src/main/resources/t10k-labels.idx1-ubyte");
            System.out.println("Successfully read MNIST data");
        }
        catch (IOException e) {
            System.out.println("Failed to load file: " + e.getMessage());
            return;
        }

        Model model;
        try {
            model = Model.load("./src/main/resources/mnistmodel.bin");
        }
        catch (IOException e) {
            System.out.println("Failed to load model: " + e.getMessage());
            return;
        }
        float[][] testX = new float[mnistTest.length][28 * 28];
        float[][] testY = new float[mnistTest.length][10];
        toFloatArray(mnistTest, testX, testY);

        for (int i = 0; i < 10; i++) {
            int index = (int)(Math.random() * mnistTest.length);
            float[] y = model.forward(testX[index]);
            int pred = argmax(y);
            System.out.println("Value: " + MnistMatrix.toASCII(testX[index]));
            System.out.println("Pred: " + pred);
        }
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
