package nn;
import java.io.*;

public class Model {
    public Layer[] layers;

    public Model(Layer... layers) {
        this.layers = layers;
    }

    public float[] forward(float[] current) {
        for (int i = 0; i < layers.length; i++) {
            current = layers[i].forward(current);
        }
        return current;
    }

    public void backward(float[] loss, float learningRate) {
        for (int i = layers.length - 1; i >= 0; i--) {
            loss = layers[i].backward(loss, learningRate);
        }
    }

    public static void save(String path, Model model) throws IOException {
        try {
            FileOutputStream fos = new FileOutputStream(path);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(model.layers);
            oos.close();
        } catch (IOException e) {
            throw e;
        }
    }

    public static Model load(String path) throws IOException {
        try {
            FileInputStream fis = new FileInputStream(path);
            ObjectInputStream ois = new ObjectInputStream(fis);
            Layer[] layers = (Layer[]) ois.readObject();
            ois.close();
            return new Model(layers);
        } catch (IOException e) {
            throw e;
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    public void fit(float[][] x, float[][] y, float learningRate, int epochs, int batchsize) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(x, y);

            System.out.println("Starting Epoch: " + (epoch + 1));
            float err = 0;
            float[] loss = new float[y[0].length];

            for (int i = 0; i < x.length; i++) {
                float[] pred = forward(x[i]);

                err += mse(y[i], pred);

                for (int j = 0; j < pred.length; j++) {
                    loss[j] += msePrime(y[i][j], pred[j]) / batchsize;
                }

                if (i % batchsize == 0) {
                    backward(loss, learningRate);
                    loss = new float[loss.length];
                }

                backward(loss, learningRate);
                loss = new float[loss.length];
            }


            err /= x.length;
            System.out.println("Error: " + err);;
        }
    }

    public void shuffle(float[][] x, float[][] y) {
        for (int i = 0; i < x.length; i++) {
            int j = (int)(Math.random() * x.length);
            float[] temp = x[i];
            x[i] = x[j];
            x[j] = temp;
            temp = y[i];
            y[i] = y[j];
            y[j] = temp;
        }
    }

    public static float mse(float y, float pred) {
        return (y - pred) * (y - pred);
    }
    
    public static float mse(float[] y, float[] pred) {
        float sum = 0;
        for (int i = 0; i < y.length; i++) {
            sum += mse(y[i], pred[i]);
        }
        return sum / y.length;
    }

    public static float msePrime(float y, float pred) {
        return  2 * (pred - y);
    }

    public static float[] msePrime(float[] y, float[] pred) {
        float[] result = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            result[i] = msePrime(y[i], pred[i]) / y.length;
        }
        return result;
    }

}
