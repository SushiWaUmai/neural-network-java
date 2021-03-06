package nn;

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

    public void fit(float[][] x, float[][] y, float learningRate, int epochs, int batchsize) {
        for (int epoch = 0; epoch < epochs; epoch++) {
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
