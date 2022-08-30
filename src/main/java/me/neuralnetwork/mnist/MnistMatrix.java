package me.neuralnetwork.mnist;

public class MnistMatrix {

  private int[][] data;

  private int nRows;
  private int nCols;

  private int label;

  public MnistMatrix(int nRows, int nCols) {
    this.nRows = nRows;
    this.nCols = nCols;

    data = new int[nRows][nCols];
  }

  public int getValue(int r, int c) { return data[r][c]; }

  public void setValue(int row, int col, int value) { data[row][col] = value; }

  public int getLabel() { return label; }

  public void setLabel(int label) { this.label = label; }

  public int getNumberOfRows() { return nRows; }

  public int getNumberOfColumns() { return nCols; }

  public String toString() {
    String result = "";
    for (int r = 0; r < nRows; r++) {
      for (int c = 0; c < nCols; c++) {
        result += data[r][c] + " ";
      }
      result += "\n";
    }
    return result;
  }

  public String toASCII() {
    String result = "";
    for (int r = 0; r < nRows; r++) {
      for (int c = 0; c < nCols; c++) {
        result += data[r][c] == 0 ? " " : "X";
      }
      result += "\n";
    }
    return result;
  }

  public static String toASCII(float[] data) {
    String result = "";
    for (int r = 0; r < 28; r++) {
      for (int c = 0; c < 28; c++) {
        result += data[r * 28 + c] == 0 ? " " : "X";
      }
      result += "\n";
    }
    return result;
  }
}
