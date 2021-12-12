package dev.damaso;

public class Main {
    static public int value() {
        return 100;
    }
    public float[][] getImage() {
        float[][] image = new float[10][20];
        for (int i=0;i<5;i++) {
            image[i][i] = 1;
        }
        return image;
    }
    static public void main(String [] args) {
        System.out.println("Hallo world!");
    }
}