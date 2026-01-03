/**
  ******************************************************************************
  * @file    lenet_cnn_fixed.c
  * @brief   Version fixed-point de LeNet (même fonctionnement que lenet_cnn_float.c)
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include "lenet_cnn_fixed.h"


/* =======================
 *  VARIABLES GLOBALES
 * ======================= */

unsigned char  REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
float          INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];

/* Poids en float (lus depuis HDF5 avec utils.c) */
float CONV1_KERNEL_F[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
float CONV1_BIAS_F[CONV1_NBOUTPUT];
float CONV2_KERNEL_F[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
float CONV2_BIAS_F[CONV2_NBOUTPUT];
float FC1_KERNEL_F[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
float FC1_BIAS_F[FC1_NBOUTPUT];
float FC2_KERNEL_F[FC2_NBOUTPUT][FC1_NBOUTPUT];
float FC2_BIAS_F[FC2_NBOUTPUT];

/* Version quantifiée en fixed */
fx_t CONV1_KERNEL_FX[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
fx_t CONV1_BIAS_FX[CONV1_NBOUTPUT];
fx_t CONV2_KERNEL_FX[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
fx_t CONV2_BIAS_FX[CONV2_NBOUTPUT];
fx_t FC1_KERNEL_FX[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
fx_t FC1_BIAS_FX[FC1_NBOUTPUT];
fx_t FC2_KERNEL_FX[FC2_NBOUTPUT][FC1_NBOUTPUT];
fx_t FC2_BIAS_FX[FC2_NBOUTPUT];


fx_t INPUT_FX[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
fx_t FC2_OUTPUT_FX[FC2_NBOUTPUT];
float SOFTMAX_OUTPUT[FC2_NBOUTPUT];


#include "weights.h"
/* =======================
 *  TOP-LEVEL FIXED
 * ======================= */

void lenet_cnn_fixed(
    fx_t  input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fx_t  conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fx_t  conv1_bias[CONV1_NBOUTPUT],
    fx_t  conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fx_t  conv2_bias[CONV2_NBOUTPUT],
    fx_t  fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fx_t  fc1_bias[FC1_NBOUTPUT],
    fx_t  fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fx_t  fc2_bias[FC2_NBOUTPUT],
    fx_t  output[FC2_NBOUTPUT])
{
    /* Buffers internes (mêmes dimensions que la version float) */
    static fx_t conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    static fx_t pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    static fx_t conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    static fx_t pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    static fx_t fc1_output[FC1_NBOUTPUT];

    Conv1_28x28x1_5x5x20_1_0_fixed(
        input,
        conv1_kernel,
        conv1_bias,
        conv1_output);

    Pool1_24x24x20_2x2x20_2_0_fixed(
        conv1_output,
        pool1_output);

    Conv2_12x12x20_5x5x40_1_0_fixed(
        pool1_output,
        conv2_kernel,
        conv2_bias,
        conv2_output);

    Pool2_8x8x40_2x2x40_2_0_fixed(
        conv2_output,
        pool2_output);

    Fc1_40_400_fixed(
        pool2_output,
        fc1_kernel,
        fc1_bias,
        fc1_output);

    Fc2_400_10_fixed(
        fc1_output,
        fc2_kernel,
        fc2_bias,
        output);
}

/* =======================
 *  SOFTMAX FIXED -> FLOAT
 * ======================= */

void Softmax_fixed(fx_t vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT])
{
    short i;
    float max_val;
    float sum = 0.0f;
    float temp[FC2_NBOUTPUT];

    /* Conversion en float + recherche du max*/
    max_val = fx_to_float(vector_in[0]);
    for (i = 0; i < FC2_NBOUTPUT; i++) {
        temp[i] = fx_to_float(vector_in[i]);
        if (temp[i] > max_val)
            max_val = temp[i];
    }

    /* exp(x - max) */
    for (i = 0; i < FC2_NBOUTPUT; i++) {
        temp[i] = expf(temp[i] - max_val);
        sum += temp[i];
    }

    if (sum == 0.0f)
        sum = 1.0f;

    for (i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = temp[i] / sum;
    }
}

/* =======================
 *  MAIN FIXED-POINT
 *  (analogue de lenet_cnn_float.c)
 * ======================= */

void main(void)
{
    short x, y, z, k, m;
   
    char *test_labels_filename = "mnist/t10k-labels-idx1-ubyte";
    FILE *label_file;
    int ret;
    unsigned char label, number;
    unsigned int error;
    unsigned char labels_legend[10] = {0,1,2,3,4,5,6,7,8,9};
    char img_filename[120];
    char img_count[10];
    float max;
    struct timeval start, end;
    double tdiff;

    printf("\e[1;1H\e[2J");

    /* Lecture des poids en float */
    printf("\nReading weights \n");
    // lecture des biais et poids depuis weights.h
    InitConv1WeightsFromHeader();   // remplit CONV1_KERNEL_F[]
    InitConv1BiasFromHeader();
    InitConv2WeightsFromHeader();
    InitConv2BiasFromHeader();
    InitFc1WeightsFromHeader();
    InitFc1BiasFromHeader();
    InitFc2WeightsFromHeader();
    InitFc2BiasFromHeader();


    /* Quantification des poids en fixed */
    for (k = 0; k < CONV1_NBOUTPUT; k++) {
        CONV1_BIAS_FX[k] = float_to_fx(CONV1_BIAS_F[k]);
        for (z = 0; z < IMG_DEPTH; z++)
            for (y = 0; y < CONV1_DIM; y++)
                for (x = 0; x < CONV1_DIM; x++)
                    CONV1_KERNEL_FX[k][z][y][x] = float_to_fx(CONV1_KERNEL_F[k][z][y][x]);
    }

    for (k = 0; k < CONV2_NBOUTPUT; k++) {
        CONV2_BIAS_FX[k] = float_to_fx(CONV2_BIAS_F[k]);
        for (z = 0; z < POOL1_NBOUTPUT; z++)
            for (y = 0; y < CONV2_DIM; y++)
                for (x = 0; x < CONV2_DIM; x++)
                    CONV2_KERNEL_FX[k][z][y][x] = float_to_fx(CONV2_KERNEL_F[k][z][y][x]);
    }

    for (k = 0; k < FC1_NBOUTPUT; k++) {
        FC1_BIAS_FX[k] = float_to_fx(FC1_BIAS_F[k]);
        for (z = 0; z < POOL2_NBOUTPUT; z++)
            for (y = 0; y < POOL2_HEIGHT; y++)
                for (x = 0; x < POOL2_WIDTH; x++)
                    FC1_KERNEL_FX[k][z][y][x] = float_to_fx(FC1_KERNEL_F[k][z][y][x]);
    }

    for (k = 0; k < FC2_NBOUTPUT; k++) {
        FC2_BIAS_FX[k] = float_to_fx(FC2_BIAS_F[k]);
        for (z = 0; z < FC1_NBOUTPUT; z++)
            FC2_KERNEL_FX[k][z] = float_to_fx(FC2_KERNEL_F[k][z]);
    }

    /* Ouverture du fichier de labels */
    printf("\nOpening labels file \n");
    label_file = fopen(test_labels_filename, "r");
    if (!label_file) {
        printf("Error: Unable to open file %s.\n", test_labels_filename);
        exit(1);
    }

    /* On saute les 8 octets d'en-tête */
    for (k = 0; k < 8; k++)
        ret = fscanf(label_file, "%c", &label);

    printf("\nProcessing \n");

    m = 0;
    error = 0;

    gettimeofday(&start, NULL);

    /* Boucle principale sur les images MNIST */
    while (1) {

        ret = fscanf(label_file, "%c", &label);
        if (feof(label_file)) break;

        strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
        sprintf(img_count, "%d", m);
        if      (m < 10)    strcat(img_filename, "0000");
        else if (m < 100)   strcat(img_filename, "000");
        else if (m < 1000)  strcat(img_filename, "00");
        else if (m < 10000) strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");

        printf("\033[%d;%dH%s\n", 7, 0, img_filename);

        /* Lecture image + normalisation float [0,1] */
        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        NormalizeImg((unsigned char *)REF_IMG, (float *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH);

        /* Conversion de l'image en fixed */
        for (z = 0; z < IMG_DEPTH; z++)
            for (y = 0; y < IMG_HEIGHT; y++)
                for (x = 0; x < IMG_WIDTH; x++)
                    INPUT_FX[z][y][x] = float_to_fx(INPUT_NORM[z][y][x]);

        /* Inference en fixed-point */
        lenet_cnn_fixed(
            INPUT_FX,
            CONV1_KERNEL_FX,
            CONV1_BIAS_FX,
            CONV2_KERNEL_FX,
            CONV2_BIAS_FX,
            FC1_KERNEL_FX,
            FC1_BIAS_FX,
            FC2_KERNEL_FX,
            FC2_BIAS_FX,
            FC2_OUTPUT_FX);

        /* Softmax (conversion en probabilités float) */
        Softmax_fixed(FC2_OUTPUT_FX, SOFTMAX_OUTPUT);

        printf("\n\nSoftmax output: \n");
        max = 0.0f;
        number = 0;
        for (k = 0; k < FC2_NBOUTPUT; k++) {
            printf("%.2f%% ", SOFTMAX_OUTPUT[k] * 100.0f);
            if (SOFTMAX_OUTPUT[k] > max) {
                max = SOFTMAX_OUTPUT[k];
                number = (unsigned char)k;
            }
        }

        printf("\n\nPredicted: %d \t Actual: %d\n",
               labels_legend[number], label);

        if (labels_legend[number] != label)
            error++;

        m++;
    } /* fin boucle principale */

    gettimeofday(&end, NULL);
    tdiff = (double)(end.tv_sec - start.tv_sec) +
            (double)(end.tv_usec - start.tv_usec) / 1e6;

    printf("TOTAL PROCESSING TIME (gettimeofday): %f s\n", tdiff);

    printf("\n\nErrors : %d / %d", error, m);
    printf("\n\nSuccess rate = %f%%",
           (1.0f - ((float)error / (float)m)) * 100.0f);

    printf("\n\n");

    fclose(label_file);
    
}

