#include "lenet_cnn_fixed.h"

/* Fc1: entrée 4x4x40 -> 400 neurones, ReLU */
void Fc1_40_400_fixed(
    fx_t  input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fx_t  kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fx_t  bias[FC1_NBOUTPUT],
    fx_t  output[FC1_NBOUTPUT])
{
    short k, z, y, x;

    for (k = 0; k < FC1_NBOUTPUT; k++) {

        fx_t acc = bias[k];

        for (z = 0; z < POOL2_NBOUTPUT; z++) {
            for (y = 0; y < POOL2_HEIGHT; y++) {
				#pragma HLS PIPELINE
                for (x = 0; x < POOL2_WIDTH; x++) {

                    fx_t v = input[z][y][x];
                    fx_t w = kernel[k][z][y][x];

                    acc += fx_mul(v, w);
                }
            }
        }

        /* ReLU */
        if (acc < 0)
            acc = 0;

        output[k] = acc;
    }
}

/* Fc2: entrée 400 -> 10 classes, sans ReLU  */
void Fc2_400_10_fixed(
    fx_t  input[FC1_NBOUTPUT],
    fx_t  kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fx_t  bias[FC2_NBOUTPUT],
    fx_t  output[FC2_NBOUTPUT])
{
    short k, j;

    for (k = 0; k < FC2_NBOUTPUT; k++) {

        fx_t acc = bias[k];
		#pragma HLS PIPELINE
        for (j = 0; j < FC1_NBOUTPUT; j++) {
            fx_t v = input[j];
            fx_t w = kernel[k][j];

            acc += fx_mul(v, w);
        }

        output[k] = acc;
    }
}

