#include "lenet_cnn_fixed.h"

/* Conv1: entrée 28x28x1 -> sortie 24x24x20 (valid, 5x5, stride 1, ReLU) */
void Conv1_28x28x1_5x5x20_1_0_fixed(
    fx_t  input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fx_t  kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fx_t  bias[CONV1_NBOUTPUT],
    fx_t  output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])
{
    short k, y, x, z, ky, kx;

    for (k = 0; k < CONV1_NBOUTPUT; k++) {
        for (y = 0; y < CONV1_HEIGHT; y++) {
            for (x = 0; x < CONV1_WIDTH; x++) {

                fx_t acc = bias[k];

                for (z = 0; z < IMG_DEPTH; z++) {
                    for (ky = 0; ky < CONV1_DIM; ky++) {
						#pragma HLS PIPELINE
                        for (kx = 0; kx < CONV1_DIM; kx++) {
                            short in_y = y * CONV1_STRIDE + ky - CONV1_PAD;
                            short in_x = x * CONV1_STRIDE + kx - CONV1_PAD;

                            fx_t v = input[z][in_y][in_x];
                            fx_t w = kernel[k][z][ky][kx];

                            acc += fx_mul(v, w);
                        }
                    }
                }

                /* ReLU */
                if (acc < 0)
                    acc = 0;

                output[k][y][x] = acc;
            }
        }
    }
}

/* Conv2: entrée 12x12x20 -> sortie 8x8x40 (valid, 5x5, stride 1, ReLU) */
void Conv2_12x12x20_5x5x40_1_0_fixed(
    fx_t  input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    fx_t  kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fx_t  bias[CONV2_NBOUTPUT],
    fx_t  output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])
{
    short k, y, x, z, ky, kx;

    for (k = 0; k < CONV2_NBOUTPUT; k++) {
        for (y = 0; y < CONV2_HEIGHT; y++) {
            for (x = 0; x < CONV2_WIDTH; x++) {

                fx_t acc = bias[k];

                for (z = 0; z < POOL1_NBOUTPUT; z++) {
                    for (ky = 0; ky < CONV2_DIM; ky++) {
						#pragma HLS PIPELINE
                        for (kx = 0; kx < CONV2_DIM; kx++) {
                            short in_y = y * CONV2_STRIDE + ky - CONV2_PAD;
                            short in_x = x * CONV2_STRIDE + kx - CONV2_PAD;

                            fx_t v = input[z][in_y][in_x];
                            fx_t w = kernel[k][z][ky][kx];

                            acc += fx_mul(v, w);
                        }
                    }
                }

                /* ReLU */
                if (acc < 0)
                    acc = 0;

                output[k][y][x] = acc;
            }
        }
    }
}

