#include "lenet_cnn_fixed.h"

/* Pool1: MaxPool 2x2, stride 2, sur 24x24x20 -> 12x12x20 */
void Pool1_24x24x20_2x2x20_2_0_fixed(
    fx_t  input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    fx_t  output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
{
    short k, y, x, dy, dx;

    for (k = 0; k < POOL1_NBOUTPUT; k++) {

        for (y = 0; y < POOL1_HEIGHT; y++) {

            for (x = 0; x < POOL1_WIDTH; x++) {

                short in_y = y * POOL1_STRIDE;
                short in_x = x * POOL1_STRIDE;

                fx_t max_val = input[k][in_y][in_x];

				#pragma HLS PIPELINE

                for (dy = 0; dy < POOL1_DIM; dy++) {
                    for (dx = 0; dx < POOL1_DIM; dx++) {
                        fx_t v = input[k][in_y + dy][in_x + dx];
                        if (v > max_val)
                            max_val = v;
                    }
                }

                output[k][y][x] = max_val;
            }
        }
    }
}

/* Pool2: MaxPool 2x2, stride 2, sur 8x8x40 -> 4x4x40 */
void Pool2_8x8x40_2x2x40_2_0_fixed(
    fx_t  input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    fx_t  output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    short k, y, x, dy, dx;

    for (k = 0; k < POOL2_NBOUTPUT; k++) {

        for (y = 0; y < POOL2_HEIGHT; y++) {
            for (x = 0; x < POOL2_WIDTH; x++) {

                short in_y = y * POOL2_STRIDE;
                short in_x = x * POOL2_STRIDE;

                fx_t max_val = input[k][in_y][in_x];

				#pragma HLS PIPELINE

                for (dy = 0; dy < POOL2_DIM; dy++) {
                    for (dx = 0; dx < POOL2_DIM; dx++) {
                        fx_t v = input[k][in_y + dy][in_x + dx];
                        if (v > max_val)
                            max_val = v;
                    }
                }

                output[k][y][x] = max_val;
            }
        }
    }
}

