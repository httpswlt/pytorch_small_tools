#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "utils.h"

typedef struct {
    int w;
    int h;
    int c;
    int bnums;
    float* data;
    float* bboxes;

} Image;

typedef struct {
    float r;
    float g;
    float b;
} RGB;

Image make_image(int w, int h, int c, int bnums);
void fill_image(Image m, float s);
void place_image(Image im, int w, int h, int dx, int dy, Image canvas);
void random_distort_image(Image im, float hue, float saturation, float exposure);
void flip_image(Image a);