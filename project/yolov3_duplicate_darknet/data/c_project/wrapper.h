#include "image.h"
#include "utils.h"
#include <time.h>

typedef struct Config{
    int w;
    int h;
    float jitter;
    float hue;
    float saturation;
    float exposure;
} Config;


typedef struct{
    int id;
    float x,y,w,h;
//    float left, right, top, bottom;
} BBox;


Image image_prehandle(Image* img, Config* config);
