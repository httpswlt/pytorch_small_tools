#include "wrapper.h"


BBox* read_boxes(Image* img)
{
    float* data = img->bboxes;
    int n = img->bnums;
    BBox* bboxes = calloc(n, sizeof(BBox));
    int ptr = 0;
    for(int i = 0; i < n; i++)
    {
        ptr = ptr + i * 5;
        bboxes[i].x = *(data + ptr);
        bboxes[i].y = *(data + ptr + 1);
        bboxes[i].w = *(data + ptr + 2);
        bboxes[i].h = *(data + ptr + 3);
        bboxes[i].id = *(data + ptr + 4);
    }
    return bboxes;
}

void randomize_boxes(BBox *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        BBox swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(Image* img, BBox* boxes, float dx, float dy, float nw, float nh, int flip)
{
    float sx = nw / img->w;
    float sy = nh / img->h;
//    printf("dx: %f, dy: %f, sx: %f, sy: %f, flip: %d\n", dx, dy, sx, sy, flip);
    int i;
    for(i = 0; i < img->bnums; ++i){
        boxes[i].x   = boxes[i].x  * sx + dx;
        boxes[i].w  = boxes[i].w * sx;
        boxes[i].y    = boxes[i].y   * sy + dy;
        boxes[i].h = boxes[i].h* sy;

        if(flip){
            boxes[i].x = nw - (boxes[i].x + boxes[i].w);
        }
    }
}

void write_to_img(Image* img, BBox* boxes)
{
    for(int i = 0; i < img->bnums; i++)
    {
        *(img->bboxes + 0 +5 * i) = boxes[i].x;
        *(img->bboxes + 1 +5 * i) = boxes[i].y;
        *(img->bboxes + 2 +5 * i) = boxes[i].w;
        *(img->bboxes + 3 +5 * i) = boxes[i].h;
        *(img->bboxes + 4 +5 * i) = boxes[i].id;
    }
}

void fill_truth_detection(Image* img, Image* sized, float dx, float dy, float nw, float nh, int flip )
{
    BBox* boxes = read_boxes(img);
    randomize_boxes(boxes, img->bnums);
    correct_boxes(img, boxes, dx, dy, nw, nh, flip);
    write_to_img(sized, boxes);

}

Image image_prehandle(Image* img, Config* config)
{
    srand((int)time(0));
    Image sized = make_image(config->w, config->h, img->c, img->bnums);
    fill_image(sized, .5);
    float dw = config->jitter * img->w;
    float dh = config->jitter * (img->w / img->h);
    float new_ar = (img->w + rand_uniform(-dw, dw)) / (img->h + rand_uniform(-dh, dh));
    float scale = 1;

    float nw, nh;
//    printf("new_ar: %f\n", new_ar);
    if(new_ar < 1){
        nh = scale * config->h;
        nw = nh * new_ar;
    } else {
        nw = scale * config->w;
        nh = nw / new_ar;
    }
    float dx = rand_uniform(0, config->w - nw);
    float dy = rand_uniform(0, config->h - nh);

    place_image(*img, nw, nh, dx, dy, sized);

    random_distort_image(sized, config->hue, config->saturation, config->exposure);
    int flip = rand()%2;
    if(flip) flip_image(sized);

//    printf("nw: %f, nh: %f, dx: %f, dy: %f, flip: %d\n", nw, nh, dx, dy, flip);

    // correct bboxes
    fill_truth_detection(img, &sized, dx, dy, nw, nh, flip);

    return sized;
}
