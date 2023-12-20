#pragma once
//define macros for info about the images to be trained and tested on
#define IMAGE_DIMENSION 28
#define NUM_PIXELS 784
#define NUM_OUTPUT 10
#define MAX_VALUE 255

//NOTE: training number must be divisible by this value
#define BATCH_SIZE 100

//multiline macro to define a check for a manually allocated memory error
#define MALLOC_CHECK(ptr) {\
if (ptr == NULL) {\
	perror("Error");\
	exit(errno);\
}\


//returns the number of images allocated
unsigned int extract_images(char*** images, char* filename);

unsigned int extract_labels(float*** labels, char* filename);

void randomize_order(char** images, float** labels, unsigned int num);

void wait_till_line(void);

void print_int_arr(int* arr, unsigned int num);

void randomize_order_ints(int* arr, unsigned int num);