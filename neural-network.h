#pragma once
#include <stdarg.h>
#include "layer.h"


typedef struct neural_network{
	int in_size;
	int out_size;
	int num_layers;
	Layer* layers;

	int image_height;
	int image_width;

	int num_con_layers;
	Convolution* con_layers;
} Network;

unsigned short init_Network(int image_height, int image_width, Network* network, int size1, ...);

unsigned short set_Convolutions(Network* network, int filter_size);

unsigned short write_Network(Network* network, char filename[]);

unsigned short extract_Network(Network* network, char filename[]);

parameter* calculate_output(Network* network, char* input);

void print_network_sizes(Network* network);

void learn_individual(Network* network, char* training_image, parameter* expected);

void learn_batch(Network* network, char** training_images, parameter** expected_outputs, parameter learn_rate, int batch_size);

void learn_batch_slow(Network* network, char** training_images, parameter** expected_outputs, parameter learn_rate, int batch_size);

float cost_average(Network* network, char** test_images, parameter** expected_outputs, unsigned int num_images);

float cost(parameter* prediction, parameter* expected, int num_outputs);