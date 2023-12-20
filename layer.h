#pragma once

#define RELU(input) (((input) > 0)? (input) : 0)

#define RELU_DERIV(input) (((input) > 0)? 1 : 0)

typedef float parameter;

typedef struct layer {
	//arrays to contain gradients for each layer
	parameter** weight_gradients;
	parameter* bias_gradients;

	//array to contain the inputs from the previous layer
	parameter* inputs;

	//arrays to contain parameters of layer
	parameter** weights;
	parameter* biases;

	//values to determine input and output size
	int in_size;
	int out_size;
} Layer;

typedef struct convolutional_layer {

	parameter* filter;
	parameter* inputs;
	parameter* gradients;
	
	int in_size;
	int filter_size;
} Convolution;

typedef struct neural_network Network;

unsigned short init_Layer(Layer* input, int in_size, int out_size);

unsigned short init_Convolution(Convolution* input, int filter_dimension);

unsigned short write_Layer(Layer* layer, int fd);

unsigned short write_Convolution(Convolution* input, int fd);

unsigned short extract_Layer(Layer* layer, int fd);

unsigned short extract_Convolution(Convolution* input, int fd);

parameter activation_function(parameter input);

parameter* calculate_next(Layer* layer, parameter* input);

void apply_gradients(Layer* input, parameter learn_rate);

parameter* calculate_gradients(Layer* layer, parameter* cost_derivative);

void adjust_gradients_slow(Network* network, Layer* layer, char* image, float* expected);

void clear_gradients(Layer* layer);

void print_weights(Layer* layer);

void print_weight_gradients(Layer* layer);

void print_biases(Layer* layer);

void print_bias_gradients(Layer* layer);