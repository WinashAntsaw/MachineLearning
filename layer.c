#include "layer.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "neural-network.h"
#include <io.h>

static void malloc_check(void* ptr);

unsigned short init_Layer(Layer* input, int in_size, int out_size) {
	unsigned short result = 0;
	int i, j;
	time_t t;
	if (input != NULL) {
		srand((unsigned) time(&t));
		input->in_size = in_size;
		input->out_size = out_size;
		//allocates space for array holding inputs into layer
		input->inputs = malloc(sizeof(*input) * in_size);

		//allocate enough space for the number of rows, and then iteratively allocate enough space for the number of weights in each row
		input->weights = malloc(sizeof(*(input->weights)) * out_size);
		input->weight_gradients = malloc(out_size * sizeof(*(input->weight_gradients)));

		input->biases = malloc(sizeof(*(input->biases)) * out_size);
		input->bias_gradients = calloc(out_size, sizeof(*(input->bias_gradients)));

		if (input->weight_gradients == NULL || input->bias_gradients == NULL) {
			perror("Malloc fail");
			exit(1);
		}

		if (input->weights != NULL && input->biases != NULL) {
			for (i = 0; i < out_size; i++) {
				(input->weights)[i] = malloc(sizeof(parameter) * in_size);
				(input->weight_gradients)[i] = calloc(in_size, sizeof(parameter));
				malloc_check((input->weights)[i]);
				malloc_check((input->weight_gradients)[i]);
				//randomize initial weights cast output of rand() to parameter so that output of division will be floating point
				for (j = 0; j < in_size; j++) {
					//gets random value between 0 and 1, multiplies it by 2 and then subtracts 1 to get random between -1 and 1
					//does not null check as it might mask errors
					(input->weights)[i][j] = (2 * (((parameter)rand()) / RAND_MAX) - 1);
				}

				input->biases[i] = (2 * (((parameter)rand()) / RAND_MAX) - 1);
			}

			result = 1;
		}
		else {
			perror("Malloc Error");
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}
	return result;
}

unsigned short init_Convolution(Convolution* input, int filter_dimension) {
	int i;
	input->filter = malloc(sizeof(*(input->filter)) * filter_dimension * filter_dimension);
	if (input->filter == NULL) {
		perror("Error in initializing convolutional layer");
		fprintf(stderr, "Please enter any character to exit.");
		getchar();
		exit(1);
	}
	srand((unsigned int)time(NULL));
	for (i = 0; i < (filter_dimension * filter_dimension); i++) {
		(input->filter)[i] = (2 * (((parameter)rand()) / RAND_MAX) - 1);
	}

	return 1;
}

unsigned short write_Convolution(Convolution* layer, int fd) {
	int i;
	_write(fd, &(layer->in_size), sizeof(layer->in_size));
	_write(fd, &(layer->filter_size), sizeof(layer->filter_size));

	for (i = 0; i < layer->filter_size; i++) {
		_write(fd, &(layer->filter[i]), sizeof(layer->filter[i]));
	}

	return 1;
}

//only to be called from write_Network, so will assume file is opened
unsigned short write_Layer(Layer* layer, int fd) {
	int node_out, node_in;
	_write(fd, &(layer->in_size), sizeof(layer->in_size));
	_write(fd, &(layer->out_size), sizeof(layer->out_size));
	for (node_out = 0; node_out < layer->out_size; node_out++) {
		_write(fd, &(layer->biases[node_out]), sizeof(parameter));
	}
	for (node_out = 0; node_out < layer->out_size; node_out++) {
		for (node_in = 0; node_in < layer->in_size; node_in++) {
			_write(fd, &(layer->weights[node_out][node_in]), sizeof(parameter));
		}
	}
	return 1;
}

unsigned short extract_Convolution(Convolution* layer, int fd) {
	int i;

	_read(fd, &(layer->in_size), sizeof(layer->in_size));
	_read(fd, &(layer->filter_size), sizeof(layer->filter_size));

	layer->filter = malloc(sizeof(*layer->filter) * layer->filter_size * layer->filter_size);

	for (i = 0; i < (layer->filter_size * layer->filter_size); i++) {
		_read(fd, &(layer->filter[i]), sizeof(layer->filter[i]));
	}
}

unsigned short extract_Layer(Layer* layer, int fd) {
	int node_out, node_in;
	_read(fd, &(layer->in_size), sizeof(layer->in_size));
	_read(fd, &(layer->out_size), sizeof(layer->out_size));

	layer->biases = malloc(sizeof(*(layer->biases)) * layer->out_size);
	layer->bias_gradients = malloc(sizeof(*(layer->bias_gradients)) * layer->out_size);
	layer->inputs = malloc(sizeof(*(layer->inputs)) * layer->in_size);

	for (node_out = 0; node_out < layer->out_size; node_out++) {
		_read(fd, &(layer->biases[node_out]), sizeof(parameter));
	}

	layer->weights = malloc(sizeof(*(layer->weights)) * layer->out_size);
	layer->weight_gradients = malloc(sizeof(*(layer->weight_gradients)) * layer->out_size);
	for (node_out = 0; node_out < layer->out_size; node_out++) {
		layer->weights[node_out] = malloc(sizeof(*(layer->weights[node_out])) * layer->in_size);
		layer->weight_gradients[node_out] = malloc(sizeof(*(layer->weight_gradients[node_out])) * layer->in_size);
		for (node_in = 0; node_in < layer->in_size; node_in++) {
			_read(fd, &(layer->weights[node_out][node_in]), sizeof(parameter));
		}
	}

	clear_gradients(layer);
	return 1;
}


void apply_gradients(Layer* input, parameter learn_rate) {
	int node_in, node_out;
	for (node_out = 0; node_out < input->out_size; node_out++) {
		input->biases[node_out] -= (input->bias_gradients[node_out] * learn_rate);
		for (node_in = 0; node_in < input->in_size; node_in++) {
			input->weights[node_out][node_in] -= ((input->weight_gradients)[node_out][node_in] * learn_rate);
		}
	}
	clear_gradients(input);
}

void clear_gradients(Layer* layer) {
	int i, j;
	for (i = 0; i < layer->out_size; i++) {
		layer->bias_gradients[i] = 0;
		for (j = 0; j < layer->in_size; j++) {
			layer->weight_gradients[i][j] = 0;
		}
	}
}

/*
Calculates the gradients given the derivative of the cost with respect to the weighted
inputs. Then propagates backward the
NOTE: SOMETHING IS WRONG IN THIS CALCULATE GRADIENTS FUNCTION. SOMETHING WEIRD GOING ON WITH THE BACKPROPAGATION, SOMETHING VERY
VERY VERY VERY VERY VERY VERY VERY BAD
P
L
E
A
S
E
F
I
X
*/
parameter* calculate_gradients(Layer* layer, parameter* cost_derivative) {
	int node_in, node_out;
	parameter* back_prop = calloc(layer->in_size, sizeof(*back_prop));
	
	if (back_prop == NULL) {
		perror("Error");
		exit(errno);
	}

	for (node_in = 0; node_in < layer->in_size; node_in++) {
		back_prop[node_in] = 0.0f;
		for (node_out = 0; node_out < layer->out_size; node_out++) {
			layer->weight_gradients[node_out][node_in] += cost_derivative[node_out] * layer->inputs[node_in];
			back_prop[node_in] += layer->weights[node_out][node_in] * cost_derivative[node_out];
		}

		//when using sigmoid:
		//back_prop[node_in] *= (layer->inputs[node_in] * (1 - layer->inputs[node_in]));
		//when using relu:
		back_prop[node_in] *= RELU_DERIV(layer->inputs[node_in]);
	}

	for (node_out = 0; node_out < layer->out_size; node_out++) {
		layer->bias_gradients[node_out] += cost_derivative[node_out];
	}
	free(cost_derivative);
	return back_prop;
}

void adjust_gradients_slow(Network* network, Layer* layer, char* image, float* expected) {
	parameter* output;
	parameter* output2;
	float h = 0.001f;
	int node_in, node_out;
	float cost1, cost2;
	
	output = calculate_output(network, image);
	cost1 = cost(output, expected, network->out_size);
	for (node_out = 0; node_out < layer->out_size; node_out++) {
		//printf("%f\n", layer->biases[node_out]);
		layer->biases[node_out] += h;
		//printf("%f\n", layer->biases[node_out]);
		output2 = calculate_output(network, image);
		cost2 = cost(output2, expected, network->out_size);
		layer->bias_gradients[node_out] += (cost2 - cost1) / h;
		layer->biases[node_out] -= h;
		free(output2);

		for (node_in = 0; node_in < layer->in_size; node_in++) {
			layer->weights[node_out][node_in] += h;
			output2 = calculate_output(network, image);
			cost2 = cost(output2, expected, network->out_size);

			layer->weight_gradients[node_out][node_in] += (cost2 - cost1) / h;
			free(output2);
			layer->weights[node_out][node_in] -= h;
		}
	}

	free(output);
}

static void malloc_check(void* ptr) {
	if (ptr == NULL) {
		perror("Malloc Error");
		exit(errno);
	}
}


parameter activation_function(parameter input) {
	//NOTE: assumes that parameter typedef is to float
	return 1.0 / (1.0 + expf(-1 * input));
}




/*
Calculates the output of this layer given input which will likely be output of
the previous layer.
NOTE: This function frees the parameter passed into it, so that we can iteratively
set a pointer to the output of this function and then pass it back in. Freeing 
inside the function will avoid memory leaks.
*/
parameter* calculate_next(Layer* layer,parameter* input) {
	int i, j;
	parameter* out_arr = calloc(layer->out_size, sizeof(*out_arr));

	if (out_arr == NULL) {
		perror("Error in calculate_next");
		exit(errno);
	}

	for (i = 0; i < layer->in_size; i++) {
		layer->inputs[i] = input[i];
	}

	for (i = 0; i < layer->out_size; i++) {
		for (j = 0; j < layer->in_size; j++) {
			out_arr[i] += layer->weights[i][j] * input[j];
		}
		out_arr[i] += layer->biases[i];

		//when using sigmoid
		//out_arr[i] = activation_function(out_arr[i]);
		out_arr[i] = RELU(out_arr[i]);
	}
	free(input);
	return out_arr;
}

void print_weights(Layer* layer) {
	int i, j;
	for (i = 0; i < layer->out_size; i++) {
		printf("[");
		for (j = 0; j < layer->in_size; j++) {
			if (j > 0)
				printf("|");
			printf("%.3f", layer->weights[i][j]);
		}
		printf("]\n");
	}
}

void print_weight_gradients(Layer* layer) {
	int i, j;
	for (i = 0; i < layer->out_size; i++) {
		printf("[");
		for (j = 0; j < layer->in_size; j++) {
			if (j > 0)
				printf(", ");
			printf("%.3f", layer->weight_gradients[i][j]);
		}
		printf("]\n");
	}
}

void print_biases(Layer* layer) {
	int i;
	printf("[");
	for (i = 0; i < layer->out_size; i++) {
		printf("%.3f, ", layer->biases[i]);
	}
	printf("]\n");
}

void print_bias_gradients(Layer* layer) {
	int i;
	printf("[");
	for (i = 0; i < layer->out_size; i++) {
		printf("%.3f, ", layer->bias_gradients[i]);
	}
	printf("]\n");
}