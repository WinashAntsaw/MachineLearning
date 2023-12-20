#include "layer.h"
#include "neural-network.h"
#include <stdlib.h>
#include <stdio.h>
#include "main.h"
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>

unsigned short init_Network(int image_height, int image_width, Network* network, int size1,...) {
	va_list ap, ap2;
	int in_size, out_size;
	int counter = 0;
	va_start(ap, network);
	in_size = size1;
	network->image_height = image_height;
	network->image_width = image_width;

	//figure out how many sizes are given, so we can allocate the right amount of layers
	out_size = va_arg(ap, int);
	printf("%d", out_size);
	fflush(stdout);
	while (out_size != 0) {
		counter++;
		out_size = va_arg(ap, int);
	}

	//if counter is 0, not enough parameters were given into the list
	if (counter == 0) {
		printf("Neural Network can not be initialized with only 1 value.\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	//allocate memory for the Neural Network
	network->layers = malloc(sizeof(*(network->layers)) * counter);
	network->in_size = size1;
	network->num_layers = counter;

	network->num_con_layers = 0;
	network->con_layers = NULL;

	va_start(ap2, network);
	out_size = va_arg(ap2, int);
	counter = 0;
	while (out_size != 0) {
		init_Layer(network->layers + counter, in_size, out_size);
		in_size = out_size;
		network->out_size = out_size;
		out_size = va_arg(ap2, int);
		counter++;
	}
	va_end(ap);
	va_end(ap2);
	return 1;
}

unsigned short set_Convolutions(Network* network, int filter_size) {
	int i;
	Convolution* layer;

	printf("gets here");
	fflush(stdout);

	if (network->con_layers == NULL) {
		network->con_layers = malloc(sizeof(*(network->con_layers)));
	}
	else {
		network->con_layers = realloc(network->con_layers, sizeof(*(network->con_layers)) * (network->num_con_layers + 1));
	}

	if (network->con_layers == NULL) {
		perror("Allocation error in setting Convolutional Layers");
		fprintf(stderr, "Enter any character to exit.\n");
		getchar();
		exit(1);
	}
	
	layer = network->con_layers + network->num_con_layers;
	init_Convolution(layer, filter_size);
	(network->num_con_layers)++;
	return 1;
}

//NOTE: must guarantee that input parameter is dynamically allocated
parameter* calculate_output(Network* network, char* input) {
	int i;
	parameter* out = malloc(sizeof(*out) * network->in_size);
	if (out == NULL) {
		perror("calculate_output error");
		exit(1);
	}

	//copies and casts all values from the char* input into the out parameter
	for (i = 0; i < network->in_size; i++) {
		//some different options for some basic preprocessing of image
		//out[i] = (16 * ((float) input[i]) / (MAX_VALUE)) - 8;
		//out[i] = activation_function((parameter)(input[i])); THIS CAUSES ERRNO 34
		out[i] = ((float)input[i]) / MAX_VALUE;
		//out[i] = (input[i] > 0) ? 1.0 : 0.0;
	}
	
	for (i = 0; i < network->num_layers; i++) {
		out = calculate_next(network->layers + i, out);
	}

	return out;
}

unsigned short write_Network(Network* network, char filename[]) {
	int fd;
	int i;


	fd = _open(filename, _O_CREAT | _O_BINARY | O_RDWR, _S_IREAD | _S_IWRITE);

	if (fd == -1)
		return 0;



	//writes all the network size parameters
	_write(fd, &(network->image_height), sizeof(int));
	_write(fd, &(network->image_width), sizeof(int));
	_write(fd, &(network->in_size), sizeof(network->in_size));
	_write(fd, &(network->out_size), sizeof(network->out_size));
	_write(fd, &(network->num_layers), sizeof(network->num_layers));
	_write(fd, &(network->num_con_layers), sizeof(int));
	
	for (i = 0; i < network->num_con_layers; i++) {
		write_Convolution(network->con_layers + i, fd);
	}

	for (i = 0; i < network->num_layers; i++) {
		write_Layer(network->layers + i, fd);
	}

	_close(fd);

	return 1;
}

unsigned short extract_Network(Network* network, char filename[]) {
	int fd, i;

	fd = _open(filename, _O_BINARY | O_RDWR);

	if (fd == -1) {
		perror("Error");
		getchar();
		exit(1);
	}

	_read(fd, &(network->image_height), sizeof(int));
	_read(fd, &(network->image_width), sizeof(int));
	_read(fd, &(network->in_size), sizeof(network->in_size));
	_read(fd, &(network->out_size), sizeof(network->out_size));
	_read(fd, &(network->num_layers), sizeof(network->num_layers));
	_read(fd, &(network->num_con_layers), sizeof(int));

	network->con_layers = malloc(sizeof(*(network->con_layers)) * network->num_con_layers);
	if (network->con_layers == NULL) {
		perror("Memory allocation fail during extraction");
		getchar();
		exit(1);
	}

	for (i = 0; i < network->num_con_layers; i++) {
		extract_Convolution(network->con_layers + i, fd);
	}

	network->layers = malloc(sizeof(Layer) * network->num_layers);

	if (network->layers == NULL) {
		perror("Error during Extraction: malloc error");
		getchar();
		exit(1);
	}

	for (i = 0; i < network->num_layers; i++) {
		extract_Layer(network->layers + i, fd);
	}

	if (network->layers[0].in_size != network->in_size || network->layers[network->num_layers - 1].out_size != network->out_size) {
		perror("Invalid input or output size found in file");
		getchar();
		exit(2);
	}

	_close(fd);
	return 1;
}

void learn_individual(Network* network, char* training_image, parameter* expected) {
	int i;
	parameter* output = calculate_output(network, training_image);
	parameter* back_prop = malloc(sizeof(*back_prop) * network->out_size);
	if (back_prop == NULL) {
		perror("Error"); 
		exit((*_errno()));
	}

	for (i = 0; i < network->out_size; i++) {
		//when using sigmoid:
		//back_prop[i] = (output[i] - expected[i]) * output[i] * (1 - output[i]);
		
		//when using RELU: RELU derivative is 1 if x > 0 and 0 if x < 0, no point checking if it's equal to 0 cause it's a floating point num
		back_prop[i] = (output[i] - expected[i]) * ((output[i] > 0) ? 1 : 0);
	}

	for (i = network->num_layers - 1; i >= 0; i--) {
		back_prop = calculate_gradients(network->layers + i, back_prop);
	}
	free(back_prop);
	free(output);
}

void learn_batch(Network* network, char** training_images, parameter** expected_outputs, parameter learn_rate, int batch_size) {
	int i;
	for (i = 0; i < batch_size; i++) {
		learn_individual(network, training_images[i], expected_outputs[i]);
	}

	for (i = 0; i < network->num_layers; i++) {
		apply_gradients(network->layers + i, learn_rate / batch_size);
	}
}

void learn_batch_slow(Network* network, char** training_images, parameter** expected_outputs, parameter learn_rate, int batch_size) {
	int i, j;
	for (i = 0; i < batch_size; i++) {
		for (j = 0; j < network->num_layers; j++) {
			adjust_gradients_slow(network, network->layers + j, training_images[i], expected_outputs[i]);
		}
	}

	for (j = 0; j < network->num_layers; j++) {
		apply_gradients(network->layers + j, learn_rate / batch_size);
	}
}

float cost_average(Network* network, char** test_images, parameter** expected_outputs, unsigned int num_images) {
	parameter* prediction;
	float total_cost = 0;
	int i;
	for (i = 0; i < num_images; i++) {
		prediction = calculate_output(network, test_images[i]);
		total_cost += cost(prediction, expected_outputs[i], network->out_size);
		free(prediction);
	}

	return total_cost / num_images;
}

float cost(parameter* prediction, parameter* expected, int num_outputs) {
	float output = 0;
	int i;
	
	for (i = 0; i < num_outputs; i++) {
		output += (prediction[i] - expected[i]) * (prediction[i] - expected[i]);
	}

	return output / num_outputs;
}

void print_network_sizes(Network* network) {
	int i;
	printf("The number of layers is: %d\n", network->num_layers);
	for (i = 0; i < network->num_layers; i++) {
		printf("For Layer %d, input = %d and output = %d", i, network->layers[i].in_size, network->layers[i].out_size);
	}
}