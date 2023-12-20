#include "layer.h"
#include <stdlib.h>
#include <Windows.h>
#include <stdio.h>
#include "neural-network.h"
#include <SDL.h>
#include "main.h"
#include "image-display.h"
#include <time.h>
#include <io.h>
#include <fcntl.h>

static void free_images(char*** images, int num_images);

static void free_labels(float*** labels, int num_images);

static int get_rand_between(int i, int j);

#define TESTING_GRADIENT_DESCENT 0
#define TESTING_RANDOM 0
int main(int argc, char* argv) {
	DWORD filegeterror;

	int i, j;
	//pointer to arrays of size NUM_PIXELS to hold each image
	char** training_images;
	float** training_labels;

	char** test_images;
	float** test_labels;

	unsigned int training_num;
	unsigned int test_num;

	int num_iterations = 1800;

	char curr_directory[BUFSIZ];

	//char file[] = "C:\\Users\\Owner\\Desktop\\MachineLearning\\network-slow-test-6";
	char file[] = "..\\..\\network-slow-test-6";
	char* network_file = file;

	Network* nn;

	float error, error_prev;

	clock_t timer;

	int previous_batch = 0;

	int arr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	parameter* output;

	float learn_rate;

#ifdef NDEBUG
	printf("RELEASE MODE\n");
#endif

#ifdef _DEBUG
	printf("DEBUG MODE\n");
	network_file = strchr(file, 'n');
#endif


	training_num = extract_images(&training_images, "C:\\Users\\Owner\\Desktop\\MachineLearning\\train-images.idx3-ubyte");

	extract_labels(&training_labels, "C:\\Users\\Owner\\Desktop\\MachineLearning\\train-labels.idx1-ubyte");
	printf("Extraction of training data complete.\n");

	test_num = extract_images(&test_images, "C:\\Users\\Owner\\Desktop\\MachineLearning\\t10k-images.idx3-ubyte");

	extract_labels(&test_labels, "C:\\Users\\Owner\\Desktop\\MachineLearning\\t10k-labels.idx1-ubyte");
	printf("Extraction of test data is complete.\n");
	fflush(stdout);		

	nn = malloc(sizeof(*nn));

	printf("Enter in 'y' to extract the network from a file, and 'n' to randomize the weights and biases.\n");
	
	if (getchar() == 'y') {
		extract_Network(nn, network_file);
	}
	else {
		init_Network(IMAGE_DIMENSION, IMAGE_DIMENSION, nn, NUM_PIXELS, 32, 16, NUM_OUTPUT, 0);
		printf("gets here");
		fflush(stdout);
		set_Convolutions(nn, 3);
	}
	


#if TESTING_GRADIENT_DESCENT
	output = calculate_output(nn, training_images[0]);
	
	printf("Prediction:\n");
	printf("[");
	for (i = 0; i < NUM_OUTPUT; i++) {
		printf("%.3f ", output[i]);
	}
	printf("]\n");

	printf("Actual:\n[");
	for (i = 0; i < NUM_OUTPUT; i++) {
		printf("%.3f ", training_labels[0][i]);
	}
	printf("]\n");
	
	printf("FOR LAST LAYER:\n");
	printf("Biases:\n");
	print_biases(nn->layers + nn->num_layers - 1);
	printf("Weights:\n");
	print_weights(nn->layers + nn->num_layers - 1);
	
	learn_batch(nn, training_images, training_labels, 1, 1);

	printf("Bias Gradients:\n");
	print_bias_gradients(nn->layers + nn->num_layers - 1);
	printf("Gradients:\n");
	print_weight_gradients(nn->layers + nn->num_layers - 1);

	printf("Biases:\n");
	print_biases(nn->layers + nn->num_layers - 1);
	printf("Weights:\n");
	print_weights(nn->layers + nn->num_layers - 1);
	

	
	

	/*
	printf("FOR SECOND TO LAST LAYER:\n");
	printf("Gradients:\n");
	print_weight_gradients(nn->layers + nn->num_layers - 2);
	printf("Weights:\n");
	print_weights(nn->layers + nn->num_layers - 2);*/
	free(output);
	output = calculate_output(nn, training_images[0]);
	printf("Prediction:\n");
	printf("[");
	for (i = 0; i < NUM_OUTPUT; i++) {
		printf("%.3f ", output[i]);
	}
	printf("]\n");

	printf("Actual:\n[");
	for (i = 0; i < NUM_OUTPUT; i++) {
		printf("%.3f ", training_labels[0][i]);
	}
	printf("]\n");

#endif

#if TESTING_RANDOM

	
	for (i = 0; i < 10; i++) {
		print_int_arr(arr, 10);
		randomize_order_ints(arr, 10);
		printf("%d\n", get_rand_between(0, 10));
	}
	
#endif

	test_user_drawn(nn);

	/*
	error = cost_average(nn, test_images, test_labels, test_num);
	printf("The average cost currently is: %f.\n", error);
	fflush(stdout);*/

	randomize_order(test_images, test_labels, test_num);

	display_numbers(nn, test_labels, test_images, test_num);
	//print_network_sizes(nn);
	//printf("\n");
	error_prev = cost_average(nn, test_images, test_labels, test_num);

	printf("Enter learning rate: ");
	scanf("%f", &learn_rate);
	printf("Learn rate selected is %f.\n", learn_rate);

	randomize_order(training_images, training_labels, training_num);
	for (j = 0; j < num_iterations; j++) {
		timer = clock();
		error = cost_average(nn, test_images, test_labels, test_num);
		printf("The average cost currently is: %f.\n", error);

		for (i = previous_batch * BATCH_SIZE; i < training_num; i += BATCH_SIZE) {
			learn_batch(nn, training_images + i, training_labels + i, learn_rate, BATCH_SIZE);
		}

		randomize_order(training_images, training_labels, training_num);
		printf("Iteration #%d complete.", j + 1);
		if (write_Network(nn, network_file) == 0) {
			perror("Couldn't save network");
			exit(1);
		}
		else {
			timer = clock() - timer;
			printf(" Saved Network. Time taken: %f seconds\n", ((float)timer) / CLOCKS_PER_SEC);
		}
	}

	error = cost_average(nn, test_images, test_labels, test_num);
	printf("The new average cost is: %f.\n", error);
	fflush(stdout);

	

	//display_numbers(nn, test_labels, test_images, test_num);

	printf("exit?");
	fflush(stdout);
	scanf("exit");

	return EXIT_SUCCESS;
}

//frees the parameter image pointer and also sets it to NULL so caller avoids accessing freed memory
static void free_images(char*** images, int num_images) {
	int i;
	for (i = 0; i < num_images; i++) {
		free((*images)[i]);
	}
	free(*images);
	*images = NULL;
}

//frees the parameter labels pointer and also sets it to NULL so caller avoids accessing freed memory
static void free_labels(float ***labels, int num_images) {
	int i;
	for (i = 0; i < num_images; i++) {
		free((*labels)[i]);
	}
	free(*labels);
	*labels = NULL;
}

unsigned int extract_images(char*** images, char* filename){
	FILE* input;
	int i, j;
	unsigned int num_images;
	//opens stream for images
	fopen_s(&input, filename, "rb");

	if (input == NULL) {
		return 0;
	}

	for (i = 0; i < 4; i++)
		fgetc(input);
	

	//gets the number of images in the training set;
	num_images = (fgetc(input) << 24) | (fgetc(input) << 16) | (fgetc(input) << 8) | fgetc(input);

	//bypasses unnecessary bytes
	for (i = 0; i < 8; i++)
		fgetc(input);

	//allocates enough memory for training images to hold each image
	//NOTE: the sizeof only works because this is in same context as training images is defined, so it is the size of a 28x28 char array

	//mallocs the array of pointers, and performs null check
	*images = malloc(sizeof(char*) * num_images);
	if (*images == NULL) {
		perror("Error");
		exit(errno);
	}

	//inserts data of all images into training_images
	for (i = 0; i < num_images; i++) {
		(*images)[i] = malloc(NUM_PIXELS);
		if ((*images)[i] == NULL) {
			perror("Error");
			exit(errno);
		}

		for (j = 0; j < NUM_PIXELS; j++) {
			(*images)[i][j] = fgetc(input);
		}
	}

	//closes the stream to the training images, then opens stream to training labels
	fclose(input);
	return num_images;
}

unsigned int extract_labels(float*** labels, char* filename) {
	FILE* input;
	int i;
	char c;
	unsigned int num_labels;
	fopen_s(&input, filename, "rb");

	if (input == NULL){
		return 0;
	}

	//bypasses unnecessary bytes in header section of file
	for (i = 0; i < 4; i++) {
		fgetc(input);
	}

	num_labels = (fgetc(input) << 24) | (fgetc(input) << 16) | (fgetc(input) << 8) | fgetc(input);

	//mallocs num_labels amount of pointers to floats, and performs NULL check
	*labels = malloc(sizeof(**labels) * num_labels);
	if (*labels == NULL) {
		perror("Error");
		exit(EXIT_FAILURE);
	}

	//inserts training label data into the training_labels, in the form of a float array of 0s with 1 at the index of the label number
	//ex: if the label was 5, the corresponding entry would be [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
	for (i = 0; i < num_labels; i++) {
		(*labels)[i] = calloc(NUM_OUTPUT, sizeof(***labels));
		if ((*labels)[i] == NULL) {
			perror("Error");
			exit(errno);
		}

		c = fgetc(input);
		(*labels)[i][c] = 1.0f;
	}
	fclose(input);

	return num_labels;
}

void randomize_order(char** images, float** labels, unsigned int num) {
	int j, i;
	char* temp_image;
	float* temp_label;
	for (j = num - 1; j > 0; j--) {
		i = get_rand_between(0, j);
		//swaps elements i and j
		temp_image = images[i];
		temp_label = labels[i];
		images[i] = images[j];
		labels[i] = labels[j];
		images[j] = temp_image;
		labels[j] = temp_label;
	}
}

void randomize_order_ints(int* arr, unsigned int num) {
	int i, j, temp;
	for (j = num - 1; j > 0; j--) {
		i = get_rand_between(0, j);
		temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
}

void print_int_arr(int* arr, unsigned int num) {
	int i;
	printf("[");
	for (i = 0; i < num; i++) {
		if (i)
			printf(", ");
		printf("%d", arr[i]);
	}
	printf("]\n");
}

//j must be greater than i
//i is included as an option, j is not
static int get_rand_between(int i, int j) {
	int output;
	time_t t;

	if (j <= i) {
		fprintf(stderr, "first num can not be greater than or equal to second.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	srand((unsigned int) time(&t));
	output = (int)(((j - i) * rand()) / RAND_MAX) + i;

	output = (int)((j - i) * (((float)rand()) / RAND_MAX)) + i;
	return output;
}

void wait_till_line(void) {
	char buffer[BUFSIZ];
	while (fgets(buffer, BUFSIZ, stdin) == NULL) {
		;
	}
}