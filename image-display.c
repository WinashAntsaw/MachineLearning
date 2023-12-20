#include <SDL.h>
#include "image-display.h"
#include <stdio.h>
#include "main.h"

void draw_number(SDL_Renderer** renderer, char *image) {
	char c;
	int i, j;
	for (i = 0; i < SCREEN_HEIGHT; i++) {
		for (j = 0; j < SCREEN_WIDTH; j++) {
			c = image[IMAGE_DIMENSION * (int)(i / PIXEL_MULTIPLIER) + (int)(j / PIXEL_MULTIPLIER)];
			fflush(stdout);
			SDL_SetRenderDrawColor(*renderer, c, c, c, 255);
			SDL_RenderDrawPoint(*renderer, j, i);
		}
	}
	SDL_RenderPresent(*renderer);
}

void display_numbers(Network* network, float **training_labels, char **training_images, int num_images) {
	SDL_Window* window = NULL;
	SDL_Event event;
	SDL_Renderer* renderer;
	int i, j;
	int current_image = 0;
	char c;

	parameter* output;

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	SDL_CreateWindowAndRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, 0, &window, &renderer);
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	draw_number(&renderer, training_images[0]);
	//continually will display image and checks if we need to switch to next image
	while (1) {
		if (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT)
				break;
			else if (event.type == SDL_KEYDOWN) {
				current_image++;
				if (current_image == num_images)
					current_image = 0;
				
				output = calculate_output(network, training_images[current_image]);
				printf("Prediction: \n[ ");
				for (j = 0; j < NUM_OUTPUT; j++) {
					printf("%f ", output[j]);
				}
				printf("]\n");
				free(output);


				printf("Expected: \n[ ");
				for (j = 0; j < NUM_OUTPUT; j++) {
					printf("%f ", training_labels[current_image][j]);
				}
				printf("]\n");
				draw_number(&renderer, training_images[current_image]);
			}
		}
	}

	//cleans up window and renderer
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void test_user_drawn(Network* network) {
	SDL_Window* window = NULL;
	SDL_Event event;
	SDL_Renderer* renderer = NULL;

	SDL_Window* network_window = NULL;
	SDL_Renderer* network_renderer;


	unsigned int mouse_state;
	int i, mouse_x, mouse_y, prediction;
	char* image;

	parameter* output;
	parameter confidence;

	image = calloc(NUM_PIXELS, sizeof(*image));

	if (image == NULL) {
		perror("test_user_drawn:");
		getchar();
		exit(1);
	}
	

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	SDL_CreateWindowAndRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, 0, &window, &renderer);
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);

	display_network_values(network, &network_renderer, &network_window);

	while (1) {
		if (SDL_PollEvent(&event)) {
			mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
			if (event.type == SDL_QUIT)
				break;
			else if (SDL_BUTTON(mouse_state) == 1) {
				i = IMAGE_DIMENSION * (int)(mouse_y / PIXEL_MULTIPLIER) + (int)(mouse_x / PIXEL_MULTIPLIER);
				mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
				image[i] = 0x0ff;
				
				
				if ((i + 28) < 784) {
					image[i + 28] = (((unsigned char)image[i + 28] < 0x77)? image[i+28] + 0x022 : image[i + 28]);
				}
				if ((i - 28) >= 0) {
					image[i - 28] = (((unsigned char) image[i - 28] < 0x77) ? image[i - 28] + 0x022 : image[i - 28]);
				}
				if ((mouse_x / PIXEL_MULTIPLIER) < (IMAGE_DIMENSION - 1)) {
					image[i + 1] = (((unsigned char) image[i + 1] < 0x77) ? image[i + 1] + 0x022 : image[i + 1]);
				}

				if ((mouse_x / PIXEL_MULTIPLIER) > 0) {
					image[i - 1] = (((unsigned char) image[i - 1] < 0x77) ? image[i - 1] + 0x022 : image[i - 1]);
				}
				output = calculate_output(network, image);
				update_network_display(network, network_renderer, network_window, output);
				free(output);
				draw_number(&renderer, image);
			}
			
			else if (event.type == SDL_KEYDOWN) {
				if (event.key.keysym.scancode == SDL_SCANCODE_BACKSPACE) {
					for (i = 0; i < NUM_PIXELS; i++) {
						image[i] = 0;
					}

					draw_number(&renderer, image);
				}
				else if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
					break;
				else {
					output = calculate_output(network, image);
					//update_network_display(network, network_renderer, network_window);

					confidence = 0;
					prediction = 0;

					printf("[");
					for (i = 0; i < NUM_OUTPUT; i++) {
						printf("%f, ", output[i]);
					}
					printf("]\n");

					for (i = 0; i < NUM_OUTPUT; i++) {
						if (output[i] > confidence) {
							confidence = output[i];
							prediction = i;
						}
					}
					confidence *= 100;

					printf("The Neural Network's prediction is: %d with a confidence of %.2f%%.\n", prediction, confidence);
					update_network_display(network, network_renderer, network_window, output);
					free(output);
					
				}

			}
		}
	}

	//cleans up window and renderer
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_DestroyRenderer(network_renderer);
	SDL_DestroyWindow(network_window);
	SDL_Quit();
}

void display_network_values(Network* network, SDL_Renderer** renderer, SDL_Window** window) {
	int in_max = 0;
	int i, j, k, l;

	//gets the largest number of inputs in a layer, stores it in in_max
	for (i = 1; i < network->num_layers; i++) {
		if (network->layers[i].in_size > in_max) {
			in_max = network->layers[i].in_size;
		}
	}

	SDL_CreateWindowAndRenderer((2 * network->num_layers - 1) * PIXEL_MULTIPLIER, in_max * PIXEL_MULTIPLIER, 0, window, renderer);
	SDL_SetRenderDrawColor(*renderer, 0, 0, 0, 255);

	for (i = 0; i < (2 * network->num_layers - 1) * PIXEL_MULTIPLIER; i++) {
		for (j = 0; j < in_max * PIXEL_MULTIPLIER; j++) {
			SDL_RenderDrawPoint(*renderer, i, j);
		}
	}

	SDL_RenderPresent(*renderer);
}

void update_network_display(Network* network, SDL_Renderer* renderer, SDL_Window* window, parameter* output) {
	int i, j, k, l;
	Uint8 color;

	for (i = 1; i < network->num_layers; i++) {
		for (j = 0; j < network->layers[i].in_size; j++) {
			color = network->layers[i].inputs[j] * 255;
			SDL_SetRenderDrawColor(renderer, color, color, color, 255);
			for (k = 0; k < PIXEL_MULTIPLIER; k++) {
				for (l = 0; l < PIXEL_MULTIPLIER; l++) {
					SDL_RenderDrawPoint(renderer, k + (2 * PIXEL_MULTIPLIER * (i - 1)), l + j * PIXEL_MULTIPLIER);
				}
			}
		}
	}

	for (j = 0; j < network->layers[i - 1].out_size; j++) {
		color = output[j] * 255;
		SDL_SetRenderDrawColor(renderer, color, color, color, 255);
		for (k = 0; k < PIXEL_MULTIPLIER; k++) {
			for (l = 0; l < PIXEL_MULTIPLIER; l++) {
				SDL_RenderDrawPoint(renderer, k + (2 * PIXEL_MULTIPLIER * (i - 1)), l + j * PIXEL_MULTIPLIER);
			}
		}
	}

	SDL_RenderPresent(renderer);
}