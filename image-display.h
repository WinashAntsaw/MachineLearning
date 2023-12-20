#pragma once
#include "main.h"
#include <SDL.h>
#include <stdio.h>
#include "neural-network.h"

//macros for displaying images to screen in a GUI
#define PIXEL_MULTIPLIER 20

#define SCREEN_WIDTH (PIXEL_MULTIPLIER * IMAGE_DIMENSION)
#define SCREEN_HEIGHT (PIXEL_MULTIPLIER * IMAGE_DIMENSION)
#define HEIGHT_MAX 700

void draw_number(SDL_Renderer** renderer, char *image);

void display_numbers(Network* network, float **training_labels, char** training_images, int num_images);

void test_user_drawn(Network* network);

void display_network_values(Network* network, SDL_Renderer** renderer, SDL_Window** window);

void update_network_display(Network* network, SDL_Renderer* renderer, SDL_Window* window, parameter* output);