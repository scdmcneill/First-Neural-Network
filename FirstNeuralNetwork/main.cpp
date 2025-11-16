#include <iostream>
#include "NeuralNetwork.h"
#include "ImageInput.h"

int main() {
	try {
		// Load and flatten image
		auto x = loadImage400x400("testdog.png");

		NeuralNetwork neuralNetwork;

		neuralNetwork.initializeNetwork(400 * 400, 16, 10);

		std::cout << "x.size() = " << x.size() << "\n";
		neuralNetwork.debugNetworkSizes();

		auto y = neuralNetwork.forwardPropogate(x);
		std::cout << "pixels: " << x.size() << "\n";
		std::cout << "outputs: " << y.size() << "\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
	}
}