#include "Perceptron.h"

Perceptron::Perceptron(double learningRate, int numOfInputs) {
	this->learning_rate = learningRate;
	this->num_of_inputs = numOfInputs;
	// Initializes weights and bias randomly
	weights_.resize(numOfInputs, 0.5);
	bias_ = 0.437;
}

double Perceptron::tanhDerivative(double x) {
	double t = tanh(x);
	// Derivative is 1 - tanh^2(x)
	return 1.0 - t * t;
}

// Forward Propogation
// Takes a vector of data and calculates summation of inputs * weights
double Perceptron::predict(const std::vector<double>& inputs) {
	double weightedSum = 0.0;
	for (int i = 0; i < num_of_inputs; ++i) {
		weightedSum += inputs[i] * weights_[i];
	}
	weightedSum += bias_;
	return tanh(weightedSum); // Apply activation function tanh to find hyperbolic tangent
}

void Perceptron::train(const std::vector<std::vector<double>>& trainingData,
	const std::vector<double>& labels, int epochs) {

	for (int epoch = 0; epoch < epochs; ++epoch) {
		double totalError = 0.0;
		for (int i = 0; i < trainingData.size(); ++i) {
			const auto& inputs = trainingData[i];
			double trueLabel = labels[i];

			// Forward Pass
			double prediction = predict(inputs);
			double error = trueLabel - prediction; // subtract prediction

			// Backpropogation and weight update
			// The derivative of the error with respect to the output (Chain Rule)
			double delta = error * tanhDerivative(prediction);

			// Update weights and bias. Aka LEARNING!
			for (int ii = 0; ii < num_of_inputs; ++ii) {
				// Adjust weight based on input, delta, and learning rate
				weights_[ii] += delta * inputs[ii] * learning_rate;
			}
			bias_ += delta * learning_rate;
			totalError += std::abs(error);
		}
		if (epoch % 1000 == 0) {
			std::cout << "Epoch" << epoch << ", Total Error: " << totalError << '\n';
		}
	}
}