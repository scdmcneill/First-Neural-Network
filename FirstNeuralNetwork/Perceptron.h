#pragma once

#include <vector>
#include <cmath>
#include <iostream>

// ==========================================================================================================
//												Neural Network Machine Learning
//												Perceptron Class
//												Author: Scott McNeill
// ==========================================================================================================

class Perceptron {
public:
	// Constructor initializes weights and learning rate
	Perceptron(double learningRate, int numOfInputs) {}

	// Returns derivative of tanh function
	double tanhDerivative(double x);

	// Forward Propogation
	// Takes a vector of data and calculates sumation of inputs * weights
	double predict(const std::vector<double>& inputs);

	// Training Function
	void train(const std::vector<std::vector<double>>& trainingData,
		const std::vector<double>& labels, int epochs);

private:
	double learning_rate; // Learning Rate Hyperparameter
	int num_of_inputs;
	std::vector<double> weights_; // Network Parameters
	double bias_;// Network Parameter
};