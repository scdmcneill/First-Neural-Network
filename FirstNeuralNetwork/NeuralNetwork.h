#pragma once

#include <random>
#include <iostream>
#include <vector>

struct Neuron {
	std::vector<double> weights;
	double output;
	double error;
	double delta;
};

using Layer = std::vector<Neuron>;
using Network = std::vector<Layer>;

using std::vector;
using std::cout;

class NeuralNetwork {
public:
	// numOfInputs = number of inputs for each Neuron.
	// numOfHidden = number of  desired Neurons in Hidden Layer.
	// numOfOutputs = number of desired Neurons in Output Layer.		/remember input is 400x400 h is 32-128 o is 2 (dog/cat)
	void initializeNetwork(int numOfinputs, int numOfHidden, int numOfOutputs);
	
	// Neuron activation calculated as weighted sum of inputs.
	double activate(const vector<double>& weights, const vector<double>& inputs);

	// Sigmoid activation function. f(x) = 1 / 1 (1 + e^-x) 
	double sigmoid(double activationValue);

	// Returns the derivative for sigmoid value. Used in Error Calculation.
	double sigmoidDerivative(double sigmoid);

	// Forward Propogation
	// Calculates output from Neural Network by propogating an input signal through
	// each layer until the output layer outputs it's values.
	vector<double> forwardPropogate(vector<double>& inputs);

	// Performs backward propogation on the Neural Network to calculate and assign 
	// error gradients (deltas) for each Neuron in all layers. Uses provided expected
	// values to adjust neuron deltas based on derivative of their activation function.
	void backwardPropogate(const vector<double>& expectedValue);

	// Updates weights and bias of each Neuron in Network using gradient descent.
	// Adjust weights based on Neuron's delta value, learning rate, and the input
	// values.
	void updateWeights(const vector<double>& dataInput, double learningRate);

	// Trains the Neural Network over multiple epochs using supervised learning.
	// Performs forward propogation, calculates prediction error, runs back propogation,
	// and updates network weights for each training sample.
	void trainNetwork(vector<vector<double>>& train, double learningRate,
		int epochs, int numOfOutputs);

	// Generates a predicted label for a given input row by performing forward
	// propogation throught the network and returning the index of the neuron
	// with the highest output value (most probable label).
	double predict(vector<double>& features);

	const Network& getNetwork() const {
		return network;
	}

	void debugNetworkSizes() const;

private:
	Network network;
};