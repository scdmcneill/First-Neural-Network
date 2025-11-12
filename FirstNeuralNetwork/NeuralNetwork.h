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
	// numOfOutputs = number of desired Neurons in Output Layer.
	void initializeNetwork(int numOfinputs, int numOfHidden, int numOfOutputs);
	
	// Forward Propogation
	// Calculates output from Neural Network by propogating an input signal through
	// each layer until the output layer outputs it's values.
	
	// Neuron activation calculated as weighted sum of inputs.
	double activate(vector<double>& weights, vector<double>& inputs);

	// Sigmoid activation function. f(x) = 1 / 1 (1 + e^-x) 
	double sigmoid(double activationValue);

	// Returns the derivative for sigmoid value. Used in Error Calculation.
	double sigmoidDerivative(double sigmoid);

	vector<double> forwardPropogate(Network& network, vector<double>& inputs);

	void backwardPropogate(Network& network, const vector<double>& expectedValue);

	void updateWeights(Network& network, vector<double>& inputRow, double learningRate);

	void trainNetwork(Network& network, vector<vector<double>>& train, double learningRate,
		int epochs, int numOfOutputs);
	
	
	// Network updateWeights(Network network, double learningRate);
	// Network trainNetwork(Network network, vector<vector<double>> trainingData,
	//	double learningRate, int epochs, int numOfOutputs);

	// Setters
	// void setWeights(double weightValue);

	// double activate(vector<double> inputs, vector<double> weights);

	// double sigmoidActivation(double activation);
	// double sigmoidDerivative(double sigmoid);
	// vector<double> forwardPropogate();
	// void backwardPropogate();
	// void trainNetwork();

	const Network& getNetwork() const {
		return network;
	}
private:
	Network network;

};