#include "NeuralNetwork.h"
#include <string>

void NeuralNetwork::initializeNetwork(int numOfInputs, int numOfHidden, int numOfOutputs) {
	network.clear(); // Ensures empty network upon initialization call

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distribution(0.0, 1.0);

	// Hidden Layer Information
	Layer hiddenLayer;
	hiddenLayer.reserve(numOfHidden);		// Reserve memory for Hidden Layer
	for (int i = 0; i < numOfHidden; ++i) {
		Neuron neuron;
		neuron.weights.reserve(numOfInputs + 1); // Reserve memory for weights
		for (int ii = 0; ii < numOfInputs; ++ii)		
			neuron.weights.push_back(distribution(gen));
		neuron.weights.push_back(distribution(gen)); // Bias added to end.
		hiddenLayer.push_back(neuron);
	}
	network.push_back(hiddenLayer);

	// Output Layer Initialization
	Layer outputLayer;
	outputLayer.reserve(numOfOutputs);		// Reserve memory for Output Layer
	for (int i = 0; i < numOfOutputs; ++i) {
		Neuron neuron;
		neuron.weights.reserve(numOfHidden + 1);
		for (int ii = 0; ii < numOfHidden; ++ii)
			neuron.weights.push_back(distribution(gen));
		neuron.weights.push_back(distribution(gen)); // Bias added to end.
		outputLayer.push_back(neuron);
	}
	network.push_back(outputLayer);
}



double NeuralNetwork::activate(const vector<double>& weights, const vector<double>& inputs) {
	// Debugging
	if (weights.size() < inputs.size() + 1) {
		throw std::runtime_error(
			"Bad Layer config: weights = " + std::to_string(weights.size()) +
			" inputs = " + std::to_string(inputs.size()) +
			" (need inputs+1 for bias)");
	}
	double activationValue = weights.back();		// Accumulates sum for bias, weight * input
	for (size_t i = 0; i < inputs.size(); ++i) 
		activationValue += weights[i] * inputs[i];
	return activationValue;
}

double NeuralNetwork::sigmoid(double activationValue) {
	return 1.0 / (1.0 + exp(-activationValue));
}

double NeuralNetwork::sigmoidDerivative(double sigmoid) {
	return sigmoid * (1.0 - sigmoid);
}

vector<double> NeuralNetwork::forwardPropogate(vector<double>& inputs) {
	for (auto& layer : this->network) {
		vector<double> outputCollector;
		
		for (auto& neuron : layer) {
			double activationValue = activate(neuron.weights, inputs);
			neuron.output = sigmoid(activationValue);
			outputCollector.push_back(neuron.output);		// collects outputs for next layer
		}

		inputs.swap(outputCollector); // faster than inputs = outputCollector. Prevents layer from reusing same input vector

	}
	return inputs;
}

void NeuralNetwork::backwardPropogate(const vector<double>& expectedValue) {
	for (int i = static_cast<int>(network.size()) - 1; i >= 0; --i) {	// iterate layers in reverse
		auto& layer = network[i]; // might need to static cast to <size_t>
		vector<double> errors(layer.size(), 0.0);

		if (i != network.size() - 1) {
			// Summation over Hidden Layers of Neural Network
			const auto& nextLayer = network[i + 1];
			for (int ii = 0; ii < layer.size(); ++ii) {
				double error = 0.0;
				for (const auto& nextNeuron : nextLayer) 
					error += nextNeuron.weights[ii] * nextNeuron.delta;
				errors[ii] = error;
			}
		}
		else {
			// Output Layer. Error = output of Neuron - expected value.
			for (int ii = 0; ii < layer.size(); ++ii)
				errors[ii] = layer[ii].output - expectedValue[ii];
		}

		// Compute and Update Deltas for Neurons
		for (int ii = 0; ii < layer.size(); ++ii) 
			layer[ii].delta = errors[ii] * sigmoidDerivative(layer[ii].output);
	}
}

void NeuralNetwork::updateWeights(const vector<double>& inputData, double learningRate) {
	for (int i = 0; i < network.size(); ++i) {
		vector<double> inputs;
		
		// Input layer weights
		if (i == 0)
			inputs.assign(inputData.begin(), inputData.end() - 1);
		else {
			// Hidden layer / Output Layer weights: uses outputs from previous layer.
			const auto& previousLayer = network[i - 1];
			for (const auto& neuron : previousLayer)
				inputs.push_back(neuron.output);
		}

		// Update value of each Neuron's weight
		for (auto& neuron : network[i]) {
			for (int ii = 0; ii < inputs.size(); ++ii) 
				neuron.weights[ii] -= learningRate * neuron.delta * inputs[ii];
			
			// Update bias weight of Neuron
			neuron.weights.back() -= learningRate * neuron.delta;
		}
	}
}

void NeuralNetwork::trainNetwork(vector<vector<double>>& trainingData, double learningRate,
	int epochs, int numOfOutputs) {
	for (int epoch = 0; epoch < epochs; ++epoch) {
		double errorSum = 0.0;

		for (const auto& row : trainingData) {
			vector<double> inputs(row.begin(), row.end() - 1);

			// Forward propogation of Neuron Outputs
			vector<double> outputs = forwardPropogate(inputs);

			vector<double> expected(numOfOutputs, 0.0);
			int label = static_cast<int>(row.back());
			
			if (label >= 0 && label < numOfOutputs) 
				expected[static_cast<size_t>(label)] = 1.0;
			else 
				continue;

			// Calculate sum of squared Errors
			for (size_t i = 0; i < expected.size(); ++i) {
				double difference = expected[i] - outputs[i];
				errorSum += difference * difference;
			}

			// Backwards Propogation and weight update
			backwardPropogate(expected);
			updateWeights(row, learningRate);
		}
	}
}

// Make a Prediction
double NeuralNetwork::predict(vector<double>& features) {
	vector<double> outputs = forwardPropogate(features);
	auto findMax = std::max_element(outputs.begin(), outputs.end());
	return static_cast<double>(std::distance(outputs.begin(), findMax));
}

void NeuralNetwork::debugNetworkSizes() const {
	for (size_t i = 0; i < network.size(); ++i) {
		cout << "Layer " << i << ": " << network[i].size() << " neurons\n";
		if (!network[i].empty()) {
			cout << " Weights per neuron: " << network[i][0].weights.size() << "\n";
		}
	}
}

