#include "NeuralNetwork.h"

int main() {
	NeuralNetwork net;
	net.initializeNetwork(3, 2, 1);
	
	const auto& network = net.getNetwork();

	for (size_t ithLayer = 0; ithLayer < network.size(); ++ithLayer) {
		cout << "Layer " << ithLayer + 1 << ":\n";
		for (size_t ithNeuron = 0; ithNeuron < network[ithLayer].size(); ++ithNeuron) {
			cout << " Neuron " << ithNeuron + 1 << " weights: ";
			for (double weight : network[ithLayer][ithNeuron].weights)
				cout << weight << " ";
			cout << '\n';
		}
	}
}