#include "dtmgeneticAlgorithm.h"
#include <bits/stdc++.h>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

DTMGeneticAlgorithm::DTMGeneticAlgorithm(Hyperparameters hyperparameters)
{
  this->hyperparameters = hyperparameters;
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const DTMGeneticAlgorithm &dtmGeneticAlgorithm)
{
  os << "DT GENETIC ALGORITHM ALGORITHM: " << std::endl;
  os << "HYPERPARAMETERS: " << std::endl;
  os << dtmGeneticAlgorithm.hyperparameters;
  os << "POPULATION: " << std::endl;
  for (auto individual: dtmGeneticAlgorithm.population)
  {
    os << individual;
  }
  return os;
}

/******************************************************************************
* UTILITIES
******************************************************************************/

size_t DTMGeneticAlgorithm::getNewUniqueSynapseId()
{
  return uniqueSynapseIdCounter++;
}

size_t DTMGeneticAlgorithm::getNewUniqueIndividualCounter()
{
  return uniqueIndividualIdCounter++;
}

void DTMGeneticAlgorithm::initializePopulation()
{
  assert(hyperparameters.populationSize != 0);

  population.resize(hyperparameters.populationSize);

  vector<Synapse> possibleSynapses{};
  // Number of possible combinations = inputSize * outputSize
  possibleSynapses.reserve(hyperparameters.inputSize * hyperparameters.outputSize);

  // Create all possible synapses
  for (size_t inputIndex = 0; inputIndex < hyperparameters.inputSize; inputIndex++)
  {

    for (size_t outputIndex = 0; outputIndex < hyperparameters.outputSize; outputIndex++)
    {
      size_t inputNeuronId = inputIndex;
      size_t outputNeuronId = hyperparameters.inputSize + outputIndex;

      pair<size_t, size_t> neuronIdPair{inputNeuronId, outputNeuronId};

      size_t synapseId = getNewUniqueSynapseId();

      synapseIdMap[neuronIdPair] = synapseId;

      double weight = DTMUtils::randomdouble();

      Synapse synapse(synapseId, inputNeuronId, outputNeuronId, weight);

      possibleSynapses.push_back(synapse);

    }

  }

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    // Shuffle list of possible synapses
    std::shuffle(possibleSynapses.begin(), possibleSynapses.end(), std::default_random_engine());

    DTModel model(hyperparameters.inputSize,
                  hyperparameters.outputSize,
                  hyperparameters.outputActivation);

    // Each initial individual has random number of synapses going from input to output
    size_t numberOfSynapses = 1 + (rand() % possibleSynapses.size());

    for (size_t synapseIndex = 0; synapseIndex < numberOfSynapses; synapseIndex++)
    {
      // We add first <numberOfSynapses> synapses to model, since we shuffled the vector
      // It is the same as if it was drawn randomly. Its computationally expensive,
      // but we elimate possibility of collisions for large populations.
      model.addSynapse(possibleSynapses[synapseIndex], false);
    }

    population[index] = DTIndividual(getNewUniqueIndividualCounter(), 0, model);
  }

}
