#include "dtmgeneticAlgorithm.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <random>
#include <utility>

/******************************************************************************
 * @brief Initializes the population with randomly generated individuals
 *
 * Creates all possible input-to-output synapses, assigns them shared
 * identifiers, and initializes each individual with a random subset of
 * those synapses.
 ******************************************************************************/
void DTMGeneticAlgorithm::initializePopulation()
{
  TIME_MEASURE_BEGIN(DTM_POPULATION_INITIALIZATION);
  assert(hyperparameters.populationSize != 0);

  population.resize(hyperparameters.populationSize);
  synapseIdMap.clear();

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
      model.addOutSynapse(possibleSynapses[synapseIndex], false);
    }

    model.sortTopologically();
    population[index] = DTIndividual(getNewUniqueIndividualCounter(), 0, std::move(model));
  }
  TIME_MEASURE_END(DTM_POPULATION_INITIALIZATION);
}
