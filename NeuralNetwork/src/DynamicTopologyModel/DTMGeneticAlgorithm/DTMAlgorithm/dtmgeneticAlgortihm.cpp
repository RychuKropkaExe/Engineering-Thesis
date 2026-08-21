#include "dtmgeneticAlgorithm.h"
#include <bits/stdc++.h>
#include <cassert>

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

vector<vector<size_t>> DTMGeneticAlgorithm::divideIntoSpecies()
{

  size_t maxDepth = 0;

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    if (population[index].model.maxDepth > maxDepth)
    {
      maxDepth = population[index].model.maxDepth;
    }
  }

  const size_t bufferInterval = hyperparameters.populationSize / maxDepth;

  vector<vector<size_t>> speciesIndexes;

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    speciesIndexes[index].reserve(bufferInterval);
  }

  for (size_t index = 0; index < hyperparameters.populationSize; index++)
  {
    size_t individualDepth = population[index].model.maxDepth;
    if (speciesIndexes[individualDepth].size() == speciesIndexes[individualDepth].capacity())
    {
      speciesIndexes[individualDepth].reserve(speciesIndexes[individualDepth].size() + bufferInterval);
    }

    speciesIndexes[individualDepth].push_back(index);
  }

  return speciesIndexes;
}

vector<vector<GAParents>> DTMGeneticAlgorithm::tournamentSelection(vector<vector<size_t>> species)
{

  vector<vector<GAParents>> parentsLists;

  for (size_t index = 0; index < species.size(); index++)
  {
    parentsLists.reserve(species[index].size());
  }

  // Fast forward protected individuals
  for (size_t index = 0; index < species.size(); index++)
  {
    for (auto individualIndex : species[index])
    {
      if (population[individualIndex].gracePeriodLength > 0)
      {
        // Pairs of parents with the same individual
        // are processed by just copying the individual to the next generation
        GAParents parents;
        parents.addParent(individualIndex);
        parents.addParent(individualIndex);
        parentsLists[index].push_back(parents);
        population[individualIndex].gracePeriodLength--;
      }
    }
  }


  for (size_t index = 0; index < parentsLists.size(); index++)
  {
    while(parentsLists[index].size() != parentsLists[index].capacity())
    {

      constexpr size_t numberOfParents = 2;

      GAParents parents;

      for (size_t _ = 0; _ < numberOfParents; _++)
      {

        vector<size_t> tournament;

        tournament.reserve(hyperparameters.tournamentSize);

        while (tournament.size() != tournament.capacity())
        {
          size_t individualIndex = rand() % species[index].size();
          tournament.push_back(individualIndex);
        }

        size_t bestIndividualIndex = 0;

        size_t bestFitness = 0;

        for (auto individualIndex : tournament)
        {
          if (population[individualIndex].fitness > bestFitness)
          {
            bestFitness = population[individualIndex].fitness;
            bestIndividualIndex = individualIndex;
          }
        }

        parents.addParent(bestIndividualIndex);

      }

      parentsLists[index].push_back(parents);


    }
  }

  return parentsLists;

}
