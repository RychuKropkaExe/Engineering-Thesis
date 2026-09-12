#include "dtmgeneticAlgorithm.h"
#include <algorithm>

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

/******************************************************************************
 * @brief Writes the genetic algorithm state to an output stream
 *
 * @param os                  Output stream to write to
 * @param dtmGeneticAlgorithm Genetic algorithm whose state should be written
 *
 * @return The output stream after writing the genetic algorithm state
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

/******************************************************************************
 * @brief Generates and returns the next unique synapse identifier
 *
 * @return A unique synapse identifier
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueSynapseId()
{
  return uniqueSynapseIdCounter++;
}

/******************************************************************************
 * @brief Generates the next unique hidden-neuron identifier
 *
 * IDs below inputSize + outputSize are reserved for the fixed input/output
 * neurons. The counter is retained across generations; crossover also advances
 * it past neuron IDs already present in the population.
 *
 * @return A new hidden-neuron identifier outside the reserved range
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueNeuronId()
{
  uniqueNeuronIdCounter = std::max(
      uniqueNeuronIdCounter, hyperparameters.inputSize + hyperparameters.outputSize);
  return uniqueNeuronIdCounter++;
}

/******************************************************************************
 * @brief Generates and returns the next unique individual identifier
 *
 * @return A unique individual identifier
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueIndividualCounter()
{
  return uniqueIndividualIdCounter++;
}

