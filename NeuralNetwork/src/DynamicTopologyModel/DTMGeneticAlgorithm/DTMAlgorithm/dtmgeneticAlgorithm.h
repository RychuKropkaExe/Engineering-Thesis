#pragma once

#include <cstddef>
#include <iostream>
#include "utils.h"
#include "hyperparameters.h"
#include "dtmodel.h"
#include "dtindividual.h"

using std::vector;
using std::pair;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @class DTMGeneticAlgorithm
 *
 * @brief Represents a single synapse connecting two neurons in a dynamic topology network
 *
 * @public @param hyperparameters Hyperparameters
 * @public @param population      List of individuals
 * @public @param synapseIdMap    Map to unfiy synapses id across individuals.
 *                                Maps pair of neuron input and output id to
 *                                synapse id.
 ******************************************************************************/
class DTMGeneticAlgorithm
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  Hyperparameters hyperparameters;

  vector<DTIndividual> population;

  size_t currentGeneration{0};

  map<pair<size_t, size_t>, size_t> synapseIdMap;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  DTMGeneticAlgorithm(Hyperparameters hyperparameters);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const DTMGeneticAlgorithm &dtmGeneticAlgorithm);

  /******************************************************************************
  * UTILITIES
  ******************************************************************************/
  void initializePopulation();
  size_t getNewUniqueSynapseId();
  size_t getNewUniqueIndividualCounter();

private:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  size_t uniqueSynapseIdCounter{0};
  size_t uniqueIndividualIdCounter{0};

};
