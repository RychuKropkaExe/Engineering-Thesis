#pragma once

#include <cstddef>
#include <iostream>
#include "utils.h"
#include "hyperparameters.h"
#include "dtmodel.h"
#include "dtindividual.h"
#include "gaparents.h"

using std::vector;
using std::pair;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;
using DTMUtils::MutationE;

/******************************************************************************
 * @class SimilarDTModelElements
 *
 * @brief Contains corresponding neurons and synapses found in two DTModels
 *
 * Entries at the same index in each pair of vectors describe one match.
 * Neurons are matched by depthId. Synapses are matched by the depthIds of
 * both neurons they connect.
 *
 * @public @param firstModelNeuronIds  IDs of matching neurons in the first model
 * @public @param secondModelNeuronIds IDs of matching neurons in the second model
 * @public @param firstModelSynapses   Matching synapses from the first model
 * @public @param secondModelSynapses  Matching synapses from the second model
 ******************************************************************************/
class SimilarDTModelElements
{
public:
  vector<size_t> firstModelNeuronIds;
  vector<size_t> secondModelNeuronIds;
  vector<Synapse> firstModelSynapses;
  vector<Synapse> secondModelSynapses;
};

/******************************************************************************
 * @class NonSimilarDTModelElements
 *
 * @brief Contains neurons and synapses not included in a model's similar elements
 *
 * @public @param neuronIds IDs of model neurons absent from the similar neuron IDs
 * @public @param synapses  Model synapses absent from the similar synapses
 ******************************************************************************/
class NonSimilarDTModelElements
{
public:
  vector<size_t> neuronIds;
  vector<Synapse> synapses;
};

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
  size_t getNewUniqueNeuronId();
  size_t getNewUniqueIndividualCounter();

  vector<vector<size_t>> divideIntoSpecies();

  vector<vector<GAParents>> tournamentSelection(vector<vector<size_t>> species);

  static SimilarDTModelElements findSimilarNeuronsAndSynapses(
      const DTModel &firstModel, const DTModel &secondModel);

  static NonSimilarDTModelElements findNonSimilarNeuronsAndSynapses(
      const DTModel &model,
      const vector<size_t> &similarNeuronIds,
      const vector<Synapse> &similarSynapses);

  void crossover(const vector<vector<GAParents>> &parentsLists);

  /******************************************************************************
  * MUTATIONS
  *
  * Each call makes one attempt on a valid acyclic model. Addition mutations
  * require model.isSorted. Failure leaves the individual unchanged; success
  * sorts the model and resets the individual's grace period.
  ******************************************************************************/
  bool addNeuronMutation(DTIndividual &individual);
  bool removeNeuronMutation(DTIndividual &individual);
  bool addSynapseMutation(DTIndividual &individual);
  bool removeSynapseMutation(DTIndividual &individual);
  bool mutate(DTIndividual &individual, MutationE mutation);

private:
  void synchronizeMutationIds(const DTModel &model);

  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  size_t uniqueSynapseIdCounter{0};
  size_t uniqueNeuronIdCounter{0};
  size_t uniqueIndividualIdCounter{0};

};
