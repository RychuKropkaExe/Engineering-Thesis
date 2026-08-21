#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include "utils.h"
#include "neuron.h"
#include "synapse.h"

using std::map;

/******************************************************************************
 * @class DTModel
 *
 * @brief Represents a model with changing topology
 *
 * @public @param inputSize   Number of inputs
 * @public @param outputSize  Number of outputs
 * @public @param maxDepth    Maximum depth of a neuron
 * @public @param indexMap    Mapping between neurons IDs and their indexes in list
 * @public @param neurons     List of neurons
 * @public @param isSorted    Indicates if neural network is topologically sorted at
 *                            the moment
 *
 ******************************************************************************/
class DTModel
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/

  size_t inputSize;
  size_t outputSize;
  size_t maxDepth;

  map<size_t, size_t> indexMap;

  vector<Neuron> neurons;

  bool isSorted;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  DTModel() = default;
  DTModel(size_t inputSize, size_t outputSize, ActivationE outputActivation);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const DTModel &dtmodel);

  /******************************************************************************
  * UTILITIES
  ******************************************************************************/
  void sortTopologically();

  void addNeuron(Neuron neuron, Synapse inSynapse, Synapse outSynapse, bool sortAfterAdding);
  void addSynapse(Synapse newSynapse, bool sortAfterAdding);

  void setBias(size_t neuronId, double value);

  void removeNeuron(size_t id, bool sortAfterRemove);
  void removeSynapse(size_t inNeuronId, size_t outNeuronId, bool sortAfterRemoveal);

  bool validateModel();

  vector<double> feedForward(vector<double> input);

private:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  static constexpr size_t NEURON_BUFFER_INTERVAL = 5;
};
