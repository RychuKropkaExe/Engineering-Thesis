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
 * @public @param id          Synapse id
 ******************************************************************************/
class DTModel
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/

  size_t id;
  size_t inputSize;
  size_t outputSize;

  map<size_t, size_t> indexMap;

  vector<Neuron> neurons;

  vector<double> output;

  bool isSorted;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  DTModel(size_t id, size_t inputSize, size_t outputSize, ActivationE outputActivation);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Synapse &synapse);

  /******************************************************************************
  * UTILITIES
  ******************************************************************************/
  void sortTopologically();

  void addNeuron(Neuron neuron, Synapse inSynapse, Synapse outSynapse, bool sortAfterAdding);
  void addSynapse(Synapse newSynapse, bool sortAfterAdding);

  void removeNeuron(size_t id);
  void removeSynapse(size_t inNeuronId, size_t outNeuronId);

  vector<double> feedForward(vector<double> input);

private:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  static constexpr size_t NEURON_BUFFER_INTERVAL = 5;
};
