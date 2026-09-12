#include "dtmodel.h"
#include <cassert>
#include <utility>

/******************************************************************************
* CONSTRUCTORS
******************************************************************************/
DTModel::DTModel(size_t inputSize, size_t outputSize, ActivationE outputActivation)
{

  assert(inputSize != 0);
  assert(outputSize != 0);

  this->inputSize = inputSize;
  this->outputSize = outputSize;

  neurons.reserve(inputSize + outputSize + NEURON_BUFFER_INTERVAL);

  size_t neuronIndex = 0;

  // Add input neurons
  for (size_t _ = 0; _ < inputSize; _++)
  {
    Neuron neuron(neuronIndex, NeuronTypeE::INPUT_NEURON, ActivationE::NO_ACTIVATION);
    indexMap[neuronIndex] = neuronIndex;
    neuronIndex++;
    neurons.push_back(neuron);
  }

  // Add output neurons
  for (size_t _ = 0; _ < outputSize; _++)
  {
    Neuron neuron(neuronIndex, NeuronTypeE::OUTPUT_NEURON, outputActivation);
    indexMap[neuronIndex] = neuronIndex;
    neuronIndex++;
    neurons.push_back(neuron);
  }

  isSorted = true;

}

/******************************************************************************
* OPERATORS
******************************************************************************/
std::ostream &operator<<(std::ostream &os, const DTModel &dtmodel)
{
  os << "DT NEURAL NETWORK" << std::endl;
  os << "NUMBER OF INPUTS: " << dtmodel.inputSize << std::endl;
  os << "NUMBER OF OUTPUTS: " << dtmodel.outputSize << std::endl;
  os << "NUMBER OF NEURONS: " << dtmodel.neurons.size() << std::endl;
  for (auto neuron : dtmodel.neurons)
  {
    os << "=====================================" << std::endl;
    os << neuron << std::endl;
    os << "=====================================" << std::endl;
  }

  return os;
}

/******************************************************************************
* UTILITIES
******************************************************************************/

/******************************************************************************
 * @brief Sorts the neural network topologically
 *
 * Recalculates depths and depthIds from the current connections, updates
 * maxDepth and indexMap, and marks the model as sorted.
 *
 ******************************************************************************/
void DTModel::sortTopologically()
{
  // Topology changes can shorten paths as well as extend them.
  maxDepth = 0;
  for (Neuron &neuron : neurons)
  {
    neuron.depth = 0;
  }

  vector<size_t> sortedNeuronsIds{};
  sortedNeuronsIds.reserve(neurons.size());

  vector<size_t> queue{};
  queue.reserve(neurons.size() - (inputSize + outputSize));

  // To not have to delete members already on queue
  // we just move the pointer to the right.
  size_t queuePointer = 0;

  vector<size_t> neuronsConnectionNum(neurons.size(), 0);

  // Calculate order of each neuron(vertex)
  for (const Neuron &neuron : neurons)
  {
    for (const Synapse &synapse : neuron.outSynapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      neuronsConnectionNum[outNeuronIndex]++;
    }
  }

  map<size_t, size_t> newIndexMap;

  // Start from input neurons since no connections go into them
  // they cant be reach normally and they are always at the beggining
  // of the order.
  for (size_t index = 0; index < inputSize; index++)
  {
    sortedNeuronsIds.push_back(index);

    size_t inputNeuronIndex = indexMap[index];

    for (const Synapse &synapse : neurons[inputNeuronIndex].outSynapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      neuronsConnectionNum[outNeuronIndex]--;

      // If vertex(neuron) is of order 0 push it onto queue
      // Don't push output neurons since they wont feed into any other neuron anyway
      if (neuronsConnectionNum[outNeuronIndex] == 0 && neurons[outNeuronIndex].type != NeuronTypeE::OUTPUT_NEURON)
      {
        queue.push_back(outNeuronId);
      }

      if (neurons[outNeuronIndex].depth <= neurons[inputNeuronIndex].depth)
      {
        neurons[outNeuronIndex].depth = neurons[inputNeuronIndex].depth + 1;
      }

      if (neurons[outNeuronIndex].depth > maxDepth)
      {
        maxDepth = neurons[outNeuronIndex].depth;
      }
    }

  }

  // Dequeu neurons id and repeat the same process until all hidden neurons
  // are processed and sorted.
  while(queuePointer != queue.size())
  {

    size_t neuronId = queue[queuePointer];

    sortedNeuronsIds.push_back(neuronId);

    size_t neuronIndex = indexMap[neuronId];

    for (const Synapse &synapse : neurons[neuronIndex].outSynapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      neuronsConnectionNum[outNeuronIndex]--;

      // If vertex(neuron) is of order 0 push it onto queue
      // Don't push output neurons since they wont feed into any other neuron anyway
      if (neuronsConnectionNum[outNeuronIndex] == 0 && neurons[outNeuronIndex].type != NeuronTypeE::OUTPUT_NEURON)
      {
        queue.push_back(outNeuronId);
      }

      if (neurons[outNeuronIndex].depth <= neurons[neuronIndex].depth)
      {
        neurons[outNeuronIndex].depth = neurons[neuronIndex].depth + 1;
      }
    }

    queuePointer++;

  }

  // Add output neurons to the sorted neurons list
  for(size_t index = 0; index < outputSize; index++)
  {
    size_t outputNeuronId = inputSize + index;
    sortedNeuronsIds.push_back(outputNeuronId);
  }

  for (size_t index = 0; index < sortedNeuronsIds.size(); index++)
  {
    newIndexMap[sortedNeuronsIds[index]] = index;
  }

  vector<Neuron> sortedNeurons{};

  sortedNeurons.reserve(neurons.size());

  for (auto neuronIndex: sortedNeuronsIds)
  {
    size_t sortedNeuronIndex = indexMap[neuronIndex];
    sortedNeurons.push_back(std::move(neurons[sortedNeuronIndex]));
  }

  indexMap = std::move(newIndexMap);
  neurons = std::move(sortedNeurons);

  size_t currentDepth = 0;
  size_t previousDepth = 0;
  size_t counter = 0;

  constexpr size_t DEPTH_ID_CONSTANT = 1000;

  for (Neuron &neuron : neurons)
  {
    currentDepth = neuron.depth;
    if (currentDepth > maxDepth)
    {
      maxDepth = currentDepth;
    }

    if (currentDepth != previousDepth)
    {
      counter = 0;
    }

    neuron.depthId = currentDepth*DEPTH_ID_CONSTANT + counter;

    previousDepth = currentDepth;

    counter++;
  }

  isSorted = true;
}

/******************************************************************************
 * @brief Checks if all hidden neurons are reachable from input. By rule
 *        dangling pointers are not allowed.
 *
 * @return if model is valid or not
 ******************************************************************************/
bool DTModel::validateModel()
{

  vector<bool> isNeuronReachable(neurons.size(), false);
  vector<size_t> numberOfSynapses(neurons.size(), 0);

  // Check each synapse to see which neurons are connected
  for (size_t index = 0; index <  neurons.size(); index++)
  {
    numberOfSynapses[index] = neurons[index].outSynapses.size();

    for (auto synapse: neurons[index].outSynapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      isNeuronReachable[outNeuronIndex] = true;
    }
  }

  for (size_t index = 0; index < neurons.size(); index++)
  {
    if (neurons[index].type == NeuronTypeE::HIDDEN_NEURON && (!isNeuronReachable[index] ||
        numberOfSynapses[index] == 0))
    {
      return false;
    }
  }

  return true;
}

/******************************************************************************
 * @brief Adds a new neuron to the network. New neuron must have an input
 *        from one of the existing neurons, and must connect to another
 *        existing neuron.
 *
 * @param neuron          Neuron to be added
 * @param sortAfterAdding If network should be sorted after adding neuron.
 *                        Useful for adding multiple neurons.
 * @param inSynapse       Synapse going into new neuron
 * @param outSynapse      Synapse going out of new neuron
 *
 ******************************************************************************/
void DTModel::addNeuron(Neuron neuron, Synapse inSynapse, Synapse outSynapse, bool sortAfterAdding)
{

  for (Neuron n : neurons)
  {
    assert(n.id != neuron.id);
  }

  // Ensure that new synapse is not going from output neurons
  assert((inSynapse.inNeuronId < inputSize || inSynapse.inNeuronId >= (inputSize + outputSize)));

  // Ensure that the synapse is going into added neuron
  assert((inSynapse.outNeuronId == neuron.id));

  // Ensure that new outgoing synapse is not feeding into input layer
  assert(outSynapse.outNeuronId >= inputSize);

  // Ensure that the synapse is going from added neuron
  assert((outSynapse.inNeuronId == neuron.id));

  // Add synapsse going in and out from the new neuron
  neuron.addOutSynapse(outSynapse);
  neuron.addInSynapse(inSynapse);

  size_t inNeuronId = inSynapse.inNeuronId;

  size_t outNeuronId = outSynapse.outNeuronId;

  assert(!(indexMap.find(inNeuronId) == indexMap.end()));

  assert(!(indexMap.find(outNeuronId) == indexMap.end()));

  size_t inNeuronIndex = indexMap[inNeuronId];

  size_t outNeuronIndex = indexMap[outNeuronId];

  // Add synapse feeding into new neuron
  neurons[inNeuronIndex].addOutSynapse(inSynapse);

  // Add synapse feeding from new neuron into
  // neuron that it feeds into
  neurons[outNeuronIndex].addInSynapse(outSynapse);

  if (neurons.size() == neurons.capacity())
  {
    neurons.reserve(neurons.size() + NEURON_BUFFER_INTERVAL);
  }

  neurons.push_back(neuron);

  indexMap[neuron.id] = neurons.size() - 1;

  if (sortAfterAdding)
  {
    sortTopologically();
    isSorted = true;
  }
  else
  {
    isSorted = false;
  }

}

/******************************************************************************
 * @brief Checks whether a directed connection already exists
 *
 * Uses the source neuron's outgoing synapses, the model's canonical connection
 * storage. Disabled synapses also count as existing connections. Neither a
 * missing neuron ID nor this query modifies indexMap or the model.
 *
 * @param inNeuronId  ID of the source neuron
 * @param outNeuronId ID of the destination neuron
 *
 * @return True if a synapse connects these endpoints in this direction
 ******************************************************************************/
bool DTModel::hasSynapse(size_t inNeuronId, size_t outNeuronId) const
{
  const auto source = indexMap.find(inNeuronId);
  if (source == indexMap.end())
  {
    return false;
  }

  for (const Synapse &synapse : neurons[source->second].outSynapses)
  {
    if (synapse.inNeuronId == inNeuronId && synapse.outNeuronId == outNeuronId)
    {
      return true;
    }
  }
  return false;
}

/******************************************************************************
 * @brief Adds a new connection to the network
 *
 * @param newSynapse      New connection
 * @param sortAfterAdding If network should be sorted after adding synapse.
 *                        Useful for adding multiple synapses.
 *
 ******************************************************************************/
void DTModel::addOutSynapse(Synapse newSynapse, bool sortAfterAdding)
{

  size_t inNeuronId = newSynapse.inNeuronId;
  size_t outNeuronId = newSynapse.outNeuronId;

  bool inNeuronFound = false;
  bool outNeuronFound = false;

  assert(inNeuronId != outNeuronId);

  for (Neuron n : neurons)
  {

    if (n.id == inNeuronId)
    {
      inNeuronFound = true;
    }

    if (n.id == outNeuronId)
    {
      outNeuronFound = true;
    }
  }

  Assert(inNeuronFound, "NO NEURON MATCHING inNeuronID " + std::to_string(inNeuronId));
  assert(outNeuronFound);

  size_t inNeuronIndex = indexMap[inNeuronId];

  neurons[inNeuronIndex].addOutSynapse(newSynapse);

  if (sortAfterAdding)
  {
    sortTopologically();
    isSorted = true;
  }
  else
  {
    isSorted = false;
  }

}

/******************************************************************************
 * @brief Sets neuron bias
 *
 * @param neuronId  Which neuron bias to set
 * @param value     New value of bias
 *
 ******************************************************************************/
void DTModel::setBias(size_t neuronId, double value)
{

  bool isNeuronInNetwork = !(indexMap.find(neuronId) == indexMap.end());

  assert(isNeuronInNetwork);

  size_t neuronIndex = indexMap[neuronId];

  neurons[neuronIndex].bias = value;

}

/******************************************************************************
 * @brief Removes a neuron from neural network
 *
 * @param id              ID of neuron to remove
 * @param sortAfterRemove If netowrk should be sorted after removal
 *
 ******************************************************************************/
void DTModel::removeNeuron(size_t id, bool sortAfterRemove)
{
  // input and output neurons cannot be removed;
  assert(id >= inputSize + outputSize);

  bool isNeuronInNetwork = !(indexMap.find(id) == indexMap.end());

  assert(isNeuronInNetwork);

  size_t neuronIndex = indexMap[id];

  size_t switchedNeuronId = neurons[neurons.size() - 1].id;

  // Overwrite neuron we want to remove with last one in the list
  // This way we can pop back neurons list. We need to sort topologically
  // this network anyway so it doesnt matter if we ruin the order
  neurons[neuronIndex] = neurons[neurons.size() - 1];

  // Update index map
  indexMap[switchedNeuronId] = neuronIndex;

  // Delete removed neuron id from map
  indexMap.erase(id);

  (void)neurons.pop_back();


  // Remove all synapses feeding into that neuron
  for (auto neuron : neurons)
  {
    vector<std::pair<size_t, size_t>> neuronOutSynapseIdPairs{};
    vector<std::pair<size_t, size_t>> neuronInSynapseIdPairs{};
    // At most we remove all synapses from a neuron
    neuronOutSynapseIdPairs.reserve(neuron.outSynapses.size());
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.outNeuronId == id)
      {
        std::pair<size_t, size_t> pair(neuron.id, synapse.id);
        neuronOutSynapseIdPairs.push_back(pair);
      }
    }

    for (auto synapse : neuron.inSynapses)
    {
      if (synapse.inNeuronId == id)
      {
        std::pair<size_t, size_t> pair(neuron.id, synapse.id);
        neuronInSynapseIdPairs.push_back(pair);
      }
    }

    for (size_t index = 0; index < neuronOutSynapseIdPairs.size(); index++)
    {
      size_t neuronId = neuronOutSynapseIdPairs[index].first;
      size_t synapseId = neuronOutSynapseIdPairs[index].second;

      size_t neuronIndex = indexMap[neuronId];

      neurons[neuronIndex].removeSynapse(synapseId);
    }

    for (size_t index = 0; index < neuronInSynapseIdPairs.size(); index++)
    {
      size_t neuronId = neuronInSynapseIdPairs[index].first;
      size_t synapseId = neuronInSynapseIdPairs[index].second;

      size_t neuronIndex = indexMap[neuronId];

      neurons[neuronIndex].removeSynapse(synapseId);
    }
  }

  isSorted = false;
  if (sortAfterRemove)
  {
    sortTopologically();
    isSorted = true;
  }

}

/******************************************************************************
 * @brief Removes a synapse from neural network
 *
 * @param inNeuronId  ID of neuron connection is going from
 * @param outNeuronId ID of neuron connection is feeding into
 *
 ******************************************************************************/
void DTModel::removeSynapse(size_t inNeuronId, size_t outNeuronId, bool sortAfterRemoveal)
{
  // No synapse can feed into input neurons
  assert(outNeuronId >= inputSize);

  bool isInNeuronInNetwork = !(indexMap.find(inNeuronId) == indexMap.end());

  assert(isInNeuronInNetwork);

  bool isOutNeuronInNetwork = !(indexMap.find(outNeuronId) == indexMap.end());

  assert(isOutNeuronInNetwork);

  size_t inNeuronIndex = indexMap[inNeuronId];
  size_t outNeuronIndex = indexMap[outNeuronId];

  for (size_t index = 0; index < neurons[inNeuronIndex].outSynapses.size(); index++)
  {
    size_t synapseInNeuronId = neurons[inNeuronIndex].outSynapses[index].inNeuronId;
    size_t synapseOutNeuronId = neurons[inNeuronIndex].outSynapses[index].outNeuronId;

    if (synapseInNeuronId == inNeuronId && synapseOutNeuronId == outNeuronId)
    {
      size_t synapseId = neurons[inNeuronIndex].outSynapses[index].id;
      neurons[inNeuronIndex].removeSynapse(synapseId);
      break;
    }

  }

  for (size_t index = 0; index < neurons[outNeuronIndex].inSynapses.size(); index++)
  {
    size_t synapseInNeuronId = neurons[outNeuronIndex].inSynapses[index].inNeuronId;
    size_t synapseOutNeuronId = neurons[outNeuronIndex].inSynapses[index].outNeuronId;

    if (synapseInNeuronId == inNeuronId && synapseOutNeuronId == outNeuronId)
    {
      size_t synapseId = neurons[outNeuronIndex].inSynapses[index].id;
      neurons[outNeuronIndex].removeSynapse(synapseId);
      break;
    }

  }

  isSorted = false;
  if (sortAfterRemoveal)
  {
    sortTopologically();
    isSorted = true;
  }

}

/******************************************************************************
 * @brief Runs input through neural netowrk
 *
 * @param input Neural network input
 *
 ******************************************************************************/
vector<double> DTModel::feedForward(vector<double> input)
{

  if (!isSorted)
  {
    sortTopologically();
    isSorted = true;
  }

  assert(input.size() == inputSize);

  vector<double> output(outputSize, 0);

  size_t outputPointer = 0;

  // Clear accumulators
  for (size_t index = 0; index < neurons.size(); index++)
  {
    neurons[index].value = 0;
  }

  for (size_t index = 0; index < inputSize; index++)
  {
    neurons[index].value = input[index];
  }

  for (size_t index = 0; index < neurons.size(); index++)
  {
    if (neurons[index].type != NeuronTypeE::INPUT_NEURON)
    {
      neurons[index].value -= neurons[index].bias;
      neurons[index].activate();
    }

    double neuronValue = neurons[index].value;

    if (neurons[index].type == NeuronTypeE::OUTPUT_NEURON)
    {
      output[outputPointer] = neuronValue;
      outputPointer++;
      continue;
    }

    for (auto synapse: neurons[index].outSynapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];

      neurons[outNeuronIndex].value += neuronValue * synapse.weight;
    }

  }

  return output;

}
