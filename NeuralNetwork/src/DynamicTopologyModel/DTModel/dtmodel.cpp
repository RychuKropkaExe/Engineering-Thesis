#include "dtmodel.h"

/******************************************************************************
* CONSTRUCTORS
******************************************************************************/
DTModel::DTModel(size_t id, size_t inputSize, size_t outputSize, ActivationE outputActivation)
{

  assert(inputSize != 0);
  assert(outputSize != 0);

  this->id = id;
  this->inputSize = inputSize;
  this->outputSize = outputSize;

  neurons.reserve(inputSize + outputSize + NEURON_BUFFER_INTERVAL);

  size_t neuronIndex = 0;

  for (size_t _ = 0; _ < inputSize; _++)
  {
    Neuron neuron(neuronIndex, NeuronTypeE::INPUT_NEURON, vector<Synapses>(), ActivationE::noActivation);
    indexMap[neuronIndex] = neuronIndex;
    neuronIndex++;
  }

  for (size_t _ = 0; _ < outputSize; _++)
  {
    Neuron neuron(neuronIndex, NeuronTypeE::OUTPUT_NEURON, vector<Synapses>(), outputActivation);
    indexMap[neuronIndex] = neuronIndex;
    neuronIndex++;
  }

  isSorted = true;

}

/******************************************************************************
* OPERATORS
******************************************************************************/
std::ostream &operator<<(std::ostream &os, const DTModel &dtmodel)
{
  os << "DT NEURAL NETWORK WITH ID: " << dtmodel.id << std::endln;
  os << "NUMBER OF INPUTS: " << dtmodel.inputSize << std::endln;
  os << "NUMBER OF OUTPUTS: " << dtmodel.outputSize << std::endln;
  os << "NUMBER OF NEURONS: " << dtmodel.neurons.size() << std::endln;
  for (auto neuron : neurons)
  {
    os << neuron;
  }
}

/******************************************************************************
* UTILITIES
******************************************************************************/

/******************************************************************************
 * @brief Sorts the neural network topologically
 *
 ******************************************************************************/
void DTModel::sortTopoligically()
{
  vector<size_t> sortedNeuronsIds{};
  sortedNeuronsIds.reserve(neurons.size());

  vector<size_t> queue{};
  queue.reserve(neurons.size() - (inputSize + outPutSize));

  // To not have to delete members already on queue
  // we just move the pointer to the right.
  size_t queuePointer = 0;

  vector<size_t> neuronsConnectionNum(neurons.size(), 0);

  // Calculate order of each neuron(vertex)
  for (auto neuron: neurons)
  {
    for (auto synapse: neuron.synapses)
    {
      size_t outNeuronId = synapses.outNeuronId;
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

    for (auto synapse : neurons[index].synapses)
    {
      size_t outNeuronId = synapses.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      neuronsConnectionNum[outNeuronIndex]--;

      // If vertex(neuron) is of order 0 push it onto queue
      // Don't push output neurons since they wont feed into any other neuron anyway
      if (neuronsConnectionNum[outNeuronIndex] == 0 && neurons[outNeuronIndex].type != NeuronTypeE::OUTPUT_NEURON)
      {
        queue.push_back(outNeuronId);
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

    for (auto synapse : neurons[neuronIndex].synapses)
    {
      size_t outNeuronId = synapses.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];
      neuronsConnectionNum[outNeuronIndex]--;

      if (neuronsConnectionNum[outNeuronIndex] == 0)
      {
        queue.push_back(outNeuronId);
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
    sortedNeurons.push_back(neurons[neuronIndex]);
  }

  indexMap = newIndexMap;
  neurons = sortedNeurons;

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
  assert((inSynapse.inNeuronId < inputSize || inSynapse.inNeuronId > (inputSize + outputSize)))

  // Ensure that the synapse is going into added neuron
  assert((inSynapse.outNeuronId == neuron.id))

  // Ensure that new outgoing synapse is not feeding into input layer
  assert(outSynapse.outNeuronId > inputSize)

  // Ensure that the synapse is going from added neuron
  assert((outSynapse.inNeuronId == neuron.id))

  for (size_t _ = 0; _ < inputSize; _++)
  {
    Neuron neuron(neuronIndex, NeuronTypeE::INPUT_NEURON, vector<Synapses>{inSynapse, outSynapse}, ActivationE::noActivation);
    indexMap[neuronIndex] = neuronIndex;
    neuronIndex++;
  }

  if (neurons.size() == neurons.capacity())
  {
    neurons.reserve(neurons.size() + NEURON_BUFFER_INTERVAL);
  }

  neurons.push_back(neuron);

  indexMap[neuron.id] = neurons.size();

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
 * @brief Adds a new connection to the network
 *
 * @param newSynapse      New connection
 * @param sortAfterAdding If network should be sorted after adding synapse.
 *                        Useful for adding multiple synapses.
 *
 ******************************************************************************/
void DTModel::addSynapse(Synapse newSynapse, bool sortAfterAdding)
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

  assert(inNeuronFound);
  assert(outNeuronFound);

  size_t inNeuronIndex = indexMap[inNeuronId];

  neurons[inNeuronIndex].addSynapse(inSynapse);

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
 * @brief Removes a neuron from neural network
 *
 * @param id  ID of neuron to remove
 *
 ******************************************************************************/
void DTModel::removeNeuron(size_t id)
{
  // input and output neurons cannot be removed;
  assert(id >= inputSize + outputSize);

  bool isNeuronInNetwork = (indexMap.find(id) == indexMap.end())

  assert(isNeuronInNetwork);

  size_t neuronIndex = indexMap[id];

  // Overwrite neuron we want to remove with last one in the list
  // This way we can pop back neurons list. We need to sort topologically
  // this network anyway so it doesnt matter if we ruin the order
  neurons[neuronIndex] = neurons[neurons.size()];

  // Update index map
  indexMap[neurons[neuronIndex].id] = neuronIndex;

  // Delete removed neuron id from map
  indexMap.erase(id);

  (void)neurons.pop_back();


  // Remove all synapses feeding into that neuron
  for (auto neuron : neurons)
  {
    for (auto synapse : neuron.syanpses)
    {
      if (synapse.outNeuronId == id)
      {
        neuron.removeSynapse(synapse.id);
      }
    }
  }

  sortTopologically();
  isSorted = true;

}

/******************************************************************************
 * @brief Removes a synapse from neural network
 *
 * @param id  ID of synapse to remove
 *
 ******************************************************************************/
void DTModel::removeSynapse(size_t inNeuronId, size_t outNeuronId)
{
  // No synapse can feed into input neurons
  assert(outNeuronId >= inputSize);

  bool isInNeuronInNetwork = (indexMap.find(inNeuronId) == indexMap.end())

  assert(isInNeuronInNetwork);

  bool isOutNeuronInNetwork = (indexMap.find(outNeuronId) == indexMap.end())

  assert(isOutNeuronInNetwork);

  size_t neuronIndex = indexMap[inNeuronId];

  for (size_t index = 0; index < neurons[neuronIndex].synapses.size(); index++)
  {
    size_t synapseInNeuronId = neurons[neuronIndex].synapses[index].inNeuronId;
    size_t synapseOutNeuronId = neurons[neuronIndex].synapses[index].outNeuronId;

    if (synapseInNeuronId == inNeuronId && synapseOutNeuronId == outNeuronId)
    {
      size_t synapseId = neurons[neuronIndex].synapses[index].id;
      neurons[neuronIndex].removeSynapse(synapseId);
      break;
    }

  }

  sortTopologically();
  isSorted = true;

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

    for (auto synapse: neurons[index].synapses)
    {
      size_t outNeuronId = synapse.outNeuronId;
      size_t outNeuronIndex = indexMap[outNeuronId];

      neurons[outNeuronIndex].value += neuronValue * synapse.weight;
    }

  }

  return output;

}
