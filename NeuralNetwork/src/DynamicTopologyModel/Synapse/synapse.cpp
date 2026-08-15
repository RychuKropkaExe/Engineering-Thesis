#include "synapse.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

Synapse::Synapse(size_t id, size_t inNeuronId, size_t outNeuronId, double weight)
{
  this->id = id;
  this->inNeuronId = inNeuronId;
  this->outNeuronId = outNeuronId;
  this->weight = weight;
  this->isActive = true;
}

/******************************************************************************
* OPERATORS
******************************************************************************/

std::ostream &operator<<(std::ostream &os, const Synapse &synapse)
{
  os << "SYNAPSE ID: " << synapse.id << " isActive: " << synapse.isActive << std::endl;
  os << "INPUT NEURON ID: " << synapse.inNeuronId << std::endl;
  os << "OUTPUT NEURON ID: " << synapse.outNeuronId << std::endl;
  os << "WEIGHT VALUE: " << synapse.weight << std::endl;
  return os;
}
