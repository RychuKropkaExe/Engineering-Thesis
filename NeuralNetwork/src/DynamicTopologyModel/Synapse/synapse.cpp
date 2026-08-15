#include "synapse.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

Synapse(size_t id, size_t inNeuronId, size_t outNeuronId, double weight)
{
  this->id = id;
  this->inNeuronId = id;
  this->outNeuronId = id;
  this->weight = weight;
  this->isActive = true;
}

/******************************************************************************
* OPERATORS
******************************************************************************/

std::ostream &operator<<(std::ostream &os, const Synapse &synapse)
{
  os << "SYNAPSE ID: " << synapse.id << " isActive: " << synapse.isActive << std::endln;
  os << "INPUT NEURON ID: " < synapse.inNeuronId << std::endln;
  os << "OUTPUT NEURON ID: " < synapse.outNeuronId << std::endln;
  os << "WEIGHT VALUE: " << synapse.weight << std::endln;
  return os;
}
