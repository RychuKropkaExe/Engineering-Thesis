#include "neuron.h"
#include <cassert>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

Neuron::Neuron(size_t id, NeuronTypeE type, vector<Synapse> synapses, ActivationE activation)
{
  this->id = id;
  this->type = type;

  bias = DTMUtils::randomdouble()();

  switch (type)
  {
  case NeuronTypeE::INPUT_TYPE:
  {
    break;
  }
  case NeuronType::HIDDEN_TYPE:
  {
    assert(synapses.size() > 0);
    break;
  }
  case NeuronType::OUTPUT_TYPE:
  {
    assert(synapses.size() == 0);
    break;
  }
  }

  this->synapses = synapses;

  this->activation = activation;
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const Neuron &neuron)
{
  os << "NEURON ID: " << neuron.id << std::endl;
  os << "NEURON TYPE: " << DTMUtils::neuronTypeToString(neuron.type) << std::endl;
  os << "ACTIVATION FUNCTION: " << DTMUtils::activationFunctionToString(neuron.activation);
  os << "OUTGOING CONNECTIONS:" << std::endl;
  for (auto synapse : neuron.synapses)
  {
    os << synapse;
  }
  return os;
}


/******************************************************************************
* UTILITIES
******************************************************************************/

/******************************************************************************
 * @brief Add a new synapse going out from neuron
 *
 * @param newSynapse Synapse to be added
 *
 ******************************************************************************/
void Neuron::addSynapse(Synapse newSynapse)
{

  for (Synapse synapse : synapses)
  {
    assert(synapse.id != newSynapse.id);
  }

  if (synapses.size() == synapse.capacity())
  {
    synapse.resize(synapse.size() + SYNAPSE_BUFFER_INTERVAL);
  }

  synapse.push_back(newSynapse);

}

/******************************************************************************
 * @brief Removes synapse from neuron
 *
 * @param id ID of synapse to be removed
 *
 ******************************************************************************/
void Neuron::removeSynapse(size_t id)
{

  bool doesSynapseExist = false;

  for (size_t index = 0; index < synapses.size(); index++)
  {
    if (synapses[index].id == id)
    {
      synapses[index] = synapses[synapses.size()];
      (void)synapses.pop_back();
      doesSynapseExist = true;
    }
  }

  assert(doesSynapseExist);

}

/******************************************************************************
 * @brief Sigmoid activation functions
 *
 * @param x Value from output FastMatrix
 *
 * @return Value mapped from (-1, 1)
 ******************************************************************************/
static inline double sigmoidf(double x)
{
    return (double)1.0 / ((double)1.0 + std::exp(-x));
}

/******************************************************************************
 * @brief Activates neuron with set activation function
 *
 ******************************************************************************/
void Neuron::activate()
{
  switch (activation)
  {
  case ActivationE::SIGMOID:
  {
      value = sigmoidf(value);
      break;
  }
  case ActivationE::RELU:
  {
      value = std::max(0.0, value);
      break;
  }
  case ActivationE::NO_ACTIVATION:
  {
      break;
  }
  }
}
