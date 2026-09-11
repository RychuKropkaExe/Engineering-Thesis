#pragma once

#include "synapse.h"
#include "utils.h"
#include <vector>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @class Neuron
 *
 * @brief Represents a single neuron in a dynamic topology network
 *
 * @public @param  id                      Neuron unique id
 * @public @param  depthId                 Neuron ID in relation to its depth. Changes dynamically,
 *                                         is not unique and can be used to compare two neurons in
 *                                         different networks.
 * @public @param  depth                   Neuron depth in network
 * @public @param  type                    Neuron type
 * @public @param  value                   Neuron value after applying weights and calculations
 * @public @param  inSynapses              Synapses going into neuron
 * @public @param  outSynapses             Synapses going out from neuron
 * @private @param SYNAPSE_BUFFER_INTERVAL Initial size of synapse buffers
 *                                         and how much larger they become after
 *                                         each reserve-ing.
 ******************************************************************************/
class Neuron
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  size_t id;
  size_t depthId{0};
  size_t depth;
  NeuronTypeE type;
  ActivationE activation;

  double value{0.0};
  double bias;

  vector<Synapse> inSynapses;
  vector<Synapse> outSynapses;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  Neuron() = default;
  Neuron(size_t id, NeuronTypeE type, ActivationE activation, vector<Synapse> outSynapses = vector<Synapse>{});

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Neuron &neuron);

  /******************************************************************************
  * UTILITIES
  ******************************************************************************/
  void addOutSynapse(Synapse newSynapse);
  void addInSynapse(Synapse newSynapse);
  void removeSynapse(size_t id);

  void activate();

private:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  static constexpr size_t SYNAPSE_BUFFER_INTERVAL = 5;
};
