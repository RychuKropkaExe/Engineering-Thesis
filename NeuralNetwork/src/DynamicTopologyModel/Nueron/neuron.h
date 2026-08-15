#pragma once

#include "synapse.h"
#include "utils.h"

using std::vector;

/******************************************************************************
 * @class Neuron
 *
 * @brief Represents a single neuron in a dynamic topology network
 *
 * @public @param  id                      Neuron id
 * @public @param  type                    Neuron type
 * @public @param  value                   Neuron value after applying weights and calculations
 * @public @param  synapses                Synapses going out from neuron
 * @private @param SYNAPSE_BUFFER_INTERVAL Initial size of synapse buffers
 *                                         and how much larger they become after
 *                                         each resize-ing.
 ******************************************************************************/
class Neuron
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  size_t id;
  NeuronTypeE type;
  AtivationFunctionE activation;

  double value;
  double bias;

  vector<Synapse> synapses;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  Neuron(size_t id, NeuronTypeE type, vector<Synapse> synapses);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Neuron &neuron);

  /******************************************************************************
  * UTILITIES
  ******************************************************************************/
  void addSynapse(Synapse newSynapse);
  void removeSynapse(size_t id);

  void activate();

private:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  constexpr size_t SYNAPSE_BUFFER_INTERVAL = 5;
};
