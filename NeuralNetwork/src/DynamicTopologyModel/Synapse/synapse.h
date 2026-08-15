#pragma once

#include <cstddef>
#include <iostream>
#include "utils.h"

/******************************************************************************
 * @class Synapse
 *
 * @brief Represents a single synapse connecting two neurons in a dynamic topology network
 *
 * @public @param id          Synapse id
 * @public @param inNeuronId  Input neuron id
 * @public @param outNeuronId Output neuron id
 * @public @param weight      Synapse weight value
 * @public @param isActive    Indicates if synapse is enabled
 ******************************************************************************/
class Synapse
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  size_t id;
  size_t inNeuronId;
  size_t outNeuronId;
  double weight;
  bool isActive;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  Synapse() = default;
  Synapse(size_t id, size_t inNeuronId, size_t outNeuronId, double weight);

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Synapse &synapse);
};
