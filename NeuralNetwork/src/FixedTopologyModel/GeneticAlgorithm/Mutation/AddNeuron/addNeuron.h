#pragma once

#include "individual.h"
#include "model.h"
#include "mutationAlgorithm.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class AddNeuron
 *
 * @brief Implements mutation by adding a single neuron to layer.
 *
 ******************************************************************************/
class AddNeuron : public MutationAlgorithm
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void mutateIndividual(Individual &individual);
};
