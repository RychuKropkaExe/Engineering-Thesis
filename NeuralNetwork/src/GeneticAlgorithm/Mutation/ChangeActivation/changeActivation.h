#pragma once

#include "individual.h"
#include "model.h"
#include "mutationAlgorithm.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class ChangeActivation
 *
 * @brief Implements activation change mutations. Changes activation function
 *        For given neuron into a random one.
 *
 * @public @param numberOfFlips How many biases should be flipped
 *
 ******************************************************************************/
class ChangeActivation : public MutationAlgorithm
{
public:
  size_t numberOfChanges{0};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  ChangeActivation(size_t numberOfChanges);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void mutateIndividual(Individual &individual);
};
