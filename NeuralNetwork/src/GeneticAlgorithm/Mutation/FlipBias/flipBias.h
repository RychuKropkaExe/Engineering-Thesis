#pragma once

#include "individual.h"
#include "model.h"
#include "mutationAlgorithm.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class FlipWeight
 *
 * @brief Implements biases flipping mutations. If a bias has a value it will
 *        be deactivated(set to 0). If it is deactivated it will be assigned a random
 *        value.
 *
 * @public @param numberOfFlips How many biases should be flipped
 *
 ******************************************************************************/
class FlipBias : public MutationAlgorithm
{
public:
  size_t numberOfFlips{0};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  FlipBias(size_t numberOfFlips);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void mutateIndividual(Individual &individual);
};
