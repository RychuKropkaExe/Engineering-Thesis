#pragma once

#include "individual.h"
#include "model.h"
#include "parents.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class MutationAlgorithm
 *
 * @brief Base class for mutating individual
 *
 ******************************************************************************/
class MutationAlgorithm
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/

  virtual void mutateIndividual(Individual &individual);
};
