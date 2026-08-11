#pragma once

#include "individual.h"
#include "model.h"
#include "parents.h"

/******************************************************************************
 * @class FitnessEvaluation
 *
 * @brief Base class for all fitness evaluation algorithms
 *
 *
 ******************************************************************************/
class FitnessEvaluation
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  virtual void evaluateIndividual(Individual &individual) = 0;
};
