#pragma once

#include "fitnessEvaluation.h"
#include "individual.h"

/******************************************************************************
 * @class FitnessEvaluation
 *
 * @brief Base class for all fitness evaluation algorithms
 *
 *
 ******************************************************************************/
class MmseEval : public FitnessEvaluation
{
public:
  TrainingData expectedResults;

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  MmseEval(TrainingData td);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void evaluateIndividual(Individual &individual);
};
