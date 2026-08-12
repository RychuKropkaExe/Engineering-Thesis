#pragma once

#include "individual.h"
#include "model.h"
#include "mutationAlgorithm.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class AdjustWeight
 *
 * @brief Implements weights adjustment mutations. Muliplies weight by an adjustment
 *        rate
 *
 * @public @param numberOfAdjustments How many weights should be adjusted
 * @public @param adjustmentRate      Value weight is multiplied by
 *
 ******************************************************************************/
class AdjustWeight : public MutationAlgorithm
{
public:
  size_t numberOfAdjustments{0};
  double adjustmentRate{1};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  AdjustWeight(size_t numberOfAdjustments, double adjustmentRate);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void mutateIndividual(Individual &individual);
};
