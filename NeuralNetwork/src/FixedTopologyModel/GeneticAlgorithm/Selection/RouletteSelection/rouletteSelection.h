#include "selectionAlgorithm.h"

#pragma once

#include "individual.h"
#include "model.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class RouletteSelection
 *
 * @brief Implementation of roulette parent selection
 * @public @param scalingFactor scales very high fitness by setting a max deviation from mean
 *                              for all individuals, meaning that in roulette less fit individuals
 *                              have higher chance of being chosen than normally.
 * @public @param isScalingUsed Indicates if scaling should be used
 *
 ******************************************************************************/
class RouletteSelection : public SelectionAlgorithm
{
public:
  double scalingFactor{};
  bool isScalingUsed{false};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  RouletteSelection(double scalingFactor);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  vector<Parents> selectParents(vector<Individual> &population, size_t numberOfParentsPairs);
};
