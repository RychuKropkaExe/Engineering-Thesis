#include "selectionAlgorithm.h"

#pragma once

#include "individual.h"
#include "model.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class TournamentSelection
 *
 * @brief Implementation of tournament selection
 *
 ******************************************************************************/
class TournamentSelection : public SelectionAlgorithm
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  vector<Parents> selectParents(vector<Individual> &population, size_t numberOfParentsPairs);
};
