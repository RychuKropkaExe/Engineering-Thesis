#pragma once

#include "individual.h"
#include "model.h"
#include "parents.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class SelectionAlgorithm
 *
 * @brief Base class for selecting parents for offspring
 *
 ******************************************************************************/
class SelectionAlgorithm
{
public:
  /******************************************************************************
   * UTILITIES
   ******************************************************************************/

  /******************************************************************************
   * @brief Adds protected individuals to parents list regardless of their fitness
   *
   * @param population Current population
   * @param parents    Reference to parents list
   * @param startIndex Start index for where pair should be inserted into parents list
   *
   * @return Current index of last non filled entry in parents list
   ******************************************************************************/
  size_t fasttrackProtectedIndividuals(vector<Individual> &population, vector<Parents> &parents, size_t startIndex);

  virtual vector<Parents> selectParents(vector<Individual> &population, size_t numberOfParentsPairs) = 0;
};
