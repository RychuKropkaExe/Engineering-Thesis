#pragma once

#include "individual.h"
#include "logger.h"
#include "model.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class GAParents
 *
 * @brief Represents parent pairs for crossover purposes of genetic algorithm
 *
 * @public @param firstParentIndex      Index of first parent
 * @public @param secondParentIndex     Index of second parent
 * @public @param isFirstParentPresent  Indicates if first parent was added
 * @public @param isSecondParentPresent Indicates if second parent was added
 ******************************************************************************/
class GAParents
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  size_t firstParentIndex;
  size_t secondParentIndex;
  bool isFirstParentPresent{false};
  bool isSecondParentPresent{false};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  GAParents() = default;

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void addParent(size_t parentIndex);
};
