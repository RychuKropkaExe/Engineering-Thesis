#pragma once

#include "individual.h"
#include "logger.h"
#include "model.h"
using std::pair;
using std::vector;

/******************************************************************************
 * @class Parents
 *
 * @brief Represents parent pairs for crossover purposes of genetic algorithm
 *
 * @public @param firstParent           First of two parents
 * @public @param secondParent          Second of two parents
 * @public @param firstParentIndex      Index of first parent
 * @public @param secondParentIndex     Index of second parent
 * @public @param isFirstParentPresent  Indicates if first parent was added
 * @public @param isSecondParentPresent Indicates if second parent was added
 ******************************************************************************/
class Parents
{
public:
  /******************************************************************************
   * CLASS MEMBERS
   ******************************************************************************/
  Individual firstParent;
  Individual secondParent;
  size_t firstParentIndex;
  size_t secondParentIndex;
  bool isFirstParentPresent{false};
  bool isSecondParentPresent{false};

  /******************************************************************************
   * CONSTRUCTORS
   ******************************************************************************/
  Parents() = default;
  Parents(Individual firstParent, size_t firstParentIndex, Individual secondParent, size_t secondParentIndex);

  /******************************************************************************
   * UTILITIES
   ******************************************************************************/
  void addParent(Individual &parent, size_t parentIndex);
};
