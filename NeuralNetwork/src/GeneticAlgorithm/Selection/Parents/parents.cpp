#include "parents.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/
Parents::Parents(Individual firstParent, size_t firstParentIndex, Individual secondParent, size_t secondParentIndex)
{
  this->firstParent = firstParent;
  this->firstParentIndex = firstParentIndex;
  isFirstParentPresent = true;
  this->secondParent = secondParent;
  this->secondParentIndex = secondParentIndex;
  isSecondParentPresent = true;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

/******************************************************************************
 * @brief Adds parent to the class
 *
 * @param parent      Parent
 * @param parentIndex Parent index in population vector
 *
 * @return Current index of last non filled entry in parents list
 ******************************************************************************/
void Parents::addParent(Individual &parent, size_t parentIndex)
{
  if (isFirstParentPresent && isSecondParentPresent)
  {
    LOG(ESSENTIAL_LOGS, ERROR_TYPE, "BOTH PARENTS ALREADY PRESENT IN CLASS");
  }

  if (isFirstParentPresent)
  {
    secondParent = parent;
    isSecondParentPresent = true;
    secondParentIndex = parentIndex;
  }
  else
  {
    firstParent = parent;
    isFirstParentPresent = true;
    firstParentIndex = parentIndex;
  }
}
