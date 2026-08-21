#include "gaparents.h"
#include <cassert>
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
void GAParents::addParent(size_t parentIndex)
{
  assert(!(isFirstParentPresent && isSecondParentPresent));

  if (isFirstParentPresent)
  {
    isSecondParentPresent = true;
    secondParentIndex = parentIndex;
  }
  else
  {
    isFirstParentPresent = true;
    firstParentIndex = parentIndex;
  }
}
