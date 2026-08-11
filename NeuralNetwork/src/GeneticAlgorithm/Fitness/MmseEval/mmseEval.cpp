#include "mmseEval.h"

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

MmseEval::MmseEval(TrainingData td)
{
  expectedResults = td;
  expectedResults.normalizeData(MIN_MAX_NORMALIZATION);
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void MmseEval::evaluateIndividual(Individual &individual)
{
  double mmseCost = individual.genotype.costMeanSquare(expectedResults);

  // LOG(ESSENTIAL_LOGS, INFO_TYPE, "Individual with id: " << individual.id << " OF GENERATION: " << individual.generation << " GOT MMSE COST: " << mmseCost);

  individual.fitness = 1 / mmseCost;
}
