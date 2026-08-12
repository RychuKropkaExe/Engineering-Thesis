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

  individual.fitness = 1 / mmseCost;
}
