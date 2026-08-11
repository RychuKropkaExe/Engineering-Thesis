#include "algorithm.h"
#include <cstdlib>
#include <iostream>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

/******************************************************************************
 * @brief Constructs GenticAlgorithm paramters
 *
 * @param populationSize Initial population size
 * @param limit          Representation of maximum initial individual Model. Each created
 *                       indiviual cannot have more layers, weights, biases, activatons etc.
 *                       Than this model.
 * @return Layer
 ******************************************************************************/
GeneticAlgorithm::GeneticAlgorithm(SelectionAlgorithm *selectionAlgorithm,
                                   CrossoverAlgorithm *crossoverAlgorithm,
                                   FitnessEvaluation *fitnessEvaluation,
                                   vector<MutationAlgorithm *> mutationAlgorithms,
                                   vector<size_t> mutationRates,
                                   size_t gracePeriodLength)

{
  this->selectionAlgorithm = selectionAlgorithm;
  this->crossoverAlgorithm = crossoverAlgorithm;
  this->fitnessEvaluation = fitnessEvaluation;
  this->mutationAlgorithms = mutationAlgorithms;
  this->mutationRates = mutationRates;
  this->gracePeriodLength = gracePeriodLength;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

void GeneticAlgorithm::initializePopulation(size_t populationSize,
                                            Model limit)
{

  this->populationSize = populationSize;

  population.resize(populationSize);

  size_t numberOfLayers{limit.archSize};
  std::vector<size_t> maxNumOfNeuronsInLayer{limit.arch};

  for (size_t index = 0; index < populationSize; index++)
  {
    size_t archSize = numberOfLayers;

    std::vector<size_t> arch{};
    std::vector<ActivationFunctionE> actFunctions{};

    arch.resize(archSize);
    actFunctions.resize(archSize);

    for (size_t layerIndex = 0; layerIndex < archSize; layerIndex++)
    {
      size_t layerSize = ((size_t)rand() % maxNumOfNeuronsInLayer[layerIndex]) + 1;

      arch[layerIndex] = layerSize;
      // Default to sigmoid activation function
      actFunctions[layerIndex] = SIGMOID;
    }

    population[index] = Individual(Model(arch, actFunctions, true));
  }
}

Individual GeneticAlgorithm::runGeneticAlgorithm(size_t numberOfIterations)
{
  Individual bestIndividual{};
  double bestFitness = 0.0;

  size_t interval = numberOfIterations / 10;

  size_t percentCompletion = 0;

  for (size_t generation = 0; generation < numberOfIterations; generation++)
  {
    interval--;

    if (interval == 0)
    {
      percentCompletion += 10;
      LOG(ESSENTIAL_LOGS, INFO_TYPE, "ALGORITHM COMPLETION: " << percentCompletion);
      interval = numberOfIterations / 10;
    }

    Individual::maxGeneration = generation;
    for (size_t index = 0; index < populationSize; index++)
    {
      // LOG(ESSENTIAL_LOGS, INFO_TYPE, "GENERATION: " << generation << std::endl << population[index]);
      fitnessEvaluation->evaluateIndividual(population[index]);

      if (population[index].fitness > bestFitness)
      {
        bestIndividual = population[index];
        bestFitness = population[index].fitness;
        LOG(ESSENTIAL_LOGS, INFO_TYPE, "NEW BEST INDIVIDUAL WITH FITNESS: " << bestFitness << " FROM GENERATION: " << Individual::maxGeneration);
      }
    }

    vector<Parents> parents = selectionAlgorithm->selectParents(population, populationSize);

    crossoverAlgorithm->produceOffspring(parents, population);

    for (size_t algorithmIndex = 0; algorithmIndex < mutationAlgorithms.size(); algorithmIndex++)
    {

      for (size_t mutationIndex = 0; mutationIndex < mutationRates[algorithmIndex]; mutationIndex++)
      {
        size_t IndividualIndex = rand() % populationSize;

        mutationAlgorithms[algorithmIndex]->mutateIndividual(population[IndividualIndex]);

        population[IndividualIndex].gracePeriodLength = gracePeriodLength;
      }
    }
  }

  return bestIndividual;
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

std::ostream &operator<<(std::ostream &os, const GeneticAlgorithm &dt)
{
  os << "POPULATION: " << std::endl;
  auto specimenCount = 0;
  for (auto individual : dt.population)
  {
    os << "SPECIMEN: " << specimenCount << std::endl;
    os << individual << std::endl;
  }
  os << std::flush;
  return os;
}
