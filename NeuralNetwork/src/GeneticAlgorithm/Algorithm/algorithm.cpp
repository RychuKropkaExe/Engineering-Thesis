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
GeneticAlgorithm::GeneticAlgorithm(size_t populationSize, Model limit)
{
  population.resize(populationSize);

  size_t maxNumOfLayers{limit.archSize};
  std::vector<size_t> maxNumOfNeuronsInLayer{limit.arch};

  for (size_t index = 0; index < populationSize; index++)
  {
    size_t archSize = ((size_t)rand() % maxNumOfLayers) + 1;

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

    population[index] = Individual(Model(arch, archSize, actFunctions, archSize, true));
  }
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
