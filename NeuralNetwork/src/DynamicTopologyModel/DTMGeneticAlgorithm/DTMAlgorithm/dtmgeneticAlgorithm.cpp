#include "dtmgeneticAlgorithm.h"
#include "logger.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

/******************************************************************************
 * @brief Stores evolution settings and a borrowed, non-null fitness strategy
 *
 * @param hyperparameters Population and genetic operator configuration
 * @param fitnessEvaluation Evaluator that must outlive the algorithm
 * @throws std::invalid_argument If the evaluator pointer is null
 ******************************************************************************/
DTMGeneticAlgorithm::DTMGeneticAlgorithm(Hyperparameters hyperparameters,
                                         DTMFitnessEvaluation *fitnessEvaluation)
    : hyperparameters(std::move(hyperparameters)), fitnessEvaluation(fitnessEvaluation)
{
  if (fitnessEvaluation == nullptr)
  {
    throw std::invalid_argument("A fitness evaluator is required");
  }
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

/******************************************************************************
 * @brief Writes the genetic algorithm state to an output stream
 *
 * @param os                  Output stream to write to
 * @param dtmGeneticAlgorithm Genetic algorithm whose state should be written
 *
 * @return The output stream after writing the genetic algorithm state
 ******************************************************************************/
std::ostream &operator<<(std::ostream &os, const DTMGeneticAlgorithm &dtmGeneticAlgorithm)
{
  os << "DT GENETIC ALGORITHM ALGORITHM: " << std::endl;
  os << "HYPERPARAMETERS: " << std::endl;
  os << dtmGeneticAlgorithm.hyperparameters;
  os << "POPULATION: " << std::endl;
  for (auto individual: dtmGeneticAlgorithm.population)
  {
    os << individual;
  }
  return os;
}

/******************************************************************************
* UTILITIES
******************************************************************************/

/******************************************************************************
 * @brief Generates and returns the next unique synapse identifier
 *
 * @return A unique synapse identifier
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueSynapseId()
{
  return uniqueSynapseIdCounter++;
}

/******************************************************************************
 * @brief Generates the next unique hidden-neuron identifier
 *
 * IDs below inputSize + outputSize are reserved for the fixed input/output
 * neurons. The counter is retained across generations; crossover also advances
 * it past neuron IDs already present in the population.
 *
 * @return A new hidden-neuron identifier outside the reserved range
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueNeuronId()
{
  uniqueNeuronIdCounter = std::max(
      uniqueNeuronIdCounter, hyperparameters.inputSize + hyperparameters.outputSize);
  return uniqueNeuronIdCounter++;
}

/******************************************************************************
 * @brief Generates and returns the next unique individual identifier
 *
 * @return A unique individual identifier
 ******************************************************************************/
size_t DTMGeneticAlgorithm::getNewUniqueIndividualCounter()
{
  return uniqueIndividualIdCounter++;
}

/******************************************************************************
 * @brief Runs dynamic-topology evolution and returns the best evaluated individual
 *
 * Initializes a fresh population using hyperparameters. Each generation groups
 * models by depth, evaluates fitness and saves any global improvement, selects
 * parents, creates offspring, and attempts the configured mutations. Mutation
 * counts select distinct individuals per type; different types may select the
 * same individual. Failed mutation attempts are not retried.
 *
 * The initial population is generation zero. Each crossover advances
 * currentGeneration. The final offspring population is left unevaluated, since
 * evaluation occurs at the start of each iteration. The returned individual is
 * an independent copy from the evaluated generations, including generation zero.
 *
 * @param numberOfGenerations Number of evaluation/reproduction iterations, > 0
 *
 * @return Individual with the highest fitness seen during evaluation
 * @throws std::invalid_argument For invalid generation/population/tournament
 *         sizes, zero model dimensions, or mismatched/out-of-range mutation counts
 * @throws Any exception raised by the evaluator, rethrown on the calling thread
 ******************************************************************************/
DTIndividual DTMGeneticAlgorithm::run(size_t numberOfGenerations)
{
  if (numberOfGenerations == 0 || hyperparameters.populationSize == 0 ||
      hyperparameters.tournamentSize == 0)
  {
    throw std::invalid_argument("Generations, population size and tournament size must be positive");
  }
  if (hyperparameters.inputSize == 0 || hyperparameters.outputSize == 0)
  {
    throw std::invalid_argument("Model input and output sizes must be positive");
  }
  if (hyperparameters.mutationTypes.size() != hyperparameters.numberOfMutations.size())
  {
    throw std::invalid_argument("Each mutation type requires a corresponding mutation count");
  }
  for (size_t count : hyperparameters.numberOfMutations)
  {
    if (count > hyperparameters.populationSize)
    {
      throw std::invalid_argument("A mutation count cannot exceed the population size");
    }
  }

  // The run timer includes calls to the individually timed GA functions.
  TIME_MEASURE_BEGIN(DTM_GENETIC_RUN);
  currentGeneration = 0;
  initializePopulation();
  std::optional<DTIndividual> bestIndividual;
  vector<size_t> mutationIndexes(population.size());

  for (size_t generation = 0; generation < numberOfGenerations; generation++)
  {
    const auto species = divideIntoSpecies();

    TIME_MEASURE_BEGIN(DTM_EVALUATE_FITNESS);
    std::exception_ptr evaluationError;
    #pragma omp parallel for
    for (DTIndividual &individual : population)
    {
      try
      {
        fitnessEvaluation->evaluateIndividual(individual);
      }
      catch (...)
      {
        // Exceptions cannot escape an OpenMP worker. Preserve one failure and
        // rethrow it on the caller after the parallel loop's implicit barrier.
        #pragma omp critical(DTM_FITNESS_ERROR)
        {
          if (!evaluationError)
          {
            evaluationError = std::current_exception();
          }
        }
      }
    }
    if (evaluationError)
    {
      TIME_MEASURE_END(DTM_EVALUATE_FITNESS);
      TIME_MEASURE_END(DTM_GENETIC_RUN);
      std::rethrow_exception(evaluationError);
    }
    for (DTIndividual &individual : population)
    {
      if (!bestIndividual || individual.fitness > bestIndividual->fitness)
      {
        // Population storage is replaced by crossover. Copy only improvements,
        // so the saved model survives replacement and subsequent mutations.
        bestIndividual = individual;
      }
    }
    TIME_MEASURE_END(DTM_EVALUATE_FITNESS);

    const auto parents = tournamentSelection(species);

    currentGeneration++;
    crossover(parents);

    TIME_MEASURE_BEGIN(DTM_MUTATE_POPULATION);
    for (size_t typeIndex = 0; typeIndex < hyperparameters.mutationTypes.size(); typeIndex++)
    {
      std::iota(mutationIndexes.begin(), mutationIndexes.end(), 0);
      const size_t count = hyperparameters.numberOfMutations[typeIndex];
      // Partial Fisher-Yates shuffle samples without replacement in O(count)
      // after filling the reusable index buffer. Models are never copied here.
      for (size_t index = 0; index < count; index++)
      {
        const size_t selected = index + rand() % (population.size() - index);
        std::swap(mutationIndexes[index], mutationIndexes[selected]);
        mutate(population[mutationIndexes[index]], hyperparameters.mutationTypes[typeIndex]);
      }
    }
    TIME_MEASURE_END(DTM_MUTATE_POPULATION);
  }

  TIME_MEASURE_END(DTM_GENETIC_RUN);
  return std::move(*bestIndividual);
}
