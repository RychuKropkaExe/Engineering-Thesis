#pragma once

#include "dtmgeneticAlgorithm.h"
#include "trainingData.h"
#include "testUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <stdexcept>

/******************************************************************************
 * @brief Tests depth-zero species, empty populations and population index mapping
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, speciationAndSelectionTest)
{
  Hyperparameters parameters{};
  parameters.tournamentSize = 3;
  DTMGeneticAlgorithm algorithm(parameters);
  EXPECT_TRUE(algorithm.divideIntoSpecies().empty());
  EXPECT_TRUE(algorithm.tournamentSelection({}).empty());

  DTModel connected(1, 1, ActivationE::SIGMOID);
  connected.addOutSynapse(Synapse(100, 0, 1, 0.5), false);
  algorithm.population.emplace_back(10, 0, connected);
  algorithm.population.emplace_back(20, 0, DTModel(1, 1, ActivationE::SIGMOID));
  algorithm.population.emplace_back(30, 0, connected);
  const auto species = algorithm.divideIntoSpecies();
  ASSERT_EQ(species.size(), 2u);
  EXPECT_EQ(species[0], (vector<size_t>{1}));
  EXPECT_EQ(species[1], (vector<size_t>{0, 2}));

  // Give the singleton the greatest fitness. A species-local index must never
  // be mistaken for a population index and let it breed in the other species.
  algorithm.population[0].fitness = -10.0;
  algorithm.population[1].fitness = 1000.0;
  algorithm.population[2].fitness = -1.0;
  algorithm.population[2].gracePeriodLength = 2;
  const auto parents = algorithm.tournamentSelection(species);
  ASSERT_EQ(parents[0].size(), 1u);
  ASSERT_EQ(parents[1].size(), 2u);
  EXPECT_EQ(parents[0][0].firstParentIndex, 1u);
  EXPECT_EQ(parents[0][0].secondParentIndex, 1u);
  EXPECT_EQ(parents[1][0].firstParentIndex, 2u);
  EXPECT_EQ(parents[1][0].secondParentIndex, 2u);
  EXPECT_EQ(algorithm.population[2].gracePeriodLength, 2u);
  for (const GAParents &pair : parents[1])
  {
    EXPECT_NE(pair.firstParentIndex, 1u);
    EXPECT_NE(pair.secondParentIndex, 1u);
  }
  algorithm.currentGeneration = 1;
  algorithm.crossover(parents);
  ASSERT_EQ(algorithm.population.size(), 3u);
  EXPECT_EQ(algorithm.population[1].gracePeriodLength, 1u);
}

/******************************************************************************
 * @brief Tests unique depth IDs when an output revisits the input depth
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, disconnectedOutputCrossoverTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 2;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 2, ActivationE::SIGMOID);
  model.addOutSynapse(Synapse(100, 0, 1, 0.5), true);
  std::set<size_t> depthIds;
  for (const Neuron &neuron : model.neurons)
  {
    EXPECT_TRUE(depthIds.insert(neuron.depthId).second);
  }
  algorithm.population.emplace_back(10, 0, model);
  GAParents pair;
  pair.addParent(0);
  pair.addParent(0);
  algorithm.crossover({{pair}});
  ASSERT_EQ(algorithm.population.size(), 1u);
  EXPECT_TRUE(algorithm.population[0].model.hasSynapse(0, 1));
  EXPECT_FALSE(algorithm.population[0].model.hasSynapse(0, 0));
  EXPECT_TRUE(algorithm.population[0].model.validateModel());
}

/******************************************************************************
 * @brief Tests best-so-far preservation, fresh runs and mutation counts
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, runTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.outputActivation = ActivationE::SIGMOID;
  parameters.populationSize = 12;
  parameters.maxNumberOfNeurons = 3;
  parameters.tournamentSize = 3;
  parameters.gracePeriodLength = 2;
  parameters.mutationTypes = {MutationE::ADD_NEURON};
  parameters.numberOfMutations = {5};
  const TrainingData data({{0.0}, {1.0}}, 1, 2, {{0.2}, {0.8}}, 1, 2);

  // Reproduce the initial population independently to know its true best.
  srand(1234);
  DTMGeneticAlgorithm initial(parameters);
  initial.initializePopulation();
  double initialBest = 0.0;
  for (DTIndividual &individual : initial.population)
  {
    DTMGeneticAlgorithm::evaluateIndividual(individual, data);
    initialBest = std::max(initialBest, individual.fitness);
  }

  srand(1234);
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual best = algorithm.run(1, data);
  EXPECT_NEAR(best.fitness, initialBest, 1e-9);
  EXPECT_EQ(best.generation, 0u);
  EXPECT_EQ(best.model.neurons.size(), 2u);
  EXPECT_EQ(algorithm.currentGeneration, 1u);
  EXPECT_EQ(algorithm.population.size(), parameters.populationSize);
  EXPECT_EQ(std::count_if(algorithm.population.begin(), algorithm.population.end(),
      [](const DTIndividual &child) { return child.model.neurons.size() == 3; }), 5);
  for (const DTIndividual &child : algorithm.population)
  {
    EXPECT_EQ(child.generation, 1u);
    EXPECT_DOUBLE_EQ(child.fitness, 0.0);
    EXPECT_EQ(child.gracePeriodLength, child.model.neurons.size() == 3 ? 2u : 0u);
  }
  EXPECT_NEAR(1.0 / best.fitness,
              DTMGeneticAlgorithm::calculateMMSE(best.model, data), 1e-9);

  srand(1234);
  best = algorithm.run(4, data);
  EXPECT_GE(best.fitness, initialBest - 1e-9);
  EXPECT_LT(best.generation, 4u);
  EXPECT_EQ(algorithm.currentGeneration, 4u);
  EXPECT_EQ(algorithm.population.size(), parameters.populationSize);
  EXPECT_NEAR(1.0 / best.fitness,
              DTMGeneticAlgorithm::calculateMMSE(best.model, data), 1e-9);

  // Mutating population storage cannot change the separately saved model.
  algorithm.population.clear();
  EXPECT_NEAR(1.0 / best.fitness,
              DTMGeneticAlgorithm::calculateMMSE(best.model, data), 1e-9);
}

/******************************************************************************
 * @brief Tests invalid iteration and mutation settings before starting evolution
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, invalidRunSettingsTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.outputActivation = ActivationE::SIGMOID;
  parameters.populationSize = 4;
  parameters.tournamentSize = 2;
  DTMGeneticAlgorithm algorithm(parameters);
  const TrainingData data({{0.0}}, 1, 1, {{1.0}}, 1, 1);
  EXPECT_THROW(algorithm.run(0, data), std::invalid_argument);
  algorithm.hyperparameters.mutationTypes = {MutationE::ADD_NEURON};
  EXPECT_THROW(algorithm.run(1, data), std::invalid_argument);
  algorithm.hyperparameters.numberOfMutations = {5};
  EXPECT_THROW(algorithm.run(1, data), std::invalid_argument);
  EXPECT_TRUE(algorithm.population.empty());
}

/******************************************************************************
 * @brief Tests evolution of the three-bit Hamming count of a seven-bit input
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, hammingLengthTest)
{
  const TrainingData data(getTestDataPath("hammingLengthTest.txt"));
  Hyperparameters parameters{};
  parameters.inputSize = 7;
  parameters.outputSize = 3;
  parameters.outputActivation = ActivationE::SIGMOID;
  parameters.gracePeriodLength = 2;
  parameters.populationSize = 100;
  parameters.maxNumberOfNeurons = 40;
  parameters.tournamentSize = 5;
  parameters.mutationTypes = {MutationE::ADD_NEURON, MutationE::REMOVE_NEURON,
                              MutationE::ADD_SYNAPSE, MutationE::REMOVE_SYNAPSE,
                              MutationE::ADJUST_WEIGHT, MutationE::ADJUST_BIAS,
                              MutationE::CHANGE_ACTIVATION};
  // Keep parameter mutations relatively sparse: successful mutations receive
  // two protected generations, so excessive mutation can crowd out selection.
  parameters.numberOfMutations = {2, 1, 4, 1, 10, 10, 1};
  parameters.weightMutationStrength = 2.0;
  parameters.biasMutationStrength = 2.0;

  srand(2026);
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual best = algorithm.run(4000, data);
  const double cost = 1.0 / best.fitness;
  RecordProperty("cost", std::to_string(cost));
  EXPECT_NEAR(cost, DTMGeneticAlgorithm::calculateMMSE(best.model, data), 1e-9);
  EXPECT_LE(cost, 0.05);
}

/******************************************************************************
 * @brief Tests evolution of digit recognition from 16 pen-stroke features
 ******************************************************************************/
TEST(DTMGeneticAlgorithmTest, digitRecognitionTest)
{
  TrainingData trainingData(getTestDataPath("pendigits.tra"));
  trainingData.normalizeData(MIN_MAX_NORMALIZATION);

  Hyperparameters parameters{};
  parameters.inputSize = 16;
  parameters.outputSize = 1;
  parameters.outputActivation = ActivationE::RELU;
  parameters.gracePeriodLength = 2;
  parameters.populationSize = 50;
  parameters.maxNumberOfNeurons = 40;
  parameters.tournamentSize = 5;
  parameters.mutationTypes = {MutationE::ADD_NEURON, MutationE::REMOVE_NEURON,
                              MutationE::ADD_SYNAPSE, MutationE::REMOVE_SYNAPSE,
                              MutationE::ADJUST_WEIGHT, MutationE::ADJUST_BIAS,
                              MutationE::CHANGE_ACTIVATION};
  parameters.numberOfMutations = {1, 1, 2, 1, 5, 5, 1};
  parameters.weightMutationStrength = 2.0;
  parameters.biasMutationStrength = 2.0;

  srand(2027);
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual best = algorithm.run(3000, trainingData);
  const double cost = 1.0 / best.fitness;
  RecordProperty("cost", std::to_string(cost));
  EXPECT_NEAR(cost, DTMGeneticAlgorithm::calculateMMSE(best.model, trainingData), 1e-9);
  EXPECT_LE(cost, 0.05);

  TrainingData testData(getTestDataPath("pendigits.tes"));
  testData.normalizeData(MIN_MAX_NORMALIZATION);
  RecordProperty("heldOutCost", std::to_string(
      DTMGeneticAlgorithm::calculateMMSE(best.model, testData)));
}
