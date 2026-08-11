#pragma once
#include "addNeuron.h"
#include "adjustWeight.h"
#include "algorithm.h"
#include "changeActivation.h"
#include "crossoverAlgorithm.h"
#include "fitnessEvaluation.h"
#include "flipBias.h"
#include "flipWeight.h"
#include "individual.h"
#include "mmseEval.h"
#include "model.h"
#include "mutationAlgorithm.h"
#include "noCrossover.h"
#include "selectionAlgorithm.h"
#include "testUtils.h"
#include "tournamentSelection.h"
#include "trainingData.h"
#include "weightedCrossover.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
using std::vector;

// /******************************************************************************
//  * @brief Tests if NeuralNetwork is able to model XOR logic gate
//  ******************************************************************************/
// TEST(GeneticAlgorithmTest, GAxorModelTest)
// {
//   TrainingData td = TrainingData(getTestDataPath(std::string("xorData.txt")));
//   vector<size_t> arch = {2, 2, 2, 1};
//   vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};

//   TournamentSelection ts = TournamentSelection();
//   WeightedCrossover wc = WeightedCrossover(1.05);
//   MmseEval fe = MmseEval(td);

//   SelectionAlgorithm *selectionAlgorithm = &ts;
//   CrossoverAlgorithm *crossoverAlgorithm = &wc;
//   FitnessEvaluation *fitnessEvaluation = &fe;

//   FlipBias fp = FlipBias(1);
//   FlipWeight fw = FlipWeight(1);
//   AddNeuron an = AddNeuron();
//   AdjustWeight aw = AdjustWeight(2, 1.05);
//   ChangeActivation ca = ChangeActivation(1);

//   vector<MutationAlgorithm *> mutationAlgorithms = {&fp, &fw, &an, &aw, &ca};
//   vector<size_t> mutationRates = {1, 1, 20, 50, 1};

//   Model model(arch, actFunc, true);

//   GeneticAlgorithm algorithm = GeneticAlgorithm(selectionAlgorithm, crossoverAlgorithm, fitnessEvaluation, mutationAlgorithms, mutationRates, 0);

//   algorithm.initializePopulation(1000, model);

//   Individual result = algorithm.runGeneticAlgorithm(10000);

//   double cost = result.genotype.costMeanSquare(td);

//   EXPECT_LE(cost, 0.05f);
// }

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to model parabole on numbers in (-20, 20)
 ******************************************************************************/
TEST(ModelTest, GAparaboleModelTest)
{
  vector<size_t> arch = {1, 2, 2, 1};

  vector<vector<double>> trainingInputs;
  vector<vector<double>> trainingOutputs;

  size_t numberOfSamples = 1000;

  size_t inputSize = 1;
  size_t outputSize = 1;

  trainingInputs.resize(numberOfSamples);
  trainingOutputs.resize(numberOfSamples);

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    trainingInputs[i].resize(inputSize);
    trainingOutputs[i].resize(outputSize);
  }

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    double inputValue = -20.f + ((double)i / 25.f);
    trainingInputs[i][0] = inputValue;
    trainingOutputs[i][0] = inputValue * inputValue;
  }

  TrainingData td = TrainingData(trainingInputs, inputSize, numberOfSamples, trainingOutputs, outputSize, numberOfSamples);

  vector<ActivationFunctionE> actFunc = {RELU, RELU, RELU, RELU};

  td.normalizeData(MIN_MAX_NORMALIZATION);

  TournamentSelection ts = TournamentSelection();
  WeightedCrossover wc = WeightedCrossover(1.05);
  MmseEval fe = MmseEval(td);

  SelectionAlgorithm *selectionAlgorithm = &ts;
  CrossoverAlgorithm *crossoverAlgorithm = &wc;
  FitnessEvaluation *fitnessEvaluation = &fe;

  FlipBias fp = FlipBias(1);
  FlipWeight fw = FlipWeight(1);
  AddNeuron an = AddNeuron();
  AdjustWeight aw = AdjustWeight(5, 1.025);
  ChangeActivation ca = ChangeActivation(1);

  vector<MutationAlgorithm *> mutationAlgorithms = {&fp, &fw, &an, &aw, &ca};
  vector<size_t> mutationRates = {0, 0, 5, 10, 1};

  Model model(arch, actFunc, true);

  model.modelXavierInitialize();

  GeneticAlgorithm algorithm = GeneticAlgorithm(selectionAlgorithm, crossoverAlgorithm, fitnessEvaluation, mutationAlgorithms, mutationRates, 0);

  algorithm.initializePopulation(100, model);

  Individual result = algorithm.runGeneticAlgorithm(10000);

  double cost = result.genotype.costMeanSquare(td);

  EXPECT_LE(cost, 0.10f);
}

// /******************************************************************************
//  * @brief Tests if NeuralNetwork is able to predict if given 8-bit number is
//  *        even or odd
//  ******************************************************************************/
// TEST(GeneticAlgorithmTest, GAparityModelTest)
// {
//   TrainingData td = TrainingData(getTestDataPath(std::string("parityTestData.txt")));
//   vector<size_t> arch = {8, 2, 1};

//   vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

//   Model model(arch, actFunc, true);

//   SelectionAlgorithm selectionAlgorithm = TournamentSelection();
//   CrossoverAlgorithm crossoverAlgorithm = WeightedCrossover(1.05);
//   FitnessEvaluation fitnessEvaluation = MmseEval(td);
//   vector<MutationAlgorithm> mutationAlgorithms = {FlipBias(1), FlipWeight(1), AddNeuron(), AdjustWeight(1, 1.05), ChangeActivation(1)};
//   vector<size_t> mutationRates = {5, 5, 5, 5, 5};

//   GeneticAlgorithm algorithm = GeneticAlgorithm(selectionAlgorithm, crossoverAlgorithm, fitnessEvaluation, mutationAlgorithms, mutationRates, 3);

//   algorithm.initializePopulation(1000, model);

//   Individual result = algorithm.runGeneticAlgorithm(10000);

//   double cost = result.genotype.costMeanSquare();

//   EXPECT_LE(cost, 0.05f);
// }

// /******************************************************************************
//  * @brief Tests if NeuralNetwork is able to calculate hamming length of 7-bit number
//  ******************************************************************************/
// TEST(ModelTest, hammingLengthTest)
// {
//   TrainingData td = TrainingData(getTestDataPath(std::string("hammingLengthTest.txt")));
//   vector<size_t> arch = {7, 10, 10, 3};

//   vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

//   Model model(arch, actFunc, true);

//   double learningRate = 1e-1;

//   model.setLearningRate(learningRate);

//   model.learn(td, 300000, false, 32);
//   double cost = model.costMeanSquare();

//   EXPECT_LE(cost, 0.10f);
// }

// /******************************************************************************
//  * @brief Tests if NeuralNetwork is able to recognize digits given their
//  *        features
//  ******************************************************************************/
// TEST(ModelTest, digitRecognitionTest)
// {
//   TrainingData td = TrainingData(getTestDataPath(std::string("pendigits.tra")));
//   td.normalizeData(MIN_MAX_NORMALIZATION);
//   vector<size_t> arch = {16, 10, 10, 1};

//   vector<ActivationFunctionE> actFunc = {RELU, RELU, RELU};

//   Model model(arch, actFunc, true);
//   model.modelXavierInitialize();

//   double learningRate = 1e-3;

//   model.setLearningRate(learningRate);

//   model.learn(td, 100000, true, 128);

//   td = TrainingData(getTestDataPath(std::string("pendigits.tes")));
//   td.normalizeData(MIN_MAX_NORMALIZATION);
//   model.trainingData = td;
//   double cost = model.costMeanSquare();
//   for (size_t i = 0; i < td.numOfSamples; i++)
//   {
//     LOG(NORMAL_LOGS, INFO_TYPE, "PREDICTION FOR SAMPLE: " << td.inputs[i]);
//     FastMatrix prediction = model.run(td.inputs[i]);
//     td.denormalizeOutput(MIN_MAX_NORMALIZATION, prediction);
//     LOG(NORMAL_LOGS, INFO_TYPE, "PREDICTION RESULT: " << prediction);
//   }

//   LOG(HEAVY_LOGS, INFO_TYPE, "CURRENT MODEL: " << model);

//   LOG(ESSENTIAL_LOGS, INFO_TYPE, "COST VALUE: " << cost);

//   EXPECT_LE(cost, 0.10f);
// }
