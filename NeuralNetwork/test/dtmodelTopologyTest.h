#pragma once

#include "dtmodel.h"
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @brief Tests topologicall sorting of Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, topologicallSortingTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  // Since sorting may not preserve order of neurons that
  // are equall in placement we define boundries for them.
  vector<size_t> neuronMinimalIndexAfterSort{0, 0, 2, 2};
  vector<size_t> neuronMaximalIndexAfterSort{1, 1, 3, 3};
  vector<size_t> expectedDepthId{0, 1, 1000, 1001};

  testModel.sortTopologically();

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronId = testModel.neurons[index].id;

    size_t neuronIndex = testModel.indexMap[neuronId];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);
    EXPECT_EQ(expectedDepthId[index], testModel.neurons[neuronIndex].depthId);

  }

  vector<size_t> neuronsIdToAdd{4, 5};

  // Add two hidden neurons between input and output layer
  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  // First two are input neurons, next two are outputs, last two are hidden
  neuronMinimalIndexAfterSort = vector<size_t>{0, 0, 4, 4, 2, 2};
  neuronMaximalIndexAfterSort = vector<size_t>{1, 1, 5, 5, 3, 3};
  expectedDepthId = vector<size_t>{0, 1, 2000, 2001, 1000, 1001};

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronIndex = testModel.indexMap[index];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);
    EXPECT_EQ(expectedDepthId[index], testModel.neurons[neuronIndex].depthId);

  }

  // Add "third layer" to the network
  vector<size_t> previousNeuronsIds{4, 5};
  neuronsIdToAdd = vector<size_t>{6, 7};

  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, previousNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  // First two are input neurons, next two are outputs, next two are hidden, last two are hidden
  // going from the previous hidden ones.
  neuronMinimalIndexAfterSort = vector<size_t>{0, 0, 6, 6, 2, 2, 4, 4};
  neuronMaximalIndexAfterSort = vector<size_t>{1, 1, 7, 7, 3, 3, 5, 5};
  expectedDepthId = vector<size_t>{0, 1, 3000, 3001, 1000, 1001, 2000, 2001};

  vector<size_t> expectedDepth{0, 0, 3, 3, 1, 1, 2, 2};

  for (size_t index = 0; index < testModel.neurons.size(); index++)
  {
    size_t neuronIndex = testModel.indexMap[index];

    EXPECT_GE(neuronIndex, neuronMinimalIndexAfterSort[index]);
    EXPECT_LE(neuronIndex, neuronMaximalIndexAfterSort[index]);
    EXPECT_EQ(testModel.neurons[neuronIndex].depth, expectedDepth[index]);
    EXPECT_EQ(expectedDepthId[index], testModel.neurons[neuronIndex].depthId);

  }

}

/******************************************************************************
 * @brief Tests model validation of Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, validateModelTest)
{
  size_t inputSize = 2;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1};
  vector<size_t> outputNeuronsIds{2, 3};

  DTModel testModel(inputSize, outputSize, ActivationE::NO_ACTIVATION);

  size_t synapseId = 0;

  // Connect every input to every output
  for (size_t inputNeuronId : inputNeuronsIds)
  {
    for (size_t outputNeuronsId : outputNeuronsIds)
    {
      Synapse newSynapse(synapseId, inputNeuronId, outputNeuronsId, 1.0);
      synapseId++;
      testModel.addOutSynapse(newSynapse, false);
    }
  }

  EXPECT_TRUE(testModel.validateModel());

  vector<size_t> neuronsIdToAdd{4, 5};

  // Add two hidden neurons between input and output layer
  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  EXPECT_TRUE(testModel.validateModel());

  // Remove one of added synapses, invalidating the model
  testModel.removeSynapse(inputNeuronsIds[0], neuronsIdToAdd[0], false);

  EXPECT_FALSE(testModel.validateModel());

  // Add the removed synapse back
  testModel.addOutSynapse(Synapse(synapseId,inputNeuronsIds[0], neuronsIdToAdd[0], 1.0), false);
  synapseId++;

  EXPECT_TRUE(testModel.validateModel());

  // Add "third layer" to the network
  vector<size_t> previousNeuronsIds{4, 5};
  neuronsIdToAdd = vector<size_t>{6, 7};

  for (size_t index = 0; index < neuronsIdToAdd.size(); index++)
  {
    Synapse inSynapse(synapseId, previousNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, true);
  }

  EXPECT_TRUE(testModel.validateModel());

  // Remove one of added synapses, invalidating the model
  testModel.removeSynapse(previousNeuronsIds[0], neuronsIdToAdd[0], false);

  EXPECT_FALSE(testModel.validateModel());

  // Add the removed synapse back
  testModel.addOutSynapse(Synapse(synapseId, previousNeuronsIds[0], neuronsIdToAdd[0], 1.0), false);
  synapseId++;

  EXPECT_TRUE(testModel.validateModel());

}

