#pragma once
#include "dtmodel.h"
#include "dtmgeneticAlgorithm.h"
#include "utils.h"
#include "testUtils.h"
#include "trainingData.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>

using std::vector;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @brief Tests adding a synapse to Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, addOutSynapseTest)
{

  size_t inputSize = 3;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1, 2};
  vector<size_t> outputNeuronsIds{3, 4};

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

  // Verify that every input neuron connects to every output neuron
  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::OUTPUT_NEURON)
    {
      continue;
    }
    EXPECT_EQ(neuron.outSynapses.size(), outputSize) << "FOR NEURON WITH ID: " << neuron.id;

    vector<bool> isOutputIdPresent{false, false};

    for (auto synapse : neuron.outSynapses)
    {
      assert(synapse.outNeuronId >= inputSize);
      isOutputIdPresent[synapse.outNeuronId - inputSize] = true;
    }

    EXPECT_TRUE(isOutputIdPresent[0]);
    EXPECT_TRUE(isOutputIdPresent[1]);

  }

}

/******************************************************************************
 * @brief Tests adding a neuron to Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, addNeuronTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  size_t eachLayerSize = 2;

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

  vector<size_t> neuronsIdToAdd{4, 5};

  for (size_t index = 0; index < eachLayerSize; index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, false);
  }

  vector<bool> newNeuronsFound{false, false};
  vector<bool> newInConnectionsFound{false, false};
  vector<bool> newOutConnectionsFound{false, false};

  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
    {
      size_t boolTableIndex = neuron.id - (inputSize + outputSize);
      newNeuronsFound[boolTableIndex] = true;
    }
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.outNeuronId >= (inputSize + outputSize))
      {
        // New connection can only come from input neurons
        EXPECT_LT(synapse.inNeuronId, inputSize) << "NEURON ID COMING INTO NEW NEURON IS TOO HIGH";
        newOutConnectionsFound[synapse.outNeuronId - (inputSize + outputSize)] = true;
      }

      if (synapse.inNeuronId >= (inputSize + outputSize))
      {
        // New connection can only feed into output neuron
        EXPECT_GE(synapse.outNeuronId, inputSize) << "NEURON ID COMING OUT OF NEW NEURON IS TOO LOW";
        // Ensure it does not go into one of the new neurons
        EXPECT_LT(synapse.outNeuronId, inputSize + outputSize) << "NEURON ID COMING OUT OF NEW NEURON IS TOO HIGH";
        newInConnectionsFound[synapse.inNeuronId - (inputSize + outputSize)] = true;
      }
    }
  }

  EXPECT_TRUE(newNeuronsFound[0]) << "DID NOT FOUND FIRST NEW NEURON";
  EXPECT_TRUE(newNeuronsFound[1]) << "DID NOT FOUND SECOND NEW NEURON";

  EXPECT_TRUE(newInConnectionsFound[0]) << "DID NOT FOUND CONNECTION FEEDING INTO FIRST NEW NEURON";
  EXPECT_TRUE(newInConnectionsFound[1]) << "DID NOT FOUND CONNECTION FEEDING INTO SECOND NEW NEURON";

  EXPECT_TRUE(newOutConnectionsFound[0]) << "DID NOT FOUND CONNECTION GOING OUT FROM FIRST NEW NEURON";
  EXPECT_TRUE(newOutConnectionsFound[0]) << "DID NOT FOUND CONNECTION GOING OUT FROM SECOND NEW NEURON";

}

/******************************************************************************
 * @brief Tests removing a neuron from Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, removeNeuronTest)
{

  size_t inputSize = 2;
  size_t outputSize = 2;

  size_t eachLayerSize = 2;

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

  vector<size_t> neuronsIdToAdd{4, 5};

  for (size_t index = 0; index < eachLayerSize; index++)
  {
    Synapse inSynapse(synapseId, inputNeuronsIds[index], neuronsIdToAdd[index], 1.0);
    synapseId++;
    Synapse outSynapse(synapseId, neuronsIdToAdd[index], outputNeuronsIds[index], 1.0);
    synapseId++;

    Neuron neuronToAdd(neuronsIdToAdd[index], NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);

    testModel.addNeuron(neuronToAdd, inSynapse, outSynapse, false);
  }

  testModel.removeNeuron(neuronsIdToAdd[0], false);
  testModel.removeNeuron(neuronsIdToAdd[1], false);

  vector<bool> newNeuronsFound{false, false};
  vector<bool> newInConnectionsFound{false, false};

  for (auto neuron : testModel.neurons)
  {
    if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
    {
      size_t boolTableIndex = neuron.id - (inputSize + outputSize);
      newNeuronsFound[boolTableIndex] = true;
    }
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.inNeuronId >= (inputSize + outputSize))
      {
        newInConnectionsFound[synapse.inNeuronId - (inputSize + outputSize)] = true;
      }
    }
  }

  EXPECT_FALSE(newNeuronsFound[0]) << "DID FOUND FIRST REMOVED NEURON";
  EXPECT_FALSE(newNeuronsFound[1]) << "DID FOUND SECOND REMOVED NEURON";

  EXPECT_FALSE(newInConnectionsFound[0]) << "DID FOUND CONNECTION FEEDING INTO FIRST REMOVED NEURON";
  EXPECT_FALSE(newInConnectionsFound[1]) << "DID FOUND CONNECTION FEEDING INTO SECOND REMOVED NEURON";

}

/******************************************************************************
 * @brief Tests removing a synapse from Dynamic Topology Model
 ******************************************************************************/
TEST(DTModelTest, removeSynapseTest)
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

  size_t removedSynapseInNeuronId = inputNeuronsIds[0];
  size_t removedSynapseOutNeuronId = outputNeuronsIds[1];

  testModel.removeSynapse(removedSynapseInNeuronId, removedSynapseOutNeuronId, false);

  bool isRemovedOutSynapsePresent = false;
  bool isRemovedInSynapsePresent = false;

  for (auto neuron : testModel.neurons)
  {
    for (auto synapse : neuron.outSynapses)
    {
      if (synapse.inNeuronId == removedSynapseInNeuronId && synapse.outNeuronId == removedSynapseOutNeuronId)
      {
        isRemovedOutSynapsePresent = true;
      }
    }

    for (auto synapse : neuron.inSynapses)
    {
      if (synapse.inNeuronId == removedSynapseInNeuronId && synapse.outNeuronId == removedSynapseOutNeuronId)
      {
        isRemovedInSynapsePresent = true;
      }
    }
  }

  EXPECT_FALSE(isRemovedOutSynapsePresent) << "DID FOUND REMOVED OUT CONNECTION";
  EXPECT_FALSE(isRemovedInSynapsePresent) << "DID FOUND REMOVED IN CONNECTION";

}

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

/******************************************************************************
 * @brief Tests feed forward of Dynamic Topology Model on following scenarios:
 *
 * All inputs set to 1.0 in all scenarios
 *
 *  First scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * expected output:
 * O1 = RELU((1.0*2.0 + 4.0*2.0) - 1.0) = 5
 * O2 = RELU((1.0*4.0 + 1.0*8.0) - 1.0) = 13
 *
 *
 *  Second scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  - 1 hidden neuron, bias set to 0.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * - I3 to H1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * expected output:
 * O1 = RELU((1.0*2.0 + 4.0*2.0 + RELU(1.0*2.0 - 0.0)*2.0) - 1.0) = 9.0
 * O2 = RELU((1.0*4.0 + 1.0*8.0) + + RELU(1.0*2.0 - 0.0)*2.0 - 1.0) = 17.0
 *
 *
 *  Second scenario:
 *  - 3 input neurons
 *  - 2 output neurons, all biases set to 1.0
 *  - 2 hidden neurons, bias set to 0.0 and 1.0
 *  - 1 deeper hidden neuron, bias set to 1.0
 *  connections:
 * - I1 to O1 with weight 2.0
 * - I1 to O2 with weight 6.0
 * - I1 to H2 with weight 8.0
 * - I2 to O1 with weight 4.0
 * - I3 to O2 with weight 8.0
 * - I3 to H1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to O1 with weight 2.0
 * - H1 to DH1 with weight 2.0
 * - H2 to DH1 with weight 1.0
 * - DH1 to O1 with weight 3.0
 * - DH1 to O1 with weight 3.0
 * expected output:
 * H1 = RELU(2.0 - 0.0) = 2
 * H2 = RELU(8.0 - 1.0) = 7
 * DH1 = RELU(H1*2.0 + H2*1.0 - 1.0) = 10
 * O1 = RELU(2.0 + H1*2.0 + DH1*3.0 + 4.0 - 1.0) = 39.0
 * O2 = RELU(8.0 + H1*2.0 + DH1*3.0 + 6.0) = 47.0
 *
 ******************************************************************************/
TEST(DTModelTest, feedForwardTest)
{

  size_t inputSize = 3;
  size_t outputSize = 2;

  vector<size_t> inputNeuronsIds{0, 1, 2};
  vector<size_t> outputNeuronsIds{3, 4};

  DTModel testModel(inputSize, outputSize, ActivationE::RELU);

  testModel.setBias(outputNeuronsIds[0], 1.0);
  testModel.setBias(outputNeuronsIds[1], 1.0);

  size_t synapseId = 0;

  // Connect every input to every output

  vector<size_t> synapseInNeuronIds{0, 0, 1, 2};
  vector<size_t> synapseOutNeuronIds{3, 4, 3, 4};

  vector<double> synapsesWeights{2.0, 6.0, 4.0, 8.0};

  for (size_t index = 0; index < synapseInNeuronIds.size(); index++)
  {
    // Connect every input to every output
    Synapse newSynapse(synapseId, synapseInNeuronIds[index], synapseOutNeuronIds[index], synapsesWeights[index]);
    synapseId++;
    testModel.addOutSynapse(newSynapse, false);
  }

  vector<double> input{1.0, 1.0, 1.0};
  vector<double> expectedOutput{5.0, 13.0};

  vector<double> result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

  // Add another neuron to network, a hidden neuron connected to third input neuron
  // and to both output neurons.
  size_t neuronToAddId1 = 5;

  Synapse inSynapse1(synapseId, inputNeuronsIds[2], neuronToAddId1, 2.0);
  synapseId++;
  Synapse outSynapse1(synapseId, neuronToAddId1, outputNeuronsIds[0], 2.0);
  synapseId++;

  Neuron neuronToAdd1(neuronToAddId1, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd1, inSynapse1, outSynapse1, true);

  testModel.setBias(neuronToAddId1, 0.0);

  Synapse additionalSynapse1(synapseId, neuronToAddId1, outputNeuronsIds[1], 2.0);
  synapseId++;

  testModel.addOutSynapse(additionalSynapse1, true);

  input = vector<double>{1.0, 1.0, 1.0};
  expectedOutput = vector<double>{9.0, 17.0};

  result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

  // Add another two neurons to network, a hidden neuron connected to first input neuron
  // and to deeper hidden neuron, and a deeper hidden neuron connected to both outputs which
  // feed from both hidden neurons
  size_t neuronToAddId2 = 6;

  // Create and add first neuron, deeper neuron
  Synapse inSynapse2(synapseId, neuronToAddId1, neuronToAddId2, 2.0);
  synapseId++;
  Synapse outSynapse2(synapseId, neuronToAddId2, outputNeuronsIds[0], 3.0);
  synapseId++;

  Neuron neuronToAdd2(neuronToAddId2, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd2, inSynapse2, outSynapse2, true);

  testModel.setBias(neuronToAddId2, 1.0);

  Synapse additionalSynapse2(synapseId, neuronToAddId2, outputNeuronsIds[1], 3.0);
  synapseId++;

  testModel.addOutSynapse(additionalSynapse2, true);

  // Create and add second neuron, connected to deeper hidden neuron
  size_t neuronToAddId3 = 7;

  Synapse inSynapse3(synapseId, inputNeuronsIds[0], neuronToAddId3, 8.0);
  synapseId++;
  Synapse outSynapse3(synapseId, neuronToAddId3, neuronToAddId2, 1.0);
  synapseId++;

  Neuron neuronToAdd3(neuronToAddId3, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);

  testModel.addNeuron(neuronToAdd3, inSynapse3, outSynapse3, true);

  testModel.setBias(neuronToAddId3, 1.0);

  input = vector<double>{1.0, 1.0, 1.0};
  expectedOutput = vector<double>{39.0, 47.0};

  result = testModel.feedForward(input);

  EXPECT_EQ(result[0], expectedOutput[0]);
  EXPECT_EQ(result[1], expectedOutput[1]);

}

/******************************************************************************
 * @brief Tests finding similar neurons and synapses in two DTModels
 ******************************************************************************/
TEST(DTModelTest, findSimilarNeuronsAndSynapsesTest)
{
  auto createNeuron = [](size_t id, size_t depthId)
  {
    Neuron neuron(id, NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);
    neuron.depthId = depthId;
    return neuron;
  };

  DTModel firstModel;
  firstModel.neurons = {
      createNeuron(10, 0),
      createNeuron(11, 1000),
      createNeuron(12, 2000),
      createNeuron(13, 4000)};
  firstModel.indexMap = {{10, 0}, {11, 1}, {12, 2}, {13, 3}};

  firstModel.neurons[0].addOutSynapse(Synapse(1, 10, 11, 0.1));
  firstModel.neurons[1].addOutSynapse(Synapse(2, 11, 12, 0.2));
  firstModel.neurons[0].addOutSynapse(Synapse(3, 10, 12, 0.3));
  firstModel.neurons[0].addOutSynapse(Synapse(4, 10, 13, 0.4));

  DTModel secondModel;
  // Different global IDs and vector order ensure that a match cannot rely on
  // neuron IDs or vector positions. Endpoints must be resolved with indexMap.
  secondModel.neurons = {
      createNeuron(102, 2000),
      createNeuron(100, 0),
      createNeuron(103, 3000),
      createNeuron(101, 1000)};
  secondModel.indexMap = {{100, 1}, {101, 3}, {102, 0}, {103, 2}};

  secondModel.neurons[1].addOutSynapse(Synapse(20, 100, 101, 9.1));
  secondModel.neurons[3].addOutSynapse(Synapse(21, 101, 102, 9.2));
  secondModel.neurons[1].addOutSynapse(Synapse(22, 100, 103, 9.3));

  // Synapse activity, weight and global ID are intentionally different and
  // must not affect similarity when both endpoint depthIds are equal.
  secondModel.neurons[1].outSynapses[0].isActive = false;

  const SimilarDTModelElements result =
      DTMGeneticAlgorithm::findSimilarNeuronsAndSynapses(firstModel, secondModel);

  const vector<size_t> expectedFirstModelNeuronIds{10, 11, 12};
  const vector<size_t> expectedSecondModelNeuronIds{100, 101, 102};

  EXPECT_EQ(result.firstModelNeuronIds, expectedFirstModelNeuronIds);
  EXPECT_EQ(result.secondModelNeuronIds, expectedSecondModelNeuronIds);

  ASSERT_EQ(result.firstModelSynapses.size(), 2);
  ASSERT_EQ(result.secondModelSynapses.size(), 2);

  // Entries at equal positions describe the same depth-based connection.
  EXPECT_EQ(result.firstModelSynapses[0].id, 1);
  EXPECT_EQ(result.secondModelSynapses[0].id, 20);
  EXPECT_EQ(result.firstModelSynapses[1].id, 2);
  EXPECT_EQ(result.secondModelSynapses[1].id, 21);

  EXPECT_NE(result.firstModelSynapses[0].weight,
            result.secondModelSynapses[0].weight);
  EXPECT_NE(result.firstModelSynapses[0].isActive,
            result.secondModelSynapses[0].isActive);
}

/******************************************************************************
 * @brief Tests finding model elements not included in its similar elements
 ******************************************************************************/
TEST(DTModelTest, findNonSimilarNeuronsAndSynapsesTest)
{
  auto createNeuron = [](size_t id, size_t depthId)
  {
    Neuron neuron(id, NeuronTypeE::HIDDEN_NEURON, ActivationE::NO_ACTIVATION);
    neuron.depthId = depthId;
    return neuron;
  };

  DTModel model;
  model.neurons = {
      createNeuron(10, 0),
      createNeuron(11, 1000),
      createNeuron(12, 2000)};
  model.indexMap = {{10, 0}, {11, 1}, {12, 2}};

  model.neurons[0].addOutSynapse(Synapse(1, 10, 11, 0.1));
  model.neurons[1].addOutSynapse(Synapse(2, 11, 12, 0.2));
  model.neurons[0].addOutSynapse(Synapse(3, 10, 12, 0.3));

  const vector<size_t> similarNeuronIds{10, 12};
  const vector<Synapse> similarSynapses{
      model.neurons[0].outSynapses[0],
      model.neurons[1].outSynapses[0]};

  const NonSimilarDTModelElements result =
      DTMGeneticAlgorithm::findNonSimilarNeuronsAndSynapses(
          model, similarNeuronIds, similarSynapses);

  const vector<size_t> expectedNeuronIds{11};

  EXPECT_EQ(result.neuronIds, expectedNeuronIds);
  ASSERT_EQ(result.synapses.size(), 1);
  EXPECT_EQ(result.synapses[0].id, 3);
  EXPECT_EQ(result.synapses[0].inNeuronId, 10);
  EXPECT_EQ(result.synapses[0].outNeuronId, 12);
}

/******************************************************************************
 * @brief Tests unique hidden-neuron IDs without consuming fixed input/output IDs
 ******************************************************************************/
TEST(DTModelTest, getNewUniqueNeuronIdTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 2;
  parameters.outputSize = 1;
  DTMGeneticAlgorithm algorithm(parameters);

  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 3u);
  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 4u);
  EXPECT_EQ(algorithm.getNewUniqueSynapseId(), 0u);
  EXPECT_EQ(algorithm.getNewUniqueNeuronId(), 5u);
}

/******************************************************************************
 * @brief Tests crossover inheritance, endpoint remapping and generation replacement
 ******************************************************************************/
TEST(DTModelTest, crossoverTest)
{
  auto createModel = [](bool first)
  {
    DTModel model(2, 1, ActivationE::RELU);
    const size_t hiddenId = first ? 10 : 20;
    const double weight = first ? 5.0 : 9.0;
    size_t synapseId = first ? 100 : 200;
    Neuron hidden(hiddenId, NeuronTypeE::HIDDEN_NEURON,
                  first ? ActivationE::RELU : ActivationE::SIGMOID);
    hidden.bias = first ? 1.0 : 2.0;
    const Synapse hiddenIn(synapseId++, 0, hiddenId, weight);
    const Synapse hiddenOut(synapseId++, hiddenId, 2, weight);
    model.addNeuron(hidden, hiddenIn, hiddenOut, false);
    model.setBias(2, first ? 3.0 : 4.0);

    // Each parent has an exclusive shortcut. Only the first has another
    // hidden neuron, at the same depth as the shared hidden neuron.
    if (first)
    {
      Neuron extra(11, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID);
      extra.bias = 7.0;
      const Synapse extraIn(synapseId++, 1, 11, weight);
      const Synapse extraOut(synapseId++, 11, 2, weight);
      model.addNeuron(extra, extraIn, extraOut, false);
    }
    model.addOutSynapse(Synapse(synapseId, first ? 0 : 1, 2, weight), false);

    for (Neuron &neuron : model.neurons)
    {
      for (Synapse &synapse : neuron.outSynapses)
      {
        synapse.isActive = first;
      }
    }
    model.sortTopologically();
    return model;
  };

  const DTModel firstModel = createModel(true);
  const DTModel secondModel = createModel(false);
  const vector<pair<double, double>> fitnessCases{{10.0, 1.0}, {1.0, 10.0}, {5.0, 5.0}};

  // A fixed seed makes the coverage repeatable without depending on an exact
  // random sequence. Check inheritance choices, not a statistical 50% quota.
  srand(271828);
  for (const auto &fitness : fitnessCases)
  {
    SCOPED_TRACE(testing::Message() << "fitness: " << fitness.first << ", " << fitness.second);
    Hyperparameters parameters{};
    parameters.inputSize = 2;
    parameters.outputSize = 1;
    parameters.populationSize = 2;
    DTMGeneticAlgorithm algorithm(parameters);
    algorithm.currentGeneration = 7;
    algorithm.population.emplace_back(40, 1, firstModel);
    algorithm.population.emplace_back(50, 1, secondModel);
    algorithm.population[0].fitness = fitness.first;
    algorithm.population[1].fitness = fitness.second;

    // Force the crossover to prepare parents whose vector indexes differ
    // from their ordinary IDs and whose sorted flags are stale.
    for (DTIndividual &parent : algorithm.population)
    {
      std::reverse(parent.model.neurons.begin(), parent.model.neurons.end());
      for (size_t index = 0; index < parent.model.neurons.size(); index++)
      {
        parent.model.indexMap[parent.model.neurons[index].id] = index;
      }
      parent.model.isSorted = false;
    }

    GAParents forward;
    forward.addParent(0);
    forward.addParent(1);
    GAParents reverse;
    reverse.addParent(1);
    reverse.addParent(0);
    const size_t pairsPerSpecies = 32;
    const vector<vector<GAParents>> parentsLists{
        vector<GAParents>(pairsPerSpecies, forward), {}, vector<GAParents>(pairsPerSpecies, reverse)};

    algorithm.crossover(parentsLists);

    ASSERT_EQ(algorithm.population.size(), 2 * pairsPerSpecies);
    EXPECT_EQ(algorithm.hyperparameters.populationSize, algorithm.population.size());
    EXPECT_EQ(algorithm.currentGeneration, 7u);
    std::set<size_t> individualIds;
    std::set<size_t> hiddenIds;
    std::set<size_t> synapseIds;
    bool sawMixedNeuronAttributes = false;
    bool sawMixedSynapseAttributes = false;
    bool sawFirstOutputBias = false;
    bool sawSecondOutputBias = false;

    for (size_t childIndex = 0; childIndex < algorithm.population.size(); childIndex++)
    {
      DTIndividual &child = algorithm.population[childIndex];
      DTModel &model = child.model;
      const bool firstDominates = fitness.first > fitness.second ||
          (fitness.first == fitness.second && childIndex < pairsPerSpecies);
      const DTModel &dominant = firstDominates ? firstModel : secondModel;

      EXPECT_GT(child.id, 50u);
      EXPECT_TRUE(individualIds.insert(child.id).second);
      EXPECT_EQ(child.generation, 7u);
      EXPECT_DOUBLE_EQ(child.fitness, 0.0);
      EXPECT_EQ(model.inputSize, 2u);
      EXPECT_EQ(model.outputSize, 1u);
      EXPECT_TRUE(model.isSorted);
      EXPECT_EQ(model.maxDepth, 2u);
      ASSERT_EQ(model.neurons.size(), dominant.neurons.size());
      ASSERT_EQ(model.indexMap.size(), model.neurons.size());
      EXPECT_TRUE(model.validateModel());

      // A topology keyed by endpoint depthIds checks exact inherited edges
      // while allowing every hidden-neuron and synapse ID to be regenerated.
      std::set<pair<size_t, size_t>> expectedConnections;
      for (const Neuron &neuron : dominant.neurons)
      {
        for (const Synapse &synapse : neuron.outSynapses)
        {
          expectedConnections.emplace(
              dominant.neurons[dominant.indexMap.at(synapse.inNeuronId)].depthId,
              dominant.neurons[dominant.indexMap.at(synapse.outNeuronId)].depthId);
        }
      }
      std::set<pair<size_t, size_t>> actualConnections;
      size_t connectionCount = 0;
      size_t incomingCount = 0;

      for (size_t index = 0; index < model.neurons.size(); index++)
      {
        const Neuron &neuron = model.neurons[index];
        EXPECT_EQ(model.indexMap.at(neuron.id), index);
        EXPECT_DOUBLE_EQ(neuron.value, 0.0);
        incomingCount += neuron.inSynapses.size();

        if (neuron.type == NeuronTypeE::INPUT_NEURON)
        {
          EXPECT_LT(neuron.id, 2u);
          EXPECT_EQ(neuron.activation, ActivationE::NO_ACTIVATION);
        }
        else if (neuron.type == NeuronTypeE::OUTPUT_NEURON)
        {
          EXPECT_EQ(neuron.id, 2u);
          EXPECT_EQ(neuron.activation, ActivationE::RELU);
          EXPECT_TRUE(neuron.bias == 3.0 || neuron.bias == 4.0);
          sawFirstOutputBias |= neuron.bias == 3.0;
          sawSecondOutputBias |= neuron.bias == 4.0;
        }
        else
        {
          EXPECT_GT(neuron.id, 20u);
          EXPECT_TRUE(hiddenIds.insert(neuron.id).second);
          if (neuron.depthId == 1000)
          {
            EXPECT_TRUE(neuron.bias == 1.0 || neuron.bias == 2.0);
            EXPECT_TRUE(neuron.activation == ActivationE::RELU ||
                        neuron.activation == ActivationE::SIGMOID);
            sawMixedNeuronAttributes |= (neuron.bias == 1.0) !=
                                        (neuron.activation == ActivationE::RELU);
          }
          else
          {
            EXPECT_TRUE(firstDominates);
            EXPECT_EQ(neuron.depthId, 1001u);
            EXPECT_DOUBLE_EQ(neuron.bias, 7.0);
            EXPECT_EQ(neuron.activation, ActivationE::SIGMOID);
          }
        }

        for (const Synapse &synapse : neuron.outSynapses)
        {
          connectionCount++;
          EXPECT_GT(synapse.id, 202u);
          EXPECT_TRUE(synapseIds.insert(synapse.id).second);
          EXPECT_EQ(synapse.inNeuronId, neuron.id);
          const size_t outIndex = model.indexMap.at(synapse.outNeuronId);
          EXPECT_LT(index, outIndex);
          const Neuron &outNeuron = model.neurons[outIndex];
          actualConnections.emplace(neuron.depthId, outNeuron.depthId);
          const bool shared = (neuron.depthId == 0 && outNeuron.depthId == 1000) ||
                              (neuron.depthId == 1000 && outNeuron.depthId == 2000);
          if (shared)
          {
            EXPECT_TRUE(synapse.weight == 5.0 || synapse.weight == 9.0);
            sawMixedSynapseAttributes |= (synapse.weight == 5.0) != synapse.isActive;
          }
          else
          {
            EXPECT_DOUBLE_EQ(synapse.weight, firstDominates ? 5.0 : 9.0);
            EXPECT_EQ(synapse.isActive, firstDominates);
          }

          const auto incoming = std::find_if(outNeuron.inSynapses.begin(), outNeuron.inSynapses.end(),
              [&](const Synapse &mirror) { return mirror.id == synapse.id; });
          ASSERT_NE(incoming, outNeuron.inSynapses.end());
          EXPECT_EQ(incoming->inNeuronId, synapse.inNeuronId);
          EXPECT_EQ(incoming->outNeuronId, synapse.outNeuronId);
          EXPECT_DOUBLE_EQ(incoming->weight, synapse.weight);
          EXPECT_EQ(incoming->isActive, synapse.isActive);
        }
      }
      EXPECT_EQ(actualConnections, expectedConnections);
      EXPECT_EQ(connectionCount, expectedConnections.size());
      EXPECT_EQ(incomingCount, connectionCount);
    }

    EXPECT_TRUE(sawMixedNeuronAttributes);
    EXPECT_TRUE(sawMixedSynapseAttributes);
    EXPECT_TRUE(sawFirstOutputBias);
    EXPECT_TRUE(sawSecondOutputBias);
    EXPECT_GT(algorithm.getNewUniqueNeuronId(), *hiddenIds.rbegin());
    EXPECT_GT(algorithm.getNewUniqueSynapseId(), *synapseIds.rbegin());
    EXPECT_GT(algorithm.getNewUniqueIndividualCounter(), *individualIds.rbegin());
  }
}

/******************************************************************************
 * @brief Tests self-crossover, repeated generations and empty parent lists
 ******************************************************************************/
TEST(DTModelTest, crossoverSameParentAndEmptyListsTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::NO_ACTIVATION);
  Neuron hidden(10, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU);
  hidden.bias = 1.0;
  model.addNeuron(hidden, Synapse(100, 0, 10, 2.0), Synapse(101, 10, 1, 3.0), false);
  model.setBias(1, 4.0);
  algorithm.population.emplace_back(30, 0, std::move(model));
  algorithm.population[0].fitness = 10.0;
  GAParents parents;
  parents.addParent(0);
  parents.addParent(0);

  size_t previousIndividualId = 30;
  size_t previousHiddenId = 10;
  size_t previousSynapseId = 101;
  for (size_t generation = 1; generation <= 2; generation++)
  {
    algorithm.currentGeneration = generation;
    algorithm.crossover({{}, {parents}});
    ASSERT_EQ(algorithm.population.size(), 1u);
    DTIndividual &child = algorithm.population[0];
    EXPECT_GT(child.id, previousIndividualId);
    EXPECT_EQ(child.generation, generation);
    EXPECT_TRUE(child.model.isSorted);
    EXPECT_EQ(child.model.maxDepth, 2u);
    ASSERT_EQ(child.model.neurons.size(), 3u);
    EXPECT_EQ(child.model.neurons[1].type, NeuronTypeE::HIDDEN_NEURON);
    EXPECT_GT(child.model.neurons[1].id, previousHiddenId);
    EXPECT_DOUBLE_EQ(child.model.neurons[1].bias, 1.0);
    EXPECT_EQ(child.model.neurons[1].activation, ActivationE::RELU);
    ASSERT_EQ(child.model.neurons[0].outSynapses.size(), 1u);
    ASSERT_EQ(child.model.neurons[1].outSynapses.size(), 1u);
    EXPECT_GT(child.model.neurons[0].outSynapses[0].id, previousSynapseId);
    EXPECT_GT(child.model.neurons[1].outSynapses[0].id, previousSynapseId);
    EXPECT_DOUBLE_EQ(child.model.neurons[0].outSynapses[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(child.model.neurons[1].outSynapses[0].weight, 3.0);
    const vector<double> output = child.model.feedForward({2.0});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_DOUBLE_EQ(output[0], 5.0);

    previousIndividualId = child.id;
    previousHiddenId = child.model.neurons[1].id;
    previousSynapseId = std::max(child.model.neurons[0].outSynapses[0].id,
                                 child.model.neurons[1].outSynapses[0].id);
  }

  algorithm.crossover({});
  EXPECT_TRUE(algorithm.population.empty());
  EXPECT_EQ(algorithm.hyperparameters.populationSize, 0u);
}

namespace DTMutationTestUtils
{

// Compare every stored field so a rejected mutation cannot silently reorder
// neurons, alter metadata, or leave an incoming-synapse mirror changed.
void expectSameSynapses(const vector<Synapse> &actual, const vector<Synapse> &expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); index++)
  {
    EXPECT_EQ(actual[index].id, expected[index].id);
    EXPECT_EQ(actual[index].inNeuronId, expected[index].inNeuronId);
    EXPECT_EQ(actual[index].outNeuronId, expected[index].outNeuronId);
    EXPECT_DOUBLE_EQ(actual[index].weight, expected[index].weight);
    EXPECT_EQ(actual[index].isActive, expected[index].isActive);
  }
}

void expectSameIndividual(const DTIndividual &actual, const DTIndividual &expected)
{
  EXPECT_EQ(actual.id, expected.id);
  EXPECT_EQ(actual.generation, expected.generation);
  EXPECT_DOUBLE_EQ(actual.fitness, expected.fitness);
  EXPECT_EQ(actual.gracePeriodLength, expected.gracePeriodLength);
  EXPECT_EQ(actual.model.inputSize, expected.model.inputSize);
  EXPECT_EQ(actual.model.outputSize, expected.model.outputSize);
  EXPECT_EQ(actual.model.maxDepth, expected.model.maxDepth);
  EXPECT_EQ(actual.model.indexMap, expected.model.indexMap);
  EXPECT_EQ(actual.model.isSorted, expected.model.isSorted);
  ASSERT_EQ(actual.model.neurons.size(), expected.model.neurons.size());
  for (size_t index = 0; index < actual.model.neurons.size(); index++)
  {
    const Neuron &neuron = actual.model.neurons[index];
    const Neuron &previous = expected.model.neurons[index];
    EXPECT_EQ(neuron.id, previous.id);
    EXPECT_EQ(neuron.depthId, previous.depthId);
    EXPECT_EQ(neuron.depth, previous.depth);
    EXPECT_EQ(neuron.type, previous.type);
    EXPECT_EQ(neuron.activation, previous.activation);
    EXPECT_DOUBLE_EQ(neuron.value, previous.value);
    EXPECT_DOUBLE_EQ(neuron.bias, previous.bias);
    expectSameSynapses(neuron.inSynapses, previous.inSynapses);
    expectSameSynapses(neuron.outSynapses, previous.outSynapses);
  }
}

size_t synapseCount(const DTModel &model)
{
  size_t count = 0;
  for (const Neuron &neuron : model.neurons)
  {
    count += neuron.outSynapses.size();
  }
  return count;
}

void expectValidModel(DTModel &model)
{
  EXPECT_TRUE(model.isSorted);
  EXPECT_TRUE(model.validateModel());
  ASSERT_EQ(model.indexMap.size(), model.neurons.size());
  std::set<size_t> synapseIds;
  std::set<pair<size_t, size_t>> endpoints;
  size_t maximumDepth = 0;
  for (size_t index = 0; index < model.neurons.size(); index++)
  {
    const Neuron &neuron = model.neurons[index];
    EXPECT_EQ(model.indexMap.at(neuron.id), index);
    maximumDepth = std::max(maximumDepth, neuron.depth);
    for (const Synapse &synapse : neuron.outSynapses)
    {
      EXPECT_TRUE(synapseIds.insert(synapse.id).second);
      EXPECT_TRUE(endpoints.emplace(synapse.inNeuronId, synapse.outNeuronId).second);
      EXPECT_EQ(synapse.inNeuronId, neuron.id);
      ASSERT_TRUE(model.indexMap.contains(synapse.outNeuronId));
      const size_t targetIndex = model.indexMap.at(synapse.outNeuronId);
      EXPECT_LT(index, targetIndex);
      EXPECT_LT(neuron.depth, model.neurons[targetIndex].depth);
      EXPECT_NE(neuron.type, NeuronTypeE::OUTPUT_NEURON);
      EXPECT_NE(model.neurons[targetIndex].type, NeuronTypeE::INPUT_NEURON);
    }
    // Existing models may omit incoming mirrors, but any mirror that is
    // present must still correspond to an actual outgoing connection.
    for (const Synapse &mirror : neuron.inSynapses)
    {
      EXPECT_EQ(mirror.outNeuronId, neuron.id);
      ASSERT_TRUE(model.indexMap.contains(mirror.inNeuronId));
      const auto &outgoing = model.neurons[model.indexMap.at(mirror.inNeuronId)].outSynapses;
      const auto match = std::find_if(outgoing.begin(), outgoing.end(),
          [&](const Synapse &synapse) { return synapse.id == mirror.id; });
      ASSERT_NE(match, outgoing.end());
      EXPECT_EQ(match->outNeuronId, mirror.outNeuronId);
      EXPECT_DOUBLE_EQ(match->weight, mirror.weight);
      EXPECT_EQ(match->isActive, mirror.isActive);
    }
  }
  EXPECT_EQ(model.maxDepth, maximumDepth);
}

}

/******************************************************************************
 * @brief Tests directed synapse lookup, including disabled and missing edges
 ******************************************************************************/
TEST(DTModelTest, hasSynapseTest)
{
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), true);
  model.neurons[model.indexMap.at(0)].outSynapses[0].isActive = false;
  const auto originalIndexMap = model.indexMap;
  const DTModel &readOnlyModel = model;

  EXPECT_TRUE(readOnlyModel.hasSynapse(0, 40));
  EXPECT_TRUE(readOnlyModel.hasSynapse(40, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(40, 0));
  EXPECT_FALSE(readOnlyModel.hasSynapse(0, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(40, 40));
  EXPECT_FALSE(readOnlyModel.hasSynapse(999, 1));
  EXPECT_FALSE(readOnlyModel.hasSynapse(0, 999));
  EXPECT_EQ(model.indexMap, originalIndexMap);
}

/******************************************************************************
 * @brief Tests neuron insertion, new gene attributes, ID uniqueness and limits
 ******************************************************************************/
TEST(DTModelTest, addNeuronMutationTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.maxNumberOfNeurons = 4;
  parameters.gracePeriodLength = 9;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(100, 0, 40, 2.5), Synapse(101, 40, 1, 3.5), true);
  DTIndividual individual(7, 3, std::move(model));
  individual.fitness = 4.5;
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;

  srand(12345);
  ASSERT_TRUE(algorithm.addNeuronMutation(individual));
  EXPECT_EQ(individual.id, before.id);
  EXPECT_EQ(individual.generation, before.generation);
  EXPECT_DOUBLE_EQ(individual.fitness, before.fitness);
  EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
  ASSERT_EQ(individual.model.neurons.size(), 4u);
  EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 4u);
  DTMutationTestUtils::expectValidModel(individual.model);

  size_t addedNeuronCount = 0;
  for (const Neuron &neuron : individual.model.neurons)
  {
    if (before.model.indexMap.contains(neuron.id))
    {
      const Neuron &previous = before.model.neurons[before.model.indexMap.at(neuron.id)];
      EXPECT_EQ(neuron.type, previous.type);
      EXPECT_EQ(neuron.activation, previous.activation);
      EXPECT_DOUBLE_EQ(neuron.bias, previous.bias);
      continue;
    }
    addedNeuronCount++;
    EXPECT_GT(neuron.id, 40u);
    EXPECT_EQ(neuron.type, NeuronTypeE::HIDDEN_NEURON);
    EXPECT_EQ(neuron.activation, ActivationE::SIGMOID);
    EXPECT_GE(neuron.bias, 0.0);
    EXPECT_LE(neuron.bias, 1.0);
    ASSERT_EQ(neuron.inSynapses.size(), 1u);
    ASSERT_EQ(neuron.outSynapses.size(), 1u);
    const Synapse &incoming = neuron.inSynapses[0];
    const Synapse &outgoing = neuron.outSynapses[0];
    EXPECT_EQ(incoming.outNeuronId, neuron.id);
    EXPECT_EQ(outgoing.inNeuronId, neuron.id);
    EXPECT_GT(incoming.id, 101u);
    EXPECT_GT(outgoing.id, 101u);
    EXPECT_NE(incoming.id, outgoing.id);
    EXPECT_TRUE(incoming.isActive);
    EXPECT_TRUE(outgoing.isActive);
    EXPECT_GE(incoming.weight, 0.0);
    EXPECT_LE(incoming.weight, 1.0);
    EXPECT_GE(outgoing.weight, 0.0);
    EXPECT_LE(outgoing.weight, 1.0);
    EXPECT_LT(before.model.indexMap.at(incoming.inNeuronId),
              before.model.indexMap.at(outgoing.outNeuronId));
  }
  EXPECT_EQ(addedNeuronCount, 1u);
  EXPECT_TRUE(individual.model.hasSynapse(0, 40));
  EXPECT_TRUE(individual.model.hasSynapse(40, 1));

  individual.gracePeriodLength = 2;
  const DTIndividual atLimit = individual;
  EXPECT_FALSE(algorithm.addNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, atLimit);
  algorithm.hyperparameters.maxNumberOfNeurons = 3;
  EXPECT_FALSE(algorithm.addNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, atLimit);
}

/******************************************************************************
 * @brief Tests removing only hidden neurons and cleaning both adjacency lists
 ******************************************************************************/
TEST(DTModelTest, removeNeuronMutationTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 8;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 1, 70, 0.4), Synapse(103, 70, 2, 0.5), true);
  DTIndividual individual(7, 3, std::move(model));

  for (size_t remainingHidden = 2; remainingHidden != 0; remainingHidden--)
  {
    const DTIndividual before = individual;
    ASSERT_TRUE(algorithm.removeNeuronMutation(individual));
    EXPECT_EQ(individual.model.neurons.size(), before.model.neurons.size() - 1);
    EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
    EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 2 * (remainingHidden - 1));
    DTMutationTestUtils::expectValidModel(individual.model);
    for (const Neuron &neuron : before.model.neurons)
    {
      if (!individual.model.indexMap.contains(neuron.id))
      {
        EXPECT_EQ(neuron.type, NeuronTypeE::HIDDEN_NEURON);
      }
      else
      {
        const Neuron &remaining = individual.model.neurons[individual.model.indexMap.at(neuron.id)];
        EXPECT_DOUBLE_EQ(remaining.bias, neuron.bias);
        EXPECT_EQ(remaining.activation, neuron.activation);
      }
    }
  }

  individual.gracePeriodLength = 2;
  const DTIndividual beforeFailure = individual;
  EXPECT_FALSE(algorithm.removeNeuronMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeFailure);
}

/******************************************************************************
 * @brief Tests rejecting either removal from a chain of dependent hidden neurons
 ******************************************************************************/
TEST(DTModelTest, removeNeuronMutationRejectsDanglingTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 8;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 40, 70, 0.4), Synapse(103, 70, 1, 0.5), false);
  model.removeSynapse(40, 1, true);
  DTIndividual individual(7, 3, std::move(model));
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;

  for (unsigned seed = 0; seed < 16; seed++)
  {
    srand(seed);
    EXPECT_FALSE(algorithm.removeNeuronMutation(individual));
    DTMutationTestUtils::expectSameIndividual(individual, before);
  }
}

/******************************************************************************
 * @brief Tests adding a connection and rejecting duplicates, including inactive ones
 ******************************************************************************/
TEST(DTModelTest, addSynapseMutationTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.gracePeriodLength = 6;
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual individual(7, 3, DTModel(1, 1, ActivationE::RELU));

  ASSERT_TRUE(algorithm.addSynapseMutation(individual));
  EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
  ASSERT_EQ(individual.model.neurons.size(), 2u);
  ASSERT_EQ(individual.model.neurons[0].outSynapses.size(), 1u);
  const Synapse &synapse = individual.model.neurons[0].outSynapses[0];
  EXPECT_EQ(synapse.inNeuronId, 0u);
  EXPECT_EQ(synapse.outNeuronId, 1u);
  EXPECT_TRUE(synapse.isActive);
  EXPECT_GE(synapse.weight, 0.0);
  EXPECT_LE(synapse.weight, 1.0);
  DTMutationTestUtils::expectSameSynapses(individual.model.neurons[1].inSynapses,
                                       individual.model.neurons[0].outSynapses);
  DTMutationTestUtils::expectValidModel(individual.model);

  individual.gracePeriodLength = 2;
  const DTIndividual beforeDuplicate = individual;
  EXPECT_FALSE(algorithm.addSynapseMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeDuplicate);

  individual.model.neurons[0].outSynapses[0].isActive = false;
  individual.model.neurons[1].inSynapses[0].isActive = false;
  const DTIndividual beforeInactiveDuplicate = individual;
  EXPECT_FALSE(algorithm.addSynapseMutation(individual));
  DTMutationTestUtils::expectSameIndividual(individual, beforeInactiveDuplicate);
}

/******************************************************************************
 * @brief Tests forward endpoint selection with hidden neurons and one-attempt failures
 ******************************************************************************/
TEST(DTModelTest, addSynapseMutationTopologyTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 6;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 2, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  model.addNeuron(Neuron(70, NeuronTypeE::HIDDEN_NEURON, ActivationE::RELU),
                  Synapse(102, 1, 70, 0.4), Synapse(103, 70, 3, 0.5), true);
  const DTIndividual original(7, 3, std::move(model));
  size_t successCount = 0;
  size_t failureCount = 0;

  for (unsigned seed = 0; seed < 64; seed++)
  {
    DTIndividual individual = original;
    srand(seed);
    if (!algorithm.addSynapseMutation(individual))
    {
      failureCount++;
      DTMutationTestUtils::expectSameIndividual(individual, original);
      continue;
    }
    successCount++;
    EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
    EXPECT_EQ(individual.model.neurons.size(), original.model.neurons.size());
    EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 5u);
    DTMutationTestUtils::expectValidModel(individual.model);
    for (const Neuron &neuron : individual.model.neurons)
    {
      for (const Synapse &synapse : neuron.outSynapses)
      {
        if (!original.model.hasSynapse(synapse.inNeuronId, synapse.outNeuronId))
        {
          EXPECT_GT(synapse.id, 103u);
          EXPECT_TRUE(synapse.isActive);
          EXPECT_GE(synapse.weight, 0.0);
          EXPECT_LE(synapse.weight, 1.0);
          EXPECT_LT(original.model.indexMap.at(synapse.inNeuronId),
                    original.model.indexMap.at(synapse.outNeuronId));
        }
      }
    }
  }
  // A graph with both missing and existing pairs must produce rejected
  // attempts too: duplicate selection must not trigger another draw.
  EXPECT_GT(successCount, 0u);
  EXPECT_GT(failureCount, 0u);
}

/******************************************************************************
 * @brief Tests valid edge removal with both complete and absent incoming mirrors
 ******************************************************************************/
TEST(DTModelTest, removeSynapseMutationTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 5;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(2, 2, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 2, 0.3), false);
  const Synapse extraIn(102, 1, 40, 0.4);
  const Synapse extraOut(103, 40, 3, 0.5);
  model.addOutSynapse(extraIn, false);
  model.addOutSynapse(extraOut, false);
  model.neurons[model.indexMap.at(40)].addInSynapse(extraIn);
  model.neurons[model.indexMap.at(3)].addInSynapse(extraOut);
  model.sortTopologically();

  for (bool omitIncoming : {false, true})
  {
    for (unsigned seed = 0; seed < 16; seed++)
    {
      DTIndividual individual(7, 3, model);
      if (omitIncoming)
      {
        for (Neuron &neuron : individual.model.neurons)
        {
          neuron.inSynapses.clear();
        }
      }
      srand(seed);
      // Two inputs and two outputs at the hidden neuron make every single
      // edge removable, regardless of whether incoming mirrors are available.
      ASSERT_TRUE(algorithm.removeSynapseMutation(individual));
      EXPECT_EQ(individual.gracePeriodLength, parameters.gracePeriodLength);
      EXPECT_EQ(individual.model.neurons.size(), model.neurons.size());
      EXPECT_EQ(DTMutationTestUtils::synapseCount(individual.model), 3u);
      DTMutationTestUtils::expectValidModel(individual.model);
      for (const Neuron &neuron : individual.model.neurons)
      {
        for (const Synapse &synapse : neuron.outSynapses)
        {
          EXPECT_TRUE(model.hasSynapse(synapse.inNeuronId, synapse.outNeuronId));
        }
      }
    }
  }
}

/******************************************************************************
 * @brief Tests rejected dangling-edge removals without retrying an optional edge
 ******************************************************************************/
TEST(DTModelTest, removeSynapseMutationRejectsDanglingTest)
{
  Hyperparameters parameters{};
  parameters.gracePeriodLength = 5;
  DTMGeneticAlgorithm algorithm(parameters);
  DTModel model(1, 1, ActivationE::RELU);
  model.addNeuron(Neuron(40, NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID),
                  Synapse(100, 0, 40, 0.2), Synapse(101, 40, 1, 0.3), true);
  DTIndividual individual(7, 3, model);
  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;
  for (unsigned seed = 0; seed < 16; seed++)
  {
    srand(seed);
    EXPECT_FALSE(algorithm.removeSynapseMutation(individual));
    DTMutationTestUtils::expectSameIndividual(individual, before);
  }

  // Only the direct input/output shortcut can be removed. Selecting either
  // edge on the hidden-neuron path must fail even though the shortcut exists.
  model.addOutSynapse(Synapse(102, 0, 1, 0.4), true);
  size_t successCount = 0;
  size_t failureCount = 0;
  for (unsigned seed = 0; seed < 32; seed++)
  {
    DTIndividual attempt(7, 3, model);
    attempt.gracePeriodLength = 2;
    const DTIndividual original = attempt;
    srand(seed);
    if (algorithm.removeSynapseMutation(attempt))
    {
      successCount++;
      EXPECT_FALSE(attempt.model.hasSynapse(0, 1));
      EXPECT_TRUE(attempt.model.hasSynapse(0, 40));
      EXPECT_TRUE(attempt.model.hasSynapse(40, 1));
      DTMutationTestUtils::expectValidModel(attempt.model);
    }
    else
    {
      failureCount++;
      DTMutationTestUtils::expectSameIndividual(attempt, original);
    }
  }
  EXPECT_GT(successCount, 0u);
  EXPECT_GT(failureCount, 0u);

  DTIndividual empty(8, 3, DTModel(1, 1, ActivationE::RELU));
  empty.gracePeriodLength = 2;
  const DTIndividual beforeEmpty = empty;
  EXPECT_FALSE(algorithm.removeSynapseMutation(empty));
  DTMutationTestUtils::expectSameIndividual(empty, beforeEmpty);
}

/******************************************************************************
 * @brief Tests all dispatcher values and unchanged individuals on invalid requests
 ******************************************************************************/
TEST(DTModelTest, mutationDispatchTest)
{
  Hyperparameters parameters{};
  parameters.inputSize = 1;
  parameters.outputSize = 1;
  parameters.maxNumberOfNeurons = 3;
  // A successful mutation must reset the grace period even when it is zero.
  parameters.gracePeriodLength = 0;
  DTMGeneticAlgorithm algorithm(parameters);
  DTIndividual individual(7, 3, DTModel(1, 1, ActivationE::RELU));

  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADD_SYNAPSE));
  EXPECT_TRUE(individual.model.hasSynapse(0, 1));
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::REMOVE_SYNAPSE));
  EXPECT_FALSE(individual.model.hasSynapse(0, 1));
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::ADD_NEURON));
  EXPECT_EQ(individual.model.neurons.size(), 3u);
  EXPECT_EQ(individual.gracePeriodLength, 0u);
  individual.gracePeriodLength = 2;
  ASSERT_TRUE(algorithm.mutate(individual, MutationE::REMOVE_NEURON));
  EXPECT_EQ(individual.model.neurons.size(), 2u);
  EXPECT_EQ(individual.gracePeriodLength, 0u);

  individual.gracePeriodLength = 2;
  const DTIndividual before = individual;
  EXPECT_FALSE(algorithm.mutate(individual, static_cast<MutationE>(999)));
  DTMutationTestUtils::expectSameIndividual(individual, before);
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::REMOVE_NEURON));
  DTMutationTestUtils::expectSameIndividual(individual, before);

  // Addition requires sorted input and must not sort as a side effect of a
  // rejected call. The caller can sort explicitly before its next attempt.
  individual.model.isSorted = false;
  const DTIndividual unsorted = individual;
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::ADD_NEURON));
  DTMutationTestUtils::expectSameIndividual(individual, unsorted);
  EXPECT_FALSE(algorithm.mutate(individual, MutationE::ADD_SYNAPSE));
  DTMutationTestUtils::expectSameIndividual(individual, unsorted);
}
