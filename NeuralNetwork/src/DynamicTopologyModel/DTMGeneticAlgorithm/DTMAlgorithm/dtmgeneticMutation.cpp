#include "dtmgeneticAlgorithm.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace
{

/******************************************************************************
 * @brief Selects one forward pair of neuron indexes in a sorted model
 *
 * @param model       Model whose inputs precede hidden neurons and outputs
 * @param sourceIndex Receives the selected input/hidden neuron's position
 * @param targetIndex Receives the selected later hidden/output neuron's position
 *
 * @return False if the model is unsorted or has no eligible endpoint pair
 ******************************************************************************/
bool selectForwardNeuronPair(const DTModel &model, size_t &sourceIndex, size_t &targetIndex)
{
  if (!model.isSorted || model.inputSize == 0 || model.outputSize == 0 ||
      model.neurons.size() < model.inputSize + model.outputSize)
  {
    return false;
  }

  // sortTopologically places inputs first and outputs last. Selecting from
  // these ranges excludes output sources, input targets and backward edges
  // without retries or constructing a vector of every possible connection.
  sourceIndex = rand() % (model.neurons.size() - model.outputSize);
  const size_t firstTargetIndex = std::max(model.inputSize, sourceIndex + 1);
  targetIndex = firstTargetIndex + rand() % (model.neurons.size() - firstTargetIndex);

  const NeuronTypeE sourceType = model.neurons[sourceIndex].type;
  const NeuronTypeE targetType = model.neurons[targetIndex].type;
  return (sourceType == NeuronTypeE::INPUT_NEURON || sourceType == NeuronTypeE::HIDDEN_NEURON) &&
         (targetType == NeuronTypeE::HIDDEN_NEURON || targetType == NeuronTypeE::OUTPUT_NEURON);
}

/******************************************************************************
 * @brief Checks hidden-neuron connectivity after a proposed removal
 *
 * @param model          Valid acyclic model to inspect without modifying it
 * @param removedNeuron  Neuron to omit, or nullptr when removing a synapse
 * @param removedSynapse Synapse to omit, or nullptr when removing a neuron
 *
 * @return True if every remaining hidden neuron retains input and output edges
 ******************************************************************************/
bool preservesHiddenConnections(const DTModel &model, const Neuron *removedNeuron,
                               const Synapse *removedSynapse)
{
  vector<bool> hasInput(model.neurons.size(), false);
  for (const Neuron &neuron : model.neurons)
  {
    if (&neuron == removedNeuron)
    {
      continue;
    }

    bool hasOutput = false;
    for (const Synapse &synapse : neuron.outSynapses)
    {
      if (&synapse == removedSynapse ||
          (removedNeuron != nullptr && synapse.outNeuronId == removedNeuron->id))
      {
        continue;
      }
      hasOutput = true;
      hasInput[model.indexMap.at(synapse.outNeuronId)] = true;
    }
    if (neuron.type == NeuronTypeE::HIDDEN_NEURON && !hasOutput)
    {
      return false;
    }
  }

  // For a valid DAG, retaining an input and output edge at every hidden
  // neuron also retains paths from inputs to outputs. Inspect outgoing lists
  // because inSynapses is not populated by every model-building operation.
  // As in validateModel(), inactive connections count structurally.
  for (size_t index = 0; index < model.neurons.size(); index++)
  {
    const Neuron &neuron = model.neurons[index];
    if (&neuron != removedNeuron && neuron.type == NeuronTypeE::HIDDEN_NEURON &&
        !hasInput[index])
    {
      return false;
    }
  }
  return true;
}

}

/******************************************************************************
 * @brief Advances gene counters beyond IDs already present in the target model
 *
 * Also supports individuals built outside this algorithm. Counters are retained
 * across calls; scanning by reference avoids copying neurons or synapse lists.
 * Called only after a mutation has passed its feasibility checks.
 *
 * @param model Model that will receive new neurons or synapses
 ******************************************************************************/
void DTMGeneticAlgorithm::synchronizeMutationIds(const DTModel &model)
{
  for (const Neuron &neuron : model.neurons)
  {
    uniqueNeuronIdCounter = std::max(uniqueNeuronIdCounter, neuron.id + 1);
    for (const Synapse &synapse : neuron.outSynapses)
    {
      uniqueSynapseIdCounter = std::max(uniqueSynapseIdCounter, synapse.id + 1);
    }
  }
}

/******************************************************************************
 * @brief Adds one hidden neuron between a randomly selected forward pair
 *
 * Requires a sorted model below maxNumberOfNeurons (including input/output
 * neurons). Creates fresh IDs, a sigmoid hidden neuron with random bias, and
 * two active synapses with independently random weights. Existing connections
 * remain present. Sorting refreshes indexes, depths and depthIds afterwards.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after adding the neuron and resetting its owner's grace period;
 *         false without changing the individual when the attempt is impossible
 ******************************************************************************/
bool DTMGeneticAlgorithm::addNeuronMutation(DTIndividual &individual)
{
  DTModel &model = individual.model;
  if (model.neurons.size() >= hyperparameters.maxNumberOfNeurons)
  {
    return false;
  }

  size_t sourceIndex = 0;
  size_t targetIndex = 0;
  if (!selectForwardNeuronPair(model, sourceIndex, targetIndex))
  {
    return false;
  }

  synchronizeMutationIds(model);
  Neuron neuron(getNewUniqueNeuronId(), NeuronTypeE::HIDDEN_NEURON, ActivationE::SIGMOID);
  const Synapse inSynapse(getNewUniqueSynapseId(), model.neurons[sourceIndex].id,
                         neuron.id, DTMUtils::randomdouble());
  const Synapse outSynapse(getNewUniqueSynapseId(), neuron.id,
                          model.neurons[targetIndex].id, DTMUtils::randomdouble());
  model.addNeuron(neuron, inSynapse, outSynapse, true);
  individual.gracePeriodLength = hyperparameters.gracePeriodLength;
  return true;
}

/******************************************************************************
 * @brief Removes one randomly selected hidden neuron and its incident synapses
 *
 * Makes one selection among hidden neurons. Rejects the removal if another
 * hidden neuron would lose all input or output edges; there is no retry or
 * cascading deletion of other neurons. A successful removal sorts the model.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after removal and grace-period reset; false with no change when
 *         no hidden neuron exists or the selected removal would be invalid
 ******************************************************************************/
bool DTMGeneticAlgorithm::removeNeuronMutation(DTIndividual &individual)
{
  DTModel &model = individual.model;
  if (model.neurons.size() <= model.inputSize + model.outputSize)
  {
    return false;
  }

  const size_t hiddenCount = std::count_if(model.neurons.begin(), model.neurons.end(),
      [](const Neuron &neuron) { return neuron.type == NeuronTypeE::HIDDEN_NEURON; });
  if (hiddenCount == 0)
  {
    return false;
  }

  size_t selectedIndex = rand() % hiddenCount;
  for (const Neuron &neuron : model.neurons)
  {
    if (neuron.type != NeuronTypeE::HIDDEN_NEURON)
    {
      continue;
    }
    if (selectedIndex != 0)
    {
      selectedIndex--;
      continue;
    }
    if (!preservesHiddenConnections(model, &neuron, nullptr))
    {
      return false;
    }

    // Copy the ID before removal invalidates references into the neuron list.
    const size_t neuronId = neuron.id;
    model.removeNeuron(neuronId, true);
    individual.gracePeriodLength = hyperparameters.gracePeriodLength;
    return true;
  }
  return false;
}

/******************************************************************************
 * @brief Attempts to add one random forward synapse to a sorted model
 *
 * Selects an input/hidden source and a later hidden/output target. If that
 * directed connection already exists (even inactive), returns immediately.
 * Otherwise adds a fresh active synapse with random weight to both endpoint
 * lists, then refreshes the model's topological metadata.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after addition and grace-period reset; false without changing
 *         the individual if no eligible pair exists or the pair is a duplicate
 ******************************************************************************/
bool DTMGeneticAlgorithm::addSynapseMutation(DTIndividual &individual)
{
  DTModel &model = individual.model;
  size_t sourceIndex = 0;
  size_t targetIndex = 0;
  if (!selectForwardNeuronPair(model, sourceIndex, targetIndex))
  {
    return false;
  }

  const size_t sourceId = model.neurons[sourceIndex].id;
  const size_t targetId = model.neurons[targetIndex].id;
  if (model.hasSynapse(sourceId, targetId))
  {
    return false;
  }

  synchronizeMutationIds(model);
  const Synapse synapse(getNewUniqueSynapseId(), sourceId, targetId, DTMUtils::randomdouble());
  model.neurons[sourceIndex].addOutSynapse(synapse);
  model.neurons[targetIndex].addInSynapse(synapse);
  model.isSorted = false;
  model.sortTopologically();
  individual.gracePeriodLength = hyperparameters.gracePeriodLength;
  return true;
}

/******************************************************************************
 * @brief Attempts to remove one random synapse while preserving model validity
 *
 * Counts outgoing synapses once, then selects a single connection across all
 * neurons without copying their lists. Checks the proposed removal before
 * modifying the model, so a rejected attempt needs neither rollback nor retry.
 * Hidden neurons must retain both input and output connections.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after removal, sorting and grace-period reset; false with no
 *         change if no synapse exists or the selected removal would be invalid
 ******************************************************************************/
bool DTMGeneticAlgorithm::removeSynapseMutation(DTIndividual &individual)
{
  DTModel &model = individual.model;
  size_t synapseCount = 0;
  for (const Neuron &neuron : model.neurons)
  {
    synapseCount += neuron.outSynapses.size();
  }
  if (synapseCount == 0)
  {
    return false;
  }

  size_t selectedIndex = rand() % synapseCount;
  for (const Neuron &neuron : model.neurons)
  {
    if (selectedIndex >= neuron.outSynapses.size())
    {
      selectedIndex -= neuron.outSynapses.size();
      continue;
    }

    const Synapse &synapse = neuron.outSynapses[selectedIndex];
    if (!preservesHiddenConnections(model, nullptr, &synapse))
    {
      return false;
    }

    // Removal changes adjacency lists and sorting moves neurons. Retain only
    // endpoint IDs across those operations, not references into either list.
    const size_t sourceId = synapse.inNeuronId;
    const size_t targetId = synapse.outNeuronId;
    model.removeSynapse(sourceId, targetId, true);
    individual.gracePeriodLength = hyperparameters.gracePeriodLength;
    return true;
  }
  return false;
}

/******************************************************************************
 * @brief Perturbs one randomly selected synapse weight in either direction
 *
 * Adds a uniform change in [-weightMutationStrength, weightMutationStrength].
 * Weights may cross zero and grow beyond their initialization range. Updates
 * any incoming mirror of the selected synapse as well as its canonical outgoing
 * entry. Connection IDs, activity and topological metadata are preserved.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after a finite weight change and grace-period reset; false with
 *         no change if no synapse exists, strength is invalid, or rounding or
 *         overflow prevents a finite change. A failed draw is not retried.
 ******************************************************************************/
bool DTMGeneticAlgorithm::adjustWeightMutation(DTIndividual &individual)
{
  const double strength = hyperparameters.weightMutationStrength;
  if (!std::isfinite(strength) || strength <= 0.0)
  {
    return false;
  }

  DTModel &model = individual.model;
  size_t count = 0;
  for (const Neuron &neuron : model.neurons)
  {
    count += neuron.outSynapses.size();
  }
  if (count == 0)
  {
    return false;
  }
  size_t selected = rand() % count;
  for (Neuron &neuron : model.neurons)
  {
    if (selected >= neuron.outSynapses.size())
    {
      selected -= neuron.outSynapses.size();
      continue;
    }
    Synapse &synapse = neuron.outSynapses[selected];
    const double weight = synapse.weight + (2.0 * DTMUtils::randomdouble() - 1.0) * strength;
    if (!std::isfinite(weight) || weight == synapse.weight)
    {
      return false;
    }
    synapse.weight = weight;
    for (Synapse &mirror : model.neurons[model.indexMap.at(synapse.outNeuronId)].inSynapses)
    {
      if (mirror.id == synapse.id)
      {
        mirror.weight = weight;
      }
    }
    individual.gracePeriodLength = hyperparameters.gracePeriodLength;
    return true;
  }
  return false;
}

/******************************************************************************
 * @brief Perturbs one randomly selected hidden or output neuron's bias
 *
 * Adds a uniform change in [-biasMutationStrength, biasMutationStrength], with
 * no sign restriction. Input neurons are excluded. The model's topology and
 * activation functions are preserved.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after a finite bias change and grace-period reset; false without
 *         modification if no eligible neuron exists or no finite change is made
 ******************************************************************************/
bool DTMGeneticAlgorithm::adjustBiasMutation(DTIndividual &individual)
{
  const double strength = hyperparameters.biasMutationStrength;
  if (!std::isfinite(strength) || strength <= 0.0)
  {
    return false;
  }
  auto eligible = [](const Neuron &neuron)
  {
    return neuron.type == NeuronTypeE::HIDDEN_NEURON || neuron.type == NeuronTypeE::OUTPUT_NEURON;
  };
  auto &neurons = individual.model.neurons;
  const size_t count = std::count_if(neurons.begin(), neurons.end(), eligible);
  if (count == 0)
  {
    return false;
  }
  size_t selected = rand() % count;
  for (Neuron &neuron : neurons)
  {
    if (!eligible(neuron))
    {
      continue;
    }
    if (selected != 0)
    {
      selected--;
      continue;
    }
    const double bias = neuron.bias + (2.0 * DTMUtils::randomdouble() - 1.0) * strength;
    if (!std::isfinite(bias) || bias == neuron.bias)
    {
      return false;
    }
    neuron.bias = bias;
    individual.gracePeriodLength = hyperparameters.gracePeriodLength;
    return true;
  }
  return false;
}

/******************************************************************************
 * @brief Switches one hidden neuron to a different supported activation function
 *
 * Selects one hidden neuron and chooses uniformly between its other two
 * supported activations (sigmoid, ReLU, or no activation). Input and output
 * activations, numerical parameters and topology are preserved.
 *
 * @param individual Individual to mutate by reference
 *
 * @return True after switching activation and resetting the grace period;
 *         false without modification when there is no hidden neuron
 ******************************************************************************/
bool DTMGeneticAlgorithm::changeActivationMutation(DTIndividual &individual)
{
  auto &neurons = individual.model.neurons;
  const size_t count = std::count_if(neurons.begin(), neurons.end(),
      [](const Neuron &neuron) { return neuron.type == NeuronTypeE::HIDDEN_NEURON; });
  if (count == 0)
  {
    return false;
  }
  size_t selected = rand() % count;
  for (Neuron &neuron : neurons)
  {
    if (neuron.type != NeuronTypeE::HIDDEN_NEURON)
    {
      continue;
    }
    if (selected != 0)
    {
      selected--;
      continue;
    }
    size_t alternative = rand() % 2;
    for (ActivationE activation : {ActivationE::SIGMOID, ActivationE::RELU, ActivationE::NO_ACTIVATION})
    {
      if (activation == neuron.activation)
      {
        continue;
      }
      if (alternative != 0)
      {
        alternative--;
        continue;
      }
      neuron.activation = activation;
      individual.gracePeriodLength = hyperparameters.gracePeriodLength;
      return true;
    }
  }
  return false;
}

/******************************************************************************
 * @brief Dispatches one topology or parameter mutation for an individual
 *
 * @param individual Individual to mutate in place
 * @param mutation   Mutation to attempt, selected with MutationE
 *
 * @return The selected mutation's success flag, or false for an unknown value;
 *         only successful mutations reset the individual's grace period
 ******************************************************************************/
bool DTMGeneticAlgorithm::mutate(DTIndividual &individual, MutationE mutation)
{
  switch (mutation)
  {
  case MutationE::ADD_NEURON:
    return addNeuronMutation(individual);
  case MutationE::REMOVE_NEURON:
    return removeNeuronMutation(individual);
  case MutationE::ADD_SYNAPSE:
    return addSynapseMutation(individual);
  case MutationE::REMOVE_SYNAPSE:
    return removeSynapseMutation(individual);
  case MutationE::ADJUST_WEIGHT:
    return adjustWeightMutation(individual);
  case MutationE::ADJUST_BIAS:
    return adjustBiasMutation(individual);
  case MutationE::CHANGE_ACTIVATION:
    return changeActivationMutation(individual);
  }
  return false;
}
