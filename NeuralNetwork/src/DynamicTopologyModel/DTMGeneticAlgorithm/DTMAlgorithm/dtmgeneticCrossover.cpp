#include "dtmgeneticAlgorithm.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace
{

class SynapseDepthIds
{
public:
  size_t inNeuronDepthId;
  size_t outNeuronDepthId;

  bool operator==(const SynapseDepthIds &other) const
  {
    return inNeuronDepthId == other.inNeuronDepthId &&
           outNeuronDepthId == other.outNeuronDepthId;
  }
};

class SynapseDepthIdsHash
{
public:
  size_t operator()(const SynapseDepthIds &depthIds) const
  {
    const size_t inNeuronHash = std::hash<size_t>{}(depthIds.inNeuronDepthId);
    const size_t outNeuronHash = std::hash<size_t>{}(depthIds.outNeuronDepthId);
    return inNeuronHash ^ (outNeuronHash << 1);
  }
};

/******************************************************************************
 * @brief Gets the comparable endpoint depthIds of a synapse
 *
 * Synapses store globally unique neuron IDs, so each ID must first be resolved
 * through the owning model's indexMap before its neuron's depthId is available.
 *
 * @param model   Model that owns the synapse and its endpoint neurons
 * @param synapse Synapse whose endpoints should be resolved
 *
 * @return Input and output neuron depthIds of the synapse
 ******************************************************************************/
SynapseDepthIds getSynapseDepthIds(const DTModel &model, const Synapse &synapse)
{
  const auto inNeuronPosition = model.indexMap.find(synapse.inNeuronId);
  const auto outNeuronPosition = model.indexMap.find(synapse.outNeuronId);

  assert(inNeuronPosition != model.indexMap.end());
  assert(outNeuronPosition != model.indexMap.end());
  assert(inNeuronPosition->second < model.neurons.size());
  assert(outNeuronPosition->second < model.neurons.size());

  const Neuron &inNeuron = model.neurons[inNeuronPosition->second];
  const Neuron &outNeuron = model.neurons[outNeuronPosition->second];

  assert(inNeuron.id == synapse.inNeuronId);
  assert(outNeuron.id == synapse.outNeuronId);

  return {inNeuron.depthId, outNeuron.depthId};
}

}

/******************************************************************************
 * @brief Finds corresponding neurons and synapses in two dynamic topology models
 *
 * Neurons correspond when their depthId values are equal. Synapses correspond
 * when both their input-neuron depthIds and output-neuron depthIds are equal.
 * Every element is matched at most once. Only outgoing synapse lists are
 * inspected because they are the canonical connection storage in DTModel.
 * Both models must have up-to-date depthId values before this function is
 * called; the comparison does not sort or otherwise modify either model.
 *
 * @param firstModel  First model to compare
 * @param secondModel Second model to compare
 *
 * @return Corresponding neuron IDs and synapses from both models
 ******************************************************************************/
SimilarDTModelElements DTMGeneticAlgorithm::findSimilarNeuronsAndSynapses(
    const DTModel &firstModel, const DTModel &secondModel)
{
  SimilarDTModelElements result;

  result.firstModelNeuronIds.reserve(
      std::min(firstModel.neurons.size(), secondModel.neurons.size()));
  result.secondModelNeuronIds.reserve(
      std::min(firstModel.neurons.size(), secondModel.neurons.size()));

  // A vector per depthId also handles malformed or transitional models that
  // temporarily contain more than one neuron with the same depthId.
  std::unordered_map<size_t, vector<size_t>> secondNeuronIdsByDepthId;
  secondNeuronIdsByDepthId.reserve(secondModel.neurons.size());

  for (const Neuron &neuron : secondModel.neurons)
  {
    secondNeuronIdsByDepthId[neuron.depthId].push_back(neuron.id);
  }

  for (const Neuron &neuron : firstModel.neurons)
  {
    const auto matchingNeurons = secondNeuronIdsByDepthId.find(neuron.depthId);

    if (matchingNeurons == secondNeuronIdsByDepthId.end() ||
        matchingNeurons->second.empty())
    {
      continue;
    }

    result.firstModelNeuronIds.push_back(neuron.id);
    result.secondModelNeuronIds.push_back(matchingNeurons->second.back());
    matchingNeurons->second.pop_back();
  }

  size_t firstModelSynapseCount = 0;
  size_t secondModelSynapseCount = 0;

  for (const Neuron &neuron : firstModel.neurons)
  {
    firstModelSynapseCount += neuron.outSynapses.size();
  }

  for (const Neuron &neuron : secondModel.neurons)
  {
    secondModelSynapseCount += neuron.outSynapses.size();
  }

  const size_t maximumMatchingSynapseCount =
      std::min(firstModelSynapseCount, secondModelSynapseCount);
  result.firstModelSynapses.reserve(maximumMatchingSynapseCount);
  result.secondModelSynapses.reserve(maximumMatchingSynapseCount);

  // Store pointers into the second model so its potentially large synapse
  // vectors are not copied while constructing the lookup table.
  std::unordered_map<SynapseDepthIds, vector<const Synapse *>, SynapseDepthIdsHash>
      secondSynapsesByDepthIds;
  secondSynapsesByDepthIds.reserve(secondModelSynapseCount);

  for (const Neuron &neuron : secondModel.neurons)
  {
    for (const Synapse &synapse : neuron.outSynapses)
    {
      const SynapseDepthIds depthIds = getSynapseDepthIds(secondModel, synapse);
      secondSynapsesByDepthIds[depthIds].push_back(&synapse);
    }
  }

  for (const Neuron &neuron : firstModel.neurons)
  {
    for (const Synapse &synapse : neuron.outSynapses)
    {
      const SynapseDepthIds depthIds = getSynapseDepthIds(firstModel, synapse);
      const auto matchingSynapses = secondSynapsesByDepthIds.find(depthIds);

      if (matchingSynapses == secondSynapsesByDepthIds.end() ||
          matchingSynapses->second.empty())
      {
        continue;
      }

      result.firstModelSynapses.push_back(synapse);
      result.secondModelSynapses.push_back(*matchingSynapses->second.back());
      matchingSynapses->second.pop_back();
    }
  }

  return result;
}

/******************************************************************************
 * @brief Finds model neurons and synapses absent from its similar elements
 *
 * Neurons and synapses are excluded using their model-local unique IDs. Only
 * outgoing synapse lists are inspected because they are the canonical
 * connection storage in DTModel. Input vectors are read by reference and are
 * expected to be the vectors for this model returned by
 * findSimilarNeuronsAndSynapses.
 *
 * @param model            Model whose non-similar elements should be found
 * @param similarNeuronIds IDs of this model's neurons found to be similar
 * @param similarSynapses  This model's synapses found to be similar
 *
 * @return IDs and synapses from the model that are absent from the inputs
 ******************************************************************************/
NonSimilarDTModelElements DTMGeneticAlgorithm::findNonSimilarNeuronsAndSynapses(
    const DTModel &model,
    const vector<size_t> &similarNeuronIds,
    const vector<Synapse> &similarSynapses)
{
  NonSimilarDTModelElements result;

  std::unordered_set<size_t> similarNeuronIdSet;
  similarNeuronIdSet.reserve(similarNeuronIds.size());
  similarNeuronIdSet.insert(similarNeuronIds.begin(), similarNeuronIds.end());

  result.neuronIds.reserve(model.neurons.size());

  for (const Neuron &neuron : model.neurons)
  {
    if (!similarNeuronIdSet.contains(neuron.id))
    {
      result.neuronIds.push_back(neuron.id);
    }
  }

  std::unordered_set<size_t> similarSynapseIdSet;
  similarSynapseIdSet.reserve(similarSynapses.size());

  for (const Synapse &synapse : similarSynapses)
  {
    similarSynapseIdSet.insert(synapse.id);
  }

  size_t modelSynapseCount = 0;

  for (const Neuron &neuron : model.neurons)
  {
    modelSynapseCount += neuron.outSynapses.size();
  }

  result.synapses.reserve(modelSynapseCount);

  for (const Neuron &neuron : model.neurons)
  {
    for (const Synapse &synapse : neuron.outSynapses)
    {
      if (!similarSynapseIdSet.contains(synapse.id))
      {
        result.synapses.push_back(synapse);
      }
    }
  }

  return result;
}



/******************************************************************************
 * @brief Builds one offspring per parent pair and replaces the population
 *
 * Refreshes each selected parent's topological order once, matches genes by
 * depthId, and obtains non-similar genes from the fitter parent. Equal fitness
 * selects the first parent as dominant. Parents must have valid acyclic models
 * with unique depthIds within each model and the same input/output layout and
 * output activation.
 *
 * Builds an input/output skeleton, then creates hidden neurons with fresh IDs.
 * Shared hidden-neuron bias and activation, and shared synapse weight and
 * activity, are chosen independently from either parent with 50% probability.
 * Shared output neurons inherit a randomly chosen bias at their fixed ID.
 * Dominant-only genes retain their attributes, with fresh IDs for hidden
 * neurons and synapses. Endpoint depthIds map connections onto child neurons.
 *
 * Sorts each child and assigns a fresh individual ID and currentGeneration.
 * Fitness starts at zero pending evaluation. Even a pair containing the same
 * parent twice produces a new individual. All species are flattened into one
 * population, whose size becomes the number of supplied parent pairs.
 *
 * @param parentsLists Parent pairs grouped by species, containing indexes into
 *                     the current population; both parents must be present
 ******************************************************************************/
void DTMGeneticAlgorithm::crossover(const vector<vector<GAParents>> &parentsLists)
{
  TIME_MEASURE_BEGIN(DTM_CROSSOVER);
  size_t offspringCount = 0;
  for (const vector<GAParents> &speciesParents : parentsLists)
  {
    offspringCount += speciesParents.size();
  }

  // Population is public and may contain externally created individuals.
  // Synchronize once per generation so fresh IDs also avoid those genes.
  for (const DTIndividual &individual : population)
  {
    uniqueIndividualIdCounter = std::max(uniqueIndividualIdCounter, individual.id + 1);
    for (const Neuron &neuron : individual.model.neurons)
    {
      uniqueNeuronIdCounter = std::max(uniqueNeuronIdCounter, neuron.id + 1);
      for (const Synapse &synapse : neuron.outSynapses)
      {
        uniqueSynapseIdCounter = std::max(uniqueSynapseIdCounter, synapse.id + 1);
      }
    }
  }

  vector<DTIndividual> newPopulation;
  newPopulation.reserve(offspringCount);
  vector<bool> preparedParents(population.size(), false);

  for (const vector<GAParents> &speciesParents : parentsLists)
  {
    for (const GAParents &parents : speciesParents)
    {
      assert(parents.isFirstParentPresent && parents.isSecondParentPresent);

      // Refresh even an initially sorted skeleton: its depthIds may never
      // have been assigned. Repeatedly selected parents are only sorted once.
      for (size_t parentIndex : {parents.firstParentIndex, parents.secondParentIndex})
      {
        DTModel &model = population.at(parentIndex).model;
        if (!preparedParents[parentIndex])
        {
          model.sortTopologically();
          preparedParents[parentIndex] = true;
        }
      }

      const DTIndividual &firstParent = population[parents.firstParentIndex];
      const DTIndividual &secondParent = population[parents.secondParentIndex];
      const DTModel &firstModel = firstParent.model;
      const DTModel &secondModel = secondParent.model;
      const bool firstIsDominant = firstParent.fitness >= secondParent.fitness;
      const DTModel &dominantModel = firstIsDominant ? firstModel : secondModel;

      size_t gracePeriodLength = firstParent.gracePeriodLength;

      if (gracePeriodLength < secondParent.gracePeriodLength)
      {
        gracePeriodLength = secondParent.gracePeriodLength;
      }

      if (gracePeriodLength != 0)
      {
        gracePeriodLength--;
      }

      assert(firstModel.inputSize == secondModel.inputSize);
      assert(firstModel.outputSize == secondModel.outputSize);

      const SimilarDTModelElements similar =
          findSimilarNeuronsAndSynapses(firstModel, secondModel);
      const vector<size_t> &dominantSimilarNeuronIds = firstIsDominant
          ? similar.firstModelNeuronIds : similar.secondModelNeuronIds;
      const vector<Synapse> &dominantSimilarSynapses = firstIsDominant
          ? similar.firstModelSynapses : similar.secondModelSynapses;
      const NonSimilarDTModelElements nonSimilar = findNonSimilarNeuronsAndSynapses(
          dominantModel, dominantSimilarNeuronIds, dominantSimilarSynapses);

      const ActivationE outputActivation = dominantModel.neurons[
          dominantModel.indexMap.at(dominantModel.inputSize)].activation;
      DTModel offspringModel(dominantModel.inputSize, dominantModel.outputSize,
                             outputActivation);
      offspringModel.neurons.reserve(dominantModel.neurons.size());

      // All child neurons are inserted before any connections. Cache vector
      // indexes under the parent's depthIds until the final sort changes them.
      std::unordered_map<size_t, size_t> offspringIndexesByDepthId;
      offspringIndexesByDepthId.reserve(dominantModel.neurons.size());
      for (size_t index = 0; index < offspringModel.neurons.size(); index++)
      {
        Neuron &neuron = offspringModel.neurons[index];
        const Neuron &source = dominantModel.neurons[dominantModel.indexMap.at(neuron.id)];
        neuron.depthId = source.depthId;
        neuron.bias = source.bias;
        offspringIndexesByDepthId.emplace(neuron.depthId, index);
      }

      auto addHiddenNeuron = [&](const Neuron &source, double bias, ActivationE activation)
      {
        Neuron neuron{};
        neuron.id = getNewUniqueNeuronId();
        neuron.depthId = source.depthId;
        neuron.type = NeuronTypeE::HIDDEN_NEURON;
        neuron.bias = bias;
        neuron.activation = activation;
        const size_t index = offspringModel.neurons.size();
        offspringIndexesByDepthId.emplace(neuron.depthId, index);
        offspringModel.indexMap.emplace(neuron.id, index);
        offspringModel.neurons.push_back(std::move(neuron));
      };

      for (size_t index = 0; index < similar.firstModelNeuronIds.size(); index++)
      {
        const Neuron &firstNeuron = firstModel.neurons[
            firstModel.indexMap.at(similar.firstModelNeuronIds[index])];
        const Neuron &secondNeuron = secondModel.neurons[
            secondModel.indexMap.at(similar.secondModelNeuronIds[index])];
        const Neuron &dominantNeuron = firstIsDominant ? firstNeuron : secondNeuron;

        if (dominantNeuron.type == NeuronTypeE::INPUT_NEURON)
        {
          continue;
        }
        if (dominantNeuron.type == NeuronTypeE::OUTPUT_NEURON)
        {
          const DTModel &biasModel = rand() % 2 == 0 ? firstModel : secondModel;
          offspringModel.neurons[offspringModel.indexMap.at(dominantNeuron.id)].bias =
              biasModel.neurons[biasModel.indexMap.at(dominantNeuron.id)].bias;
          continue;
        }

        const double bias = rand() % 2 == 0 ? firstNeuron.bias : secondNeuron.bias;
        const ActivationE activation = rand() % 2 == 0
            ? firstNeuron.activation : secondNeuron.activation;
        addHiddenNeuron(dominantNeuron, bias, activation);
      }

      for (size_t neuronId : nonSimilar.neuronIds)
      {
        const Neuron &neuron = dominantModel.neurons[dominantModel.indexMap.at(neuronId)];
        if (neuron.type == NeuronTypeE::HIDDEN_NEURON)
        {
          addHiddenNeuron(neuron, neuron.bias, neuron.activation);
        }
      }

      auto addSynapse = [&](const Synapse &source, double weight, bool isActive)
      {
        const SynapseDepthIds depthIds = getSynapseDepthIds(dominantModel, source);
        Neuron &inNeuron = offspringModel.neurons[
            offspringIndexesByDepthId.at(depthIds.inNeuronDepthId)];
        Neuron &outNeuron = offspringModel.neurons[
            offspringIndexesByDepthId.at(depthIds.outNeuronDepthId)];
        Synapse synapse(getNewUniqueSynapseId(), inNeuron.id, outNeuron.id, weight);
        synapse.isActive = isActive;

        // IDs and endpoints are known here. Append directly to avoid model
        // insertion's full neuron scan, maintaining both adjacency lists.
        inNeuron.outSynapses.push_back(synapse);
        outNeuron.inSynapses.push_back(synapse);
      };

      for (size_t index = 0; index < similar.firstModelSynapses.size(); index++)
      {
        const Synapse &firstSynapse = similar.firstModelSynapses[index];
        const Synapse &secondSynapse = similar.secondModelSynapses[index];
        const double weight = rand() % 2 == 0 ? firstSynapse.weight : secondSynapse.weight;
        const bool isActive = rand() % 2 == 0 ? firstSynapse.isActive : secondSynapse.isActive;
        addSynapse(dominantSimilarSynapses[index], weight, isActive);
      }
      for (const Synapse &synapse : nonSimilar.synapses)
      {
        addSynapse(synapse, synapse.weight, synapse.isActive);
      }

      offspringModel.isSorted = false;
      offspringModel.sortTopologically();
      newPopulation.emplace_back(getNewUniqueIndividualCounter(), currentGeneration,
                                 std::move(offspringModel));
      // To avoid copy we use emplace_back but after that we need to update gracePeriodLength
      // Since its not a part of the constructor.
      newPopulation[newPopulation.size() - 1].gracePeriodLength = gracePeriodLength;
    }
  }

  population = std::move(newPopulation);
  hyperparameters.populationSize = population.size();
  TIME_MEASURE_END(DTM_CROSSOVER);
}
