#include "utils.h"
#include <vector>

using std::vector;
using std::pair;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;
using DTMUtils::MutationE;

/******************************************************************************
 * @class Hyperparameters
 *
 * @brief Represents hyperparameters used for genetic algorithm
 *
 * @public @param mutationTypes     Mutations attempted after each crossover
 * @public @param numberOfMutations Number of distinct individuals selected for
 *                                  each mutation at the corresponding index
 * @public @param weightMutationStrength Maximum absolute additive weight change
 * @public @param biasMutationStrength   Maximum absolute additive bias change
 ******************************************************************************/
class Hyperparameters
{
public:
  /******************************************************************************
  * CLASS MEMBERS
  ******************************************************************************/
  size_t populationSize{};
  size_t maxNumberOfNeurons{};

  size_t inputSize{};
  size_t outputSize{};

  size_t gracePeriodLength{};

  size_t tournamentSize;

  vector<MutationE> mutationTypes{};
  vector<size_t> numberOfMutations{};

  double weightMutationStrength{1.0};
  double biasMutationStrength{1.0};

  ActivationE outputActivation;

  /******************************************************************************
  * CONSTRUCTORS
  ******************************************************************************/
  Hyperparameters() = default;

  /******************************************************************************
  * OPERATORS
  ******************************************************************************/
  friend std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparameters);
};
