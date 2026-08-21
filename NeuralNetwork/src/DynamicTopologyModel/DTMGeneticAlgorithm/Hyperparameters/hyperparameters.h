#include "utils.h"
#include <vector>

using std::vector;
using std::pair;
using DTMUtils::NeuronTypeE;
using DTMUtils::ActivationE;

/******************************************************************************
 * @class Hyperparameters
 *
 * @brief Represents hyperparameters used for genetic algorithm
 *
 * @public @param id          Synapse id
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
