#ifndef PYTHON_API_H
#define PYTHON_API_H
#include "FastMatrix.h"
#include "model.h"
#include "trainingData.h"

/******************************************************************************
 * @enum ActionsE
 *
 * @brief List of all actions that agent can take. Note, that in this
 *        case NO_ACTION is a valid action that makes the agent do
 *        nothing(it wont accelerate, break nor steer)
 *
 ******************************************************************************/
enum ActionsE
{
    NO_ACTION,
    FORWARD,
    // LEFT,
    // RIGHT,
    FORWARD_RIGHT,
    FORWARD_LEFT,
    ACTIONS_COUNT
};

/******************************************************************************
 * @class RunCompletion
 *
 * @brief Wrapper for data from one agent run
 *
 * @public @param stateBuffer   All states from given run
 * @public @param actionBuffer  All action taken in given run
 * @public @param bufferSize    Size of @stateBuffer vector
 * @public @param reward        Reward for given run
 * @public @param done          Tells if run was completed, or terminated
 ******************************************************************************/
class RunCompletion
{
public:
    /******************************************************************************
     * CLASS MEMBERS
     ******************************************************************************/
    vector<vector<double>> stateBuffer{};
    vector<ActionsE> actionBuffer{};
    size_t bufferSize{};
    double reward{};
    bool done;
};

/******************************************************************************
 * @class TrackmaniaAgent
 *
 * @brief Class representing agent along with its "mind"
 *
 * @public @param mainModel     Main model for agent
 * @public @param targetModel   Target model, used for q-learing
 * @public @param trainingData  Training data for main model
 * @public @param bestRuns      Vector of
 * @public @param maxBufferSize Tells if run was completed, or terminated
 ******************************************************************************/
class TrackmaniaAgent
{

public:
    /******************************************************************************
     * CLASS MEMBERS
     ******************************************************************************/
    Model mainModel;
    Model targetModel;
    TrainingData trainingData;

    vector<RunCompletion> bestRuns;

    size_t maxBufferSize{};
};

#endif
