#ifndef PYTHON_API_H
#define PYTHON_API_H
#include "FastMatrix.h"
#include "trainingData.h"
#include "model.h"

enum ActionsE{
    NO_ACTION,
    FORWARD,
    // LEFT,
    // RIGHT,
    FORWARD_RIGHT,
    FORWARD_LEFT,
    ACTIONS_COUNT
};

class TrackmaniaAgent{

    public:
        Model mainModel;
        Model targetModel;
        TrainingData trainingData;

        vector<vector<double>> stateBuffer {};
        vector<ActionsE> actionBuffer {};
        vector<double> rewardBuffer {};
        vector<vector<double>> nextStateBuffer {};
        vector<bool> doneBuffer {};
        size_t bufferSize {};
        size_t maxBufferSize {};
        

};

#endif