#ifndef PYTHON_API_H
#define PYTHON_API_H
#include "../fastMatrix/FastMatrix.h"
#include "../model/trainingData.h"
#include "../model/model.h"

enum ActionsE{
    NO_ACTION,
    FORWARD,
    LEFT,
    RIGHT,
    FORWARD_RIGHT,
    FORWARD_LEFT,
    ACTIONS_COUNT
};

class TrackmaniaAgent{

    public:
        Model mainModel;
        Model targetModel;
        TrainingData trainingData;

        vector<vector<float>> stateBuffer {};
        vector<ActionsE> actionBuffer {};
        vector<float> rewardBuffer {};
        vector<vector<float>> nextStateBuffer {};
        vector<bool> doneBuffer {};
        size_t bufferSize {};
        size_t maxBufferSize {};
        

};

#endif