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
    FORWARD_LEFT
};

class TrackmaniaAgent{

    public:
        Model mainModel;
        Model targetModel;
        TrainingData trainingData;

        vector<vector<float>> stateBuffer {};
        size_t stateSize {};
        vector<ActionsE> actionBuffer {};
        size_t actionSiize {};
        vector<float> rewardBuffer {};
        vector<vector<float>> nextStateBuffer {};
        vector<bool> doneBuffer {};
        size_t bufferSize {};
        size_t maxBufferSize {};
        

};

#endif