import agent
from src.logger.log import log
from enum import IntEnum, auto
from tminterface.interface import TMInterface
from src.blocks.blockPositions import (
    STADIUM_BLOCKS_DICT,
    getEndsAngles,
    createPositionDictionary,
    checkPosition,
    checkNextBlock,
    checkNextElements,
    getEndsDistances,
    hashPosition,
)
import random
from src.blocks.stadiumBlocks import BLOCK_SIZE_XZ, BLOCK_SIZE_Y

SPEED_CONSTANT = 100


class StateValuesE(IntEnum):
    X_CORD: int = 0
    # Y_CORD: int  = auto()
    Z_CORD: int = auto()
    SPEED: int = auto()
    # VELOCITY_X: int  = auto()
    # VELOCITY_Z: int  = auto()
    YAW: int = auto()
    # PITCH: int  = auto()
    # ROLL: int  = auto()
    TURNING_RATE: int = auto()
    # GEAR_BOX: int  = auto()
    CURR_BLOCK_ID: int = auto()
    CURR_BLOCK_ROTATION: int = auto()
    NEXT_BLOCK_ID: int = auto()
    NEXT_BLOCK_ROTATION: int = auto()
    # SECOND_NEXT_BLOCK_ID: int   = auto()
    # SECOND_NEXT_BLOCK_ROTATION: int  = auto()
    DISTANCE_TO_END_1: int = auto()
    ANGLE_TO_END_1: int = auto()
    DISTANCE_TO_END_2: int = auto()
    ANGLE_TO_END_2: int = auto()
    STATE_VALUES_COUNT: int = auto()


# Hyperparameters.
numOfEpisodes = 1000
iterationsInEachLearningSession = 10
batchSize = 200
discount = 0.99
curFrame = 0
MAX_NUMBER_OF_BATCHES = 200000

# def normalize(state):
#     for i in range(len(state)):
#         state[i] = state[i]


def createState(iface: TMInterface, _time):

    state = iface.get_simulation_state()
    curState = [0.0] * int(StateValuesE.STATE_VALUES_COUNT)
    # Neural network inputs:
    # map = MAPS_SET[mapName] -> probably not
    x = state.position[0]
    y = state.position[1]
    z = state.position[2]

    speed = state.display_speed
    velocity = state.velocity

    ypw = state.yaw_pitch_roll
    yaw = ypw[0]
    pitch = ypw[1]
    roll = ypw[2]

    turningRate = state.scene_mobil.turning_rate
    gerabox = state.scene_mobil.engine.gear

    currBlock = checkPosition(state.position)
    currBlockId = STADIUM_BLOCKS_DICT[currBlock.blockName]
    currBlockRotation = currBlock.rotation

    nextBlocks = checkNextElements(state.position)
    nextBlock = nextBlocks[0]
    nextBlockId = STADIUM_BLOCKS_DICT[nextBlock.blockName]
    nextBlockRotation = nextBlock.rotation

    secondNextBlock = nextBlocks[1]
    secondNextBlockId = STADIUM_BLOCKS_DICT[secondNextBlock.blockName]
    secondNextBlockRotation = secondNextBlock.rotation

    distancesToEnds = getEndsDistances(currBlock, state.position)
    end1Distance = distancesToEnds[0]
    end2Distance = distancesToEnds[1]
    endsAngles = getEndsAngles(currBlock, state.position)
    end1Angle = endsAngles[0]
    end2Angle = endsAngles[1]

    curState[StateValuesE.ANGLE_TO_END_1] = end1Angle
    curState[StateValuesE.ANGLE_TO_END_2] = end2Angle
    # distanceToNextBlock -> Distance to current block end
    # distance to secondNextBlock -> Distance to next block end
    curState[StateValuesE.X_CORD] = round(float(x) / 100, 3)
    # curState[StateValuesE.Y_CORD] = round(float(y)/1000, 3)
    curState[StateValuesE.Z_CORD] = round(float(z) / 100, 3)
    curState[StateValuesE.SPEED] = round(float(speed) / SPEED_CONSTANT, 3)
    # curState[StateValuesE.VELOCITY_X] = round(float(velocity[0])/100, 3)
    # curState[StateValuesE.VELOCITY_Z] = round(float(velocity[2])/100, 3)
    curState[StateValuesE.YAW] = round(float(yaw) / 10, 3)
    # curState[StateValuesE.PITCH] = round(float(pitch), 3)
    # curState[StateValuesE.ROLL] = round(float(roll), 3)
    curState[StateValuesE.TURNING_RATE] = round(float(turningRate) / 1, 3)
    # curState[StateValuesE.GEAR_BOX] = round(float(gerabox)/10, 3)
    curState[StateValuesE.CURR_BLOCK_ID] = round(float(currBlockId), 3)
    curState[StateValuesE.CURR_BLOCK_ROTATION] = round(float(currBlockRotation), 3)
    curState[StateValuesE.NEXT_BLOCK_ID] = round(float(nextBlockId), 3)
    curState[StateValuesE.NEXT_BLOCK_ROTATION] = round(float(nextBlockRotation), 3)
    # curState[StateValuesE.SECOND_NEXT_BLOCK_ID] = round(float(secondNextBlockId), 3)
    # curState[StateValuesE.SECOND_NEXT_BLOCK_ROTATION] = round(float(secondNextBlockRotation), 3)
    curState[StateValuesE.DISTANCE_TO_END_1] = round(float(end1Distance) / 10, 3)
    curState[StateValuesE.DISTANCE_TO_END_2] = round(float(end2Distance) / 10, 3)

    # normalize(curState)

    # return curState, currBlock.elementsDictKey
    return curState, hashPosition(
        int(x / BLOCK_SIZE_XZ), int(y / BLOCK_SIZE_Y), int(z / BLOCK_SIZE_XZ)
    )


def reward(prevState, curState, prevKey, curKey, blockChangeTime, time):
    res = 0
    # print("PREVIOUS KEY: ", prevKey)
    # print("CURRENT KEY: ", curKey)
    if prevKey != curKey:
        # print("REWARD GIVEN")
        if blockChangeTime <= 300:
            res += 4
        else:
            res += 2
    # if curState[StateValuesE.SPEED]*SPEED_CONSTANT > prevState[StateValuesE.SPEED]*SPEED_CONSTANT:
    #     res+=0.05
    # res+=(curState[StateValuesE.SPEED]*SPEED_CONSTANT)/250
    res += int(
        (
            curState[StateValuesE.SPEED] * SPEED_CONSTANT
            - prevState[StateValuesE.SPEED] * SPEED_CONSTANT
        )
    ) / (15)
    # return res
    return res * (10000 / (10000 + time))


def remember(state, action, nextState, reward, done):
    # print("TRYING TO SAVE VALUES: ", state, action, nextState, reward, done)
    # print("CURRENT REWARD: ", reward/10)
    agent.remember(state, action, nextState, reward / 10, done)
    # print("SAVED VALUES SUCCESFULLY")


def train(sampleCount, clipGradient):
    agent.agentLearn(
        1,
        sampleCount,
        int(StateValuesE.STATE_VALUES_COUNT),
        int(agent.ActionsE.ACTIONS_COUNT),
        discount,
        batchSize,
        clipGradient,
    )
    # agent.printModels()


def setThresholds(minThreshold, maxThreshold):
    agent.setMinThreshold(minThreshold)
    agent.setMaxThreshold(maxThreshold)


def createModel():
    modelArch = [
        int(StateValuesE.STATE_VALUES_COUNT),
        int(StateValuesE.STATE_VALUES_COUNT),
        10,
        10,
        int(agent.ActionsE.ACTIONS_COUNT),
    ]
    archSize = len(modelArch)
    activationLayers = [
        agent.ActivationFunctionE.RELU,
        agent.ActivationFunctionE.RELU,
        agent.ActivationFunctionE.RELU,
        agent.ActivationFunctionE.SOFTMAX,
    ]
    activationLayersSize = len(activationLayers)
    agent.createMainModel(
        modelArch, archSize, activationLayers, activationLayersSize, True
    )
    agent.createTargetModel()
    agent.initializeBuffers(MAX_NUMBER_OF_BATCHES)


def setLearningRate(rate):
    agent.setTrainingRate(rate)


def printModels():
    agent.printModels()


def updateTargetModel():
    agent.createTargetModel()


def getGreedyInput(state, epsilon):
    # print("EPSILON VALUE: ", epsilon)
    """Take random action with probability epsilon, else take best action."""
    value = random.random()
    # print("RANDOM VALUE: ", value)
    if value > epsilon:
        result = agent.runModel(state, int(StateValuesE.STATE_VALUES_COUNT))
        print("RESULT:", result)
        moveChance = random.random()
        prob = 1
        resultAction = 0
        for i in range(len(result)):
            if result[i] >= moveChance:
                resultAction = i
                break
        return resultAction
    else:
        randomAction = random.randint(0, int(agent.ActionsE.ACTIONS_COUNT) - 1)
        # print("RANDOM ACTION: ", randomAction)
        return randomAction
