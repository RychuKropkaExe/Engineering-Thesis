import agent
from src.logger.log import log
from enum import IntEnum, auto
from tminterface.interface import TMInterface
from src.blocks.blockPositions import STADIUM_BLOCKS_DICT, createPositionDictionary, checkPosition, checkNextBlock, checkNextElements, getEndsDistances
import random

class StateValuesE(IntEnum):
    X_CORD: int = auto()
    Y_CORD: int  = auto()
    Z_CORD: int  = auto()
    SPEED: int  = auto()
    VELOCITY_X: int  = auto()
    VELOCITY_Z: int  = auto()
    YAW: int  = auto()
    PITCH: int  = auto()
    ROLL: int  = auto()
    TURNING_RATE: int  = auto()
    GEAR_BOX: int  = auto()
    CURR_BLOCK_ID: int  = auto()
    CURR_BLOCK_ROTATION: int  = auto()
    NEXT_BLOCK_ID: int  = auto()
    NEXT_BLOCK_ROTATION: int  = auto()
    SECOND_NEXT_BLOCK_ID: int   = auto()
    SECOND_NEXT_BLOCK_ROTATION: int  = auto()
    DISTANCE_TO_END_1: int  = auto()
    DISTANCE_TO_END_2: int  = auto()
    STATE_VALUES_COUNT: int  = auto()

# Hyperparameters.
numOfEpisodes = 1000
iterationsInEachLearningSession = 1
batchSize = 25
discount = 0.5
curFrame = 0
MAX_NUMBER_OF_BATCHES=100000

def createState(iface: TMInterface):

    state = iface.get_simulation_state()
    curState = [0.0]*int(StateValuesE.STATE_VALUES_COUNT)
    #Neural network inputs:
    #map = MAPS_SET[mapName] -> probably not
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
    # distanceToNextBlock -> Distance to current block end
    # distance to secondNextBlock -> Distance to next block end
    curState[StateValuesE.X_CORD] = float(x)/100
    curState[StateValuesE.Y_CORD] = float(y)/100
    curState[StateValuesE.Z_CORD] = float(z)/100
    curState[StateValuesE.SPEED] = float(speed/50)
    curState[StateValuesE.VELOCITY_X] = float(velocity[0]/10)
    curState[StateValuesE.VELOCITY_Z] = float(velocity[2]/10)
    curState[StateValuesE.YAW] = float(yaw)
    curState[StateValuesE.PITCH] = float(pitch)
    curState[StateValuesE.ROLL] = float(roll)
    curState[StateValuesE.TURNING_RATE] = float(turningRate)
    curState[StateValuesE.GEAR_BOX] = float(gerabox)
    curState[StateValuesE.CURR_BLOCK_ID] = float(currBlockId)
    curState[StateValuesE.CURR_BLOCK_ROTATION] = float(currBlockRotation)
    curState[StateValuesE.NEXT_BLOCK_ID] = float(nextBlockId)
    curState[StateValuesE.NEXT_BLOCK_ROTATION] = float(nextBlockRotation)
    curState[StateValuesE.SECOND_NEXT_BLOCK_ID] = float(secondNextBlockId)
    curState[StateValuesE.SECOND_NEXT_BLOCK_ROTATION] = float(secondNextBlockRotation)
    curState[StateValuesE.DISTANCE_TO_END_1] = float(end1Distance)/100
    curState[StateValuesE.DISTANCE_TO_END_2] = float(end2Distance)/100

    return curState, currBlock.elementsDictKey

def reward(prevState, curState, prevKey, curKey, time):
    res = 0
    if prevKey != curKey:
        res+=10
    
    res += int((curState[StateValuesE.SPEED] - prevState[StateValuesE.SPEED]))

    return res/(time/1000)

def remember(state, action, nextState, reward, done):
    print("TRYING TO SAVE VALUES: ", state, action, nextState, reward, done)
    agent.remember(state, action, nextState, reward, done)
    print("SAVED VALUES SUCCESFULLY")

def train(stateSize):
    agent.agentLearn(iterationsInEachLearningSession, batchSize, stateSize, int(agent.ActionsE.ACTIONS_COUNT), discount)

def createModel():
    modelArch = [int(StateValuesE.STATE_VALUES_COUNT), int(StateValuesE.STATE_VALUES_COUNT), 30, 10, int(agent.ActionsE.ACTIONS_COUNT)]
    archSize = len(modelArch)
    activationLayers = [agent.ActivationFunctionE.RELU, agent.ActivationFunctionE.RELU, agent.ActivationFunctionE.RELU, agent.ActivationFunctionE.NO_ACTIVATION]
    activationLayersSize = len(activationLayers)
    agent.createMainModel(modelArch, archSize, activationLayers, activationLayersSize, True)
    agent.createTargetModel()
    agent.initializeBuffers(MAX_NUMBER_OF_BATCHES)

def printModels():
    agent.printModels()

def updateTargetModel():
    agent.createTargetModel()

def getGreedyInput(state, epsilon):
    """Take random action with probability epsilon, else take best action."""
    value = random.random()
    if value > epsilon:
        result = agent.runModel(state, int(StateValuesE.STATE_VALUES_COUNT))
        print("RESULT:", result)
        maxQ = -1
        resultAction = 0
        for i in range(len(result)):
            if result[i] > maxQ:
                maxQ = result[i]
                resultAction = i
        # Random action (left or right).
        return resultAction
    else:
        return random.randint(0, int(agent.ActionsE.ACTIONS_COUNT)-1)