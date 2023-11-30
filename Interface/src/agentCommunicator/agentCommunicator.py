import agent
from src.logger.log import log
from enum import Enum, auto
from tminterface.interface import TMInterface
from src.blocks.blockPositions import STADIUM_BLOCKS_DICT, createPositionDictionary, checkPosition, checkNextBlock, checkNextElements, getEndsDistances

class StateValuesE(Enum):
    X_CORD = auto()
    Y_CORD = auto()
    Z_CORD = auto()
    SPEED = auto()
    VELOCITY = auto()
    YAW = auto()
    PITCH = auto()
    ROLL = auto()
    TURNING_RATE = auto()
    GEAR_BOX = auto()
    CURR_BLOCK_ID = auto()
    CURR_BLOCK_ROTATION = auto()
    NEXT_BLOCK_ID = auto()
    NEXT_BLOCK_ROTATION = auto()
    SECOND_NEXT_BLOCK_ID  = auto()
    SECOND_NEXT_BLOCK_ROTATION = auto()
    DISTANCE_TO_END_1 = auto()
    DISTANCE_TO_END_2 = auto()
    STATE_VALUES_COUNT = auto()

# Hyperparameters.
numOfEpisodes = 1000
iterationsInEachLearningSession = 10
epsilon = 1.0
batchSize = 100
discount = 0.99
curFrame = 0

def createState(iface: TMInterface):

    state = iface.get_simulation_state()
    curState = [0]*int(StateValuesE.STATE_VALUES_COUNT)
    #Neural network inputs:
    #map = MAPS_SET[mapName] -> probably not
    x = state.position.x
    y = state.position.y
    z = state.position.z

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

    distancesToEnds = getEndsDistances(currBlock)
    end1Distance = distancesToEnds[0]
    end2Distance = distancesToEnds[1]
    # distanceToNextBlock -> Distance to current block end
    # distance to secondNextBlock -> Distance to next block end
    curState[StateValuesE.X_CORD] = x
    curState[StateValuesE.Y_CORD] = y
    curState[StateValuesE.Z_CORD] = z
    curState[StateValuesE.SPEED] = speed
    curState[StateValuesE.VELOCITY] = velocity
    curState[StateValuesE.YAW] = yaw
    curState[StateValuesE.PITCH] = pitch
    curState[StateValuesE.ROLL] = roll
    curState[StateValuesE.TURNING_RATE] = turningRate
    curState[StateValuesE.GEAR_BOX] = gerabox
    curState[StateValuesE.CURR_BLOCK_ID] = currBlockId
    curState[StateValuesE.CURR_BLOCK_ROTATION] = currBlockRotation
    curState[StateValuesE.NEXT_BLOCK_ID] = nextBlockId
    curState[StateValuesE.NEXT_BLOCK_ROTATION] = nextBlockRotation
    curState[StateValuesE.SECOND_NEXT_BLOCK_ID] = secondNextBlockId
    curState[StateValuesE.SECOND_NEXT_BLOCK_ROTATION] = secondNextBlockRotation
    curState[StateValuesE.DISTANCE_TO_END_1] = end1Distance
    curState[StateValuesE.DISTANCE_TO_END_2] = end2Distance

    return curState


def remember(state, action, nextState, reward, done):
    agent.remember(state, action, nextState, reward, done)

def train(stateSize):
    agent.agentLearn(iterationsInEachLearningSession, batchSize, stateSize, int(agent.ActionsE.ACTIONS_COUNT), discount)

def createModel():
    return

def updateTargetModel():
    return