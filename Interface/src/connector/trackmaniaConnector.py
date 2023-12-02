import array
from tminterface.commandlist import InputCommand, InputType
from pygbx import Gbx, GbxType
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
from src.logger.log import log
from src.blocks.blockPositions import STADIUM_BLOCKS_DICT, createPositionDictionary, checkPosition, checkNextBlock, checkNextElements
from src.agentCommunicator.agentCommunicator import createModel, createState, reward, remember, getGreedyInput, batchSize, train, updateTargetModel, printModels
from agent import ActionsE
from pynput.keyboard import Key, Controller
import time
import threading
keyboard = Controller()

def pressEnterKey():
    print("SIEMA ENIU PRESS KEY")
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

def pressResetKey():
    print("SIEMA ENIU PRESS KEY")
    keyboard.press(Key.backspace)
    keyboard.release(Key.backspace)

class MainClient(Client):
    logFile = open("logFile.txt", "w")

    nextInputTime = 200
    timeDelta = 200

    maxStagnationTime = 3000
    lastBlockChangeTime: int = 0

    curState: list[float] = []
    prevState: list[float] = []

    actionTaken = 0

    curBlockKey: int = 0
    prevBlockKey: int = 0

    curReward: float = 0.0

    firstStateFlag = True

    epsilon = 1.0

    resetFlag = False

    curIteration = 0

    samplesTaken = 0

    bestTime = 1000000000

    def __init__(self) -> None:
        createModel()
        #printModels()
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        iface.set_speed(4.0)

    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()
        if _time == 0:
            self.curReward = 0
            iface.set_input_state(accelerate=True)
            self.actionTaken = ActionsE.FORWARD
            self.resetFlag = False
        if self.firstStateFlag == True and _time >= 100:
            self.curReward = 0
            self.prevState, self.curBlockKey = createState(iface)
            self.nextInputTime += self.timeDelta
            #log("FIRST STATE: ", self.prevState)
            self.firstStateFlag = False
        if _time >= self.nextInputTime:
            self.curState, self.curBlockKey = createState(iface)
            self.curReward += reward(self.prevState, self.curState, self.prevBlockKey, self.curBlockKey, _time)
            if self.curBlockKey != self.prevBlockKey:
                self.lastBlockChangeTime = _time
            remember(self.prevState, self.actionTaken, self.curState, self.curReward, False)
            self.samplesTaken += 1
            #log("PREV STATE: ", self.prevState, "CUR STATE: ", self.curState, "REWARD: ", self.curReward)
            self.actionTaken = getGreedyInput(self.curState, self.epsilon)
            match int(self.actionTaken):
                case int(ActionsE.NO_ACTION):
                    iface.set_input_state(accelerate=False, left=False, right=False)
                    self.actionTaken = ActionsE.NO_ACTION
                case int(ActionsE.FORWARD):
                    iface.set_input_state(accelerate=True, left=False, right=False)
                    self.actionTaken = ActionsE.FORWARD
                case int(ActionsE.LEFT):
                    iface.set_input_state(accelerate=False, left=True, right=False)
                    self.actionTaken = ActionsE.LEFT
                case int(ActionsE.RIGHT):
                    iface.set_input_state(accelerate=False, left=False, right=True)
                    self.actionTaken = ActionsE.RIGHT
                case int(ActionsE.FORWARD_RIGHT):
                    iface.set_input_state(accelerate=True, left=False, right=True)
                    self.actionTaken = ActionsE.FORWARD_RIGHT
                case int(ActionsE.FORWARD_LEFT):
                    iface.set_input_state(accelerate=True, left=True, right=False)
                    self.actionTaken = ActionsE.FORWARD_LEFT
                case default:
                    print("ERROR NO VALID INPUT: ", int(self.actionTaken))
            if self.samplesTaken >= batchSize:
                #print("STARTING TRENING")
                train(self.samplesTaken)
            if self.samplesTaken % 50 == 0:
                printModels()
                updateTargetModel()
            self.nextInputTime += self.timeDelta
            self.prevBlockKey = self.curBlockKey
            self.prevState = self.curState
        if state.player_info.race_finished == True and self.resetFlag == False:
            print("FINISHED!!!!!!!!!!!")
            if _time < self.bestTime:
                print("NEW RECORD: ", _time)
                self.bestTime = _time
            self.curState, self.curBlockKey = createState(iface)
            self.curReward += reward(self.prevState, self.curState, self.prevBlockKey, self.curBlockKey, _time)
            self.curReward += 100
            remember(self.prevState, self.actionTaken, self.curState, self.curReward, True)
            self.samplesTaken += 1
            self.resetFlag = True
            train(self.samplesTaken)
            self.nextInputTime = 200
            if self.epsilon >= 0.02:
                self.epsilon -= 0.01
            self.firstStateFlag = True
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
            threading.Timer(0.01, pressResetKey).start()
            threading.Timer(3, pressEnterKey).start()
        if _time - self.lastBlockChangeTime >= 10000 and self.resetFlag == False:
            self.curState, self.curBlockKey = createState(iface)
            self.curReward += reward(self.prevState, self.curState, self.prevBlockKey, self.curBlockKey, _time)
            remember(self.prevState, self.actionTaken, self.curState, self.curReward, False)
            self.samplesTaken += 1
            train(self.samplesTaken)
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
            if self.epsilon >= 0.02:
                self.epsilon -= 0.01
            self.nextInputTime = 200
            self.firstStateFlag = True
            self.lastBlockChangeTime = 0
            self.curReward = 0
            self.resetFlag = True


def start():
    g = Gbx('C:\\Users\\Admin\\Desktop\\Umieralnia\\PracaDyplomowa\\Engineering-Thesis\\Interface\\MyChallenges\\Map2.Challenge.Gbx')
    challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    challenge = challenges[0]
    log("MAP BLOCKS:")
    for block in challenge.blocks:
        log("-",block.name)
    createPositionDictionary(challenge.blocks)
    server_name = f'TMInterface{sys.argv[2]}' if len(sys.argv) > 2 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)