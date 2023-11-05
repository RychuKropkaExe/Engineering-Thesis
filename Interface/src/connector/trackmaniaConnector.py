from tminterface.commandlist import InputCommand, InputType
from pygbx import Gbx, GbxType
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
from src.logger.log import log
from src.blocks.blockPositions import STADIUM_BLOCKS_DICT, createPositionDictionary, checkPosition, checkNextBlock, checkNextElements
class MainClient(Client):
    logFile = open("logFile.txt", "w")
    x = 1000
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()
        checkPosition(state.position)
        # Neural network inputs:
        # map = MAPS_SET[mapName]
        # x = state.position.x
        # y = state.position.y
        # z = state.position.z
        # speed = state.display_speed
        # velocity = state.velocity
        # ypw = state.yaw_pitch_roll
        # yaw = ypw[0]
        # pitch = ypw[1]
        # roll = ypw[2]
        # turningRate = state.scene_mobil.turning_rate
        # gerabox = state.scene_mobil.engine.gear
        # currBlock = checkPosition(state.position)
        # currBlockRotation = 
        # nextBlocks = checkNextElements(state.position)
        # nextBlock = nextElements[0]
        # nextBlockRotation = 
        # secondNextBlock = nextElements[1]
        # secondNextBlockRotation = 
        # distanceToCurrentBlock ? 
        # distanceToNextBlock -> Distance to current block end
        # distance to secondNextBlock -> Distance to next block end 
        # iface.set_input_state(accelerate=True)
        if _time > self.x:
            print("TURNING RATE:", state.scene_mobil.turning_rate)
            print("CURRENT GEAR:", state.scene_mobil.engine.gear)
            print("CURRENT BLOCK:",checkPosition(state.position))
            nextElements = checkNextElements(state.position)
            print("NEXT BLOCK:", nextElements[0])
            print("SECOND NEXT BLOCK:", nextElements[1])
            print(state.yaw_pitch_roll)
            self.x += 1000
        #     iface.set_input_state(accelerate=False, steer=30000)
        # if _time > 10000:
        #     iface.give_up()
        #self.logFile.write(str(state.dyna.current_state.position))
        # self.logFile.write(
        #     f'Time: {_time}\n'
        #     f'Display Speed: {state.display_speed}\n'
        #     f'Position: {state.position}\n'
        #     f'Velocity: {state.velocity}\n'
        #     f'YPW: {state.yaw_pitch_roll}\n'
        # )


def start():
    g = Gbx('C:\\Users\Admin\Documents\\TrackMania\\Tracks\\Challenges\\My Challenges\\Test12.Challenge.Gbx')
    challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    challenge = challenges[0]
    log("MAP BLOCKS:")
    for block in challenge.blocks:
        log("-",block.name)
    createPositionDictionary(challenge.blocks)
    server_name = f'TMInterface{sys.argv[2]}' if len(sys.argv) > 2 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)