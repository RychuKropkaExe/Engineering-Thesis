from tminterface.interface import TMInterface
from tminterface.commandlist import InputCommand, InputType
from tminterface.client import Client, run_client
from pygbx import Gbx, GbxType
import sys
from src.stadiumBlocks import STADIUM_BLOCKS_DICT, createPositionDictionary, checkPosition
class MainClient(Client):
    logFile = open("logFile.txt", "w")
    x = 1000
    def __init__(self) -> None:
        print("SIEMA ENIU")
        super(MainClient, self).__init__()
        print("PODAJ TLENU")

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()

        # iface.set_input_state(accelerate=True)
        if _time > self.x:
            print(checkPosition(state.position))
            self.x += 1000
        #     iface.set_input_state(accelerate=False, steer=30000)
        # if _time > 10000:
        #     iface.give_up()
        self.logFile.write(str(state.dyna.current_state.position))
        # self.logFile.write(
        #     f'Time: {_time}\n'
        #     f'Display Speed: {state.display_speed}\n'
        #     f'Position: {state.position}\n'
        #     f'Velocity: {state.velocity}\n'
        #     f'YPW: {state.yaw_pitch_roll}\n'
        # )


def start():
    g = Gbx('C:\\Users\Admin\Documents\\TrackMania\\Tracks\\Challenges\\My Challenges\\Test9.Challenge.Gbx')
    challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    challenge = challenges[0]
    for block in challenge.blocks:
        print(block.name)
    createPositionDictionary(challenge.blocks)
    print()
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 2 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)