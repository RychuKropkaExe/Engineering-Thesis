from stadiumBlocks import *
from blockRotation import getRotatedPositions
import math
positionsDict = {}

# Simple Hash function
# Creates a bitstring from all three
# And then convert it back to number
def hashPosition(x, y, z):
    # [2:] to trim 0b from the string
    bitStringX = bin(x)[2:]
    bitStringY = bin(y)[2:]
    bitStringZ = bin(z)[2:]
    return int(bitStringX+bitStringY+bitStringZ, 2)

# Inserts into dictionary all block positions, and types
# Function uses block coordinates which are x/32 y/8 z/32
# If block is rotated it performs a calculation to determine
# Its position. It also puts there the ends of each block
# so that we can predict which block is next on track.
def createPositionDictionary(blocks):
    for block in blocks:
        rotation = block.rotation
        posX = block.position.x
        posY = block.position.y
        posZ = block.position.z
        print("BLOCK POSITION: ", block.name, posX, posY, posZ)
        print("BLOCK ROTATION: ", rotation)
        if rotation == 0:  
            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                print("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                positionsDict[hashValue] = (block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'])
        elif len(STADIUM_BLOCK_OFFSETS[block.name]['positions']) == 1:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']

            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                print("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                positionsDict[hashValue] = (block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'])
        else:
            newBlocksPositions = getRotatedPositions(STADIUM_BLOCK_OFFSETS[block.name]['positions'], rotation) 

            #print("FOR BLOCK:", block.name, "ROTATED POSITIONS ARE: ", newBlocksPositions)
            for x, y, z in newBlocksPositions:
                print("NEW BLOCK POSITION: ", block.name, posX + x, posY + y, posZ + z)
                hashValue = hashPosition(posX + x, posY + y,posZ + z)
                positionsDict[hashValue] = (block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'])
    #print(positionsDict)



def checkPosition(position):
    print("RAW POSITION: ",position)
    # posX = round(int(position[0]) / BLOCK_SIZE_XZ)
    # posY = round(int(position[1]) / BLOCK_SIZE_Y)
    # posZ = round(int(position[2]) / BLOCK_SIZE_XZ)
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    print("CAR POSITION: ", posX, posY, posZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        return positionsDict[hashValue]
    else:
        return 'Nothing'