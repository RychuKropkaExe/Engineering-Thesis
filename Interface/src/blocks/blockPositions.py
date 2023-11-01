from src.blocks.stadiumBlocks import *
from src.blocks.blockRotation import getRotatedPositions
from src.logger.log import log
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
        log("BLOCK POSITION: ", str(block.name), str(posX), str(posY), str(posZ))
        log("BLOCK ROTATION: ", rotation)
        if rotation == 0:  
            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                log("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                positionsDict[hashValue] = (block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'])
        elif len(STADIUM_BLOCK_OFFSETS[block.name]['positions']) == 1:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']

            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                log("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                positionsDict[hashValue] = (block.name, block.rotation, ends)
        else:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']
            newBlocksPositions, newEnds = getRotatedPositions(STADIUM_BLOCK_OFFSETS[block.name]['positions'], ends, rotation)

            #log("FOR BLOCK:", block.name, "ROTATED POSITIONS ARE: ", newBlocksPositions)
            for x, y, z in newBlocksPositions:
                log("NEW BLOCK POSITION: ", block.name, posX + x, posY + y, posZ + z)
                hashValue = hashPosition(posX + x, posY + y,posZ + z)
                positionsDict[hashValue] = (block.name, block.rotation, newEnds)
    #log(positionsDict)



def checkPosition(position):
    log("RAW POSITION: ",position)
    # posX = round(int(position[0]) / BLOCK_SIZE_XZ)
    # posY = round(int(position[1]) / BLOCK_SIZE_Y)
    # posZ = round(int(position[2]) / BLOCK_SIZE_XZ)
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    log("CAR POSITION: ", posX, posY, posZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        return positionsDict[hashValue]
    else:
        return 'Nothing'