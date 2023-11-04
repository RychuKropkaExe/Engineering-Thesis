from src.blocks.stadiumBlocks import *
from src.blocks.blockRotation import getRotatedPositions
from src.logger.log import log
import math

# Dictionary containing info about given position
# e.g what block is that
# Strutcutre:
# {Int(hash)} => (String(block name), Int(block rotation), Tuple{Array, Array}(Block ending points), Bool(if block was visited), Tuple{Int,Int,Int}(block starting position))
BLOCK_NAME_KEY = 0
BLOCK_ROTATION_KEY = 1
BLOCK_ENDINGS_KEY = 2
BLOCK_VISITED_FLAG_KEY = 3
BLOCK_STARTING_POSITION_KEY = 4

positionsDict = {}

# Simple Hash function
# Creates a bitstring from all three
# And then convert it back to number
def hashPosition(x, y, z):
    #log("POSITION TO HASH:", x, y, z)
    # [2:] to trim 0b from the string
    bitStringX = bin(x)[2:]
    bitStringY = bin(y)[2:]
    bitStringZ = bin(z)[2:]
    #log("RESULTRINGH BITSTRING:", bitStringX, bitStringY, bitStringZ)
    return int(bitStringX+bitStringY+bitStringZ, 2)

# Inserts into dictionary all block positions, and types
# Function uses block coordinates which are x/32 y/8 z/32
# If block is rotated it performs a calculation to determine
# Its position. It also puts there the ends of each block
# so that we can predict which block is next on track.
def createPositionDictionary(blocks):
    for block in blocks:
        log("BLOCK OFFSET: ", STADIUM_BLOCK_OFFSETS[block.name]['positions'])
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
                positionsDict[hashValue] = (block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'], False, [posX, posY, posZ])
        elif len(STADIUM_BLOCK_OFFSETS[block.name]['positions']) == 1:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']
            end1 = ends[0]
            end2 = ends[1]
            if rotation % 2 == 1:
                newEnd1 = [end1[2], end1[1], end1[0]]
                newEnd2 = [end2[2], end2[1], end2[0]]
            else:
                newEnd1 = end1
                newEnd2 = end2
            log("newEnd1:", newEnd1)
            log("newEnd2:", newEnd2)
            newEnds = [newEnd1, newEnd2]

            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                log("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                positionsDict[hashValue] = (block.name, block.rotation, newEnds, False, [posX, posY, posZ])
        else:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']
            newBlocksPositions, newEnds = getRotatedPositions(STADIUM_BLOCK_OFFSETS[block.name]['positions'], ends, rotation)

            #log("FOR BLOCK:", block.name, "ROTATED POSITIONS ARE: ", newBlocksPositions)
            for x, y, z in newBlocksPositions:
                log("NEW BLOCK POSITION: ", block.name, posX + x, posY + y, posZ + z)
                hashValue = hashPosition(posX + x, posY + y,posZ + z)
                positionsDict[hashValue] = (block.name, block.rotation, newEnds, False, [posX, posY, posZ])
    #log(positionsDict)

def checkNextBlock(position):
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        positionInfo = positionsDict[hashValue]
        endingPoints = positionInfo[BLOCK_ENDINGS_KEY]
        startingPoint = positionInfo[BLOCK_STARTING_POSITION_KEY]
        end1 = endingPoints[0]
        end2 = endingPoints[1]
        end1HashValue = hashPosition(end1[0]+startingPoint[0], end1[1]+startingPoint[1], end1[2]+startingPoint[2])
        end2HashValue = hashPosition(end2[0]+startingPoint[0], end2[1]+startingPoint[1], end2[2]+startingPoint[2])
        if end1HashValue in positionsDict:
            if positionsDict[end1HashValue][3] == False:
                return positionsDict[end1HashValue]
        if end2HashValue in positionsDict:
            if positionsDict[end2HashValue][3] == False:
                return positionsDict[end2HashValue]
        else:
            return 'Nothing'
        # Add entry to position dictionary that will be a key to a dictionary containing all given block positions
    else:
        return 'Nothing'

def checkPosition(position):
    print("RAW POSITION: ",position)
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    print("CAR POSITION: ", posX, posY, posZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        temp = list(positionsDict[hashValue])
        temp[3] = True
        positionsDict[hashValue] = tuple(temp)
        return positionsDict[hashValue]
    else:
        return 'Nothing'