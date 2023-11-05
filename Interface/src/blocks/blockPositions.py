from src.blocks.stadiumBlocks import *
from src.blocks.blockRotation import getRotatedPositions
from src.logger.log import log
import math

# Wrapper for elements of position dictionary
class PositionDictEntery():

    blockName: str
    rotation: int
    blockEndingPoints: tuple
    visited: int
    blockStartingPosition: tuple
    elementsDictKey: int

    def __init__(self, blockName: str, rotation: int, blockEndingPoints: tuple, visited: int, blockStartingPosition: tuple, elementsDictKey: int):
        self.blockName = blockName
        self.rotation = rotation
        self.blockEndingPoints = blockEndingPoints
        self.visited = visited
        self.blockStartingPosition = blockStartingPosition
        self.elementsDictKey = elementsDictKey
    
    def __str__(self):
        return f'Name: {self.blockName} Rotation: {self.rotation} Visited: {self.visited} BlockEndingPoints: {self.blockEndingPoints} BlockStartingPosition: {self.blockStartingPosition}'

# Dictionary containing info about given position
# e.g what block is that
# Strutcutre:
# {Int(hash)} => (String(block name), Int(block rotation), Tuple{Array, Array}(Block ending points), Bool(if block was visited), Tuple{Int,Int,Int}(block starting position))
# BLOCK_NAME_KEY = 0
# BLOCK_ROTATION_KEY = 1
# BLOCK_ENDINGS_KEY = 2
# BLOCK_VISITED_FLAG_KEY = 3
# BLOCK_STARTING_POSITION_KEY = 4
positionsDict = {}

# Dictionary containing all blocks of an elements
# Used to determinate more than one next block
elementsDict = {}

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

        elementsDictKey = hashPosition(posX, posY, posZ)
        
        elementsBlocksList = [] 

        if rotation == 0:  
            for x, y, z in STADIUM_BLOCK_OFFSETS[block.name]['positions']:
                log("SUBBLOCK POSITION: ", posX+x, posY+y, posZ+z)
                hashValue = hashPosition(posX+x,posY+y,posZ+z)
                elementsBlocksList.append(hashValue)
                positionsDict[hashValue] = PositionDictEntery(block.name, block.rotation, STADIUM_BLOCK_OFFSETS[block.name]['ends'], 0, [posX, posY, posZ], elementsDictKey)
            
            elementsDict[elementsDictKey] = elementsBlocksList

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
                elementsBlocksList.append(hashValue)
                positionsDict[hashValue] = PositionDictEntery(block.name, block.rotation, newEnds, 0, [posX, posY, posZ], elementsDictKey)
            elementsDict[elementsDictKey] = elementsBlocksList
        else:
            ends = STADIUM_BLOCK_OFFSETS[block.name]['ends']
            newBlocksPositions, newEnds = getRotatedPositions(STADIUM_BLOCK_OFFSETS[block.name]['positions'], ends, rotation)

            #log("FOR BLOCK:", block.name, "ROTATED POSITIONS ARE: ", newBlocksPositions)
            for x, y, z in newBlocksPositions:
                log("NEW BLOCK POSITION: ", block.name, posX + x, posY + y, posZ + z)
                hashValue = hashPosition(posX + x, posY + y,posZ + z)
                elementsBlocksList.append(hashValue)
                positionsDict[hashValue] = PositionDictEntery(block.name, block.rotation, newEnds, 0, [posX, posY, posZ], elementsDictKey)
            elementsDict[elementsDictKey] = elementsBlocksList
    #log(positionsDict)

def checkNextBlock(position, excludedElementHash):
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        positionInfo = positionsDict[hashValue]
        endingPoints = positionInfo.blockEndingPoints
        startingPoint = positionInfo.blockStartingPosition
        end1 = endingPoints[0]
        end2 = endingPoints[1]
        end1HashValue = hashPosition(end1[0]+startingPoint[0], end1[1]+startingPoint[1], end1[2]+startingPoint[2])
        end2HashValue = hashPosition(end2[0]+startingPoint[0], end2[1]+startingPoint[1], end2[2]+startingPoint[2])
        if end1HashValue in positionsDict:
            if positionsDict[end1HashValue].visited == False and positionsDict[end1HashValue].elementsDictKey != excludedElementHash:
                endingPosition = [(end1[0]+startingPoint[0])*BLOCK_SIZE_XZ, (end1[1]+startingPoint[1])*BLOCK_SIZE_Y, (end1[2]+startingPoint[2])*BLOCK_SIZE_XZ]
                return (positionsDict[end1HashValue], endingPosition)
        if end2HashValue in positionsDict:
            if positionsDict[end2HashValue].visited == False and positionsDict[end2HashValue].elementsDictKey != excludedElementHash:
                endingPosition = [(end2[0]+startingPoint[0])*BLOCK_SIZE_XZ, (end2[1]+startingPoint[1])*BLOCK_SIZE_Y, (end2[2]+startingPoint[2])*BLOCK_SIZE_XZ]
                return (positionsDict[end2HashValue], endingPosition)
        else:
            print("SECOND NOTHING")
            return 'Nothing'
        print("THIRD NOTHING")
    else:
        print("FIRST NOTHING")
        return 'Nothing'

def checkPosition(position):
    #print("RAW POSITION: ",position)
    posX = math.floor(int(position[0]) / BLOCK_SIZE_XZ)
    posY = math.floor(int(position[1]) / BLOCK_SIZE_Y)
    posZ = math.floor(int(position[2]) / BLOCK_SIZE_XZ)
    #print("CAR POSITION: ", posX, posY, posZ)
    hashValue = hashPosition(posX,posY,posZ)
    if hashValue in positionsDict:
        positionsDict[hashValue].visited = True
        return positionsDict[hashValue]
    else:
        return 'Nothing'

def checkNextElements(position):
    currPosition = checkPosition(position)
    excludedHash = currPosition.elementsDictKey
    nextElement = checkNextBlock(position, excludedHash)
    nextElementBlockPosition = nextElement[1]
    nextExcludedHash = nextElement[0].elementsDictKey
    
    secondNextElement = checkNextBlock(nextElementBlockPosition, nextExcludedHash)
    print(secondNextElement)
    return (nextElement[0], secondNextElement[0])