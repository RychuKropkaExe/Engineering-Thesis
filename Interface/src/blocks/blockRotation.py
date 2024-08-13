from src.logger.log import log


def getNumberSign(number):
    if number >= 0:
        return 1
    elif number < 0:
        return -1


# Next block of 2d block can be either in x axis
# or z axis, this array describes the cycle
ROTATION_CYCLE = [[-1, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, 1]]


# This function returns what values to add to sub block to get
# The whole block ending point
# Then, we also return what values needs to be added after rotation
# Both are nedeed in order to find rotated blocks beggining
# and end
def findRotatedEndings(endsPosition, rotation, maxX, maxZ):
    end1 = endsPosition[0]
    end1CycleIndex = 0

    end2 = endsPosition[1]
    end2CycleIndex = 0

    log("END1:", end1)
    log("END2:", end2)

    # counter = 0
    # for x, _, z in ROTATION_CYCLE:
    #     if getNumberSign(x) == getNumberSign(end1[0]) and getNumberSign(z) == getNumberSign(end1[2]):
    #         end1CycleIndex = counter
    #     if getNumberSign(x) == getNumberSign(end2[0]) and getNumberSign(z) == getNumberSign(end2[2]):
    #         end2CycleIndex = counter
    #     counter += 1

    if end1[0] < 0:
        end1CycleIndex = 0
    elif end1[0] > maxX:
        end1CycleIndex = 2
    elif end1[2] < 0:
        end1CycleIndex = 1
    elif end1[2] > maxZ:
        end1CycleIndex = 3

    if end2[0] < 0:
        end2CycleIndex = 0
    elif end2[0] > maxX:
        end2CycleIndex = 2
    elif end2[2] < 0:
        end2CycleIndex = 1
    elif end2[2] > maxZ:
        end2CycleIndex = 3

    log("END1 CYCLE INDEX:", end1CycleIndex)
    log("END2 CYCLE INDEX:", end2CycleIndex)
    return [
        (ROTATION_CYCLE[end1CycleIndex], ROTATION_CYCLE[end2CycleIndex]),
        (
            ROTATION_CYCLE[(end1CycleIndex + rotation) % 4],
            ROTATION_CYCLE[(end2CycleIndex + rotation) % 4],
        ),
    ]


# Rotate matrix 90 degrees with respect to [0,0,0] point
# ROTATIONS_TYPES
# 0 - Normal, take just offsets
# 1 - Transponse matrix once
# 2 - Transponse second time
# 3 - Transponse thrid time
def rotateMatrix(mat):

    # Number of rows
    N = len(mat)

    # Number of columns
    M = len(mat[0])

    newN = M
    newM = N

    result = []

    for _ in range(newN):
        row = [0] * newM
        result.append(row)

    for i in range(N):
        for j in range(M):
            result[j][newM - (i + 1)] = mat[i][j]

    return result


def getRotatedPositions(currentPositions, endingPoints, rotation):
    end1 = endingPoints[0]
    end2 = endingPoints[1]
    log("ENDING POINTS: ", end1, end2)

    maxX = 0
    maxZ = 0
    for x, y, z in currentPositions:
        if x > maxX:
            maxX = x
        if z > maxZ:
            maxZ = z
    # To compensate for not inclusive python
    endingPointsRotatedPosition = findRotatedEndings(endingPoints, rotation, maxX, maxZ)

    endingPointsDifference = endingPointsRotatedPosition[0]
    end1PDF = endingPointsDifference[0]
    end2PDF = endingPointsDifference[1]

    rotatedEndingPointsDifference = endingPointsRotatedPosition[1]
    end1RPDF = rotatedEndingPointsDifference[0]
    end2RPDF = rotatedEndingPointsDifference[1]

    maxX += 1
    maxZ += 1

    blockMatrix = []

    for _ in range(maxZ):
        blockMatrixRow = [0] * maxX
        blockMatrix.append(blockMatrixRow)
    # No idea how it works
    log("END1 PDF: ", end1PDF)
    log("END2 PDF: ", end2PDF)
    log("END1 RPDF: ", end1RPDF)
    log("END2 RPDF: ", end2RPDF)
    for x, y, z in currentPositions:
        # 0 -> fake block
        # 1 -> normal block
        # 2 -> end1 congruent block
        # 3 -> end2 congruent block
        # log("CURRENT X:",x,"Z:", z, "end2: X:", end2[0], "Z:",end2)
        if x + end1PDF[0] == end1[0] and z + end1PDF[2] == end1[2]:
            log("ENDING BLOCK POSITION 1: X:", x + end1PDF[0], "Y:", z + end1PDF[2])
            blockMatrix[z][x] = 2
        elif x + end2PDF[0] == end2[0] and z + end2PDF[2] == end2[2]:
            log("ENDING BLOCK POSITION 2: X:", x + end2PDF[0], "Y:", z + end2PDF[2])
            blockMatrix[z][x] = 3
        else:
            blockMatrix[z][x] = 1
    log("MATRIX BEFORE ROTATION:", blockMatrix)
    for i in range(rotation):
        blockMatrix = rotateMatrix(blockMatrix)
        log("MATRIX AFTER", i, "ROTATION:", blockMatrix)

    newBlocksPositions = []
    newEnd1 = [0, 0, 0]
    newEnd2 = [0, 0, 0]
    # Try to remember why it works like that challenge(impossible)
    for x in range(len(blockMatrix)):
        for z in range(len(blockMatrix[0])):
            if blockMatrix[x][z] == 1:
                newBlocksPositions.append([z, 0, x])
            elif blockMatrix[x][z] == 2:
                log("ROTATED END CONGRUENT POSITION: ", [z, 0, x])
                newBlocksPositions.append([z, 0, x])
                newEnd1 = [z + end1RPDF[0], 0, x + end1RPDF[2]]
                log("newEnd1 POSITION: ", newEnd1)
            elif blockMatrix[x][z] == 3:
                log("ROTATED END CONGRUENT POSITION 2: ", [z, 0, x])
                newBlocksPositions.append([z, 0, x])
                newEnd2 = [z + end2RPDF[0], 0, x + end2RPDF[2]]
                log("newEnd2 POSITION: ", newEnd2)

    return (newBlocksPositions, [newEnd1, newEnd2])
