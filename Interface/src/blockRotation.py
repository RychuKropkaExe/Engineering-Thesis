# Todo: Implement
def findRotatedPointPosition(point, rows, columns, rotation):
    return


def getNumberSign(number):
   if (number>0):
      return 1
   elif (number<0):
      return -1
   else:
      return 0

# Next block of 2d block can be either in x axis
# or z axis, this array describes the cycle
ROTATION_CYCLE = [[-1, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, 1]]
def findRotatedEndings(endsPosition, rotation):
    end1 = endsPosition[0]
    end1CycleIndex = 0

    end2 = endsPosition[1]
    end2CycleIndex = 0

    counter = 0
    for x, _, z in ROTATION_CYCLE:
        if getNumberSign(x) == getNumberSign(end1[0]) and getNumberSign(z) == getNumberSign(end1[2]):
            end1CycleIndex = counter
        if getNumberSign(x) == getNumberSign(end2[0]) and getNumberSign(z) == getNumberSign(end2[2]):
            end2CycleIndex = counter
        counter += 1
    
    return (ROTATION_CYCLE[(end1CycleIndex + rotation) % 4], ROTATION_CYCLE[(end2CycleIndex + rotation) % 4])


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
        row = [0]*newM
        result.append(row)

    for i in range(N):
        for j in range(M):
            result[j][newM-(i+1)] = mat[i][j]
    
    return result

def getRotatedPositions(currentPositions, rotation):
    maxX = 0
    maxZ = 0
    for x, y, z in currentPositions:
        if x > maxX:
            maxX = x
        if z > maxZ:
            maxZ = z
    # To compensate for not inclusive python 
    maxX += 1
    maxZ += 1

    blockMatrix = []

    for _ in range(maxZ):
        blockMatrixRow = [0] * maxX
        blockMatrix.append(blockMatrixRow)
    # No idea how it works
    for x, y, z in currentPositions:
        blockMatrix[z][x] = 1
    print(blockMatrix)
    for _ in range(rotation):
        blockMatrix = rotateMatrix(blockMatrix)
        print(blockMatrix)

    newBlocksPositions = []

    # Try to remember why it works like that challenge(impossible)
    for x in range(len(blockMatrix)):
        for z in range(len(blockMatrix[0])):
            if blockMatrix[x][z] == 1:
                newBlocksPositions.append([z, 0, x])
    return newBlocksPositions