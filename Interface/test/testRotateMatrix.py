from src.blocks.blockRotation import rotateMatrix
import agent

def runMatrixTest():
    #a = [[1,2], [3,4], [5,6]]
    #a = [[5, 3, 1], [6, 4, 2]]
    #a = [[6, 5], [4, 3], [2, 1]]
    a = [[2, 4, 6], [1, 3, 5]]
    a = rotateMatrix(a)
    log(a)