from src.stadiumBlocks import rotateMatrix
from test.testRotateMatrix import runMatrixTest
import sys

def main():
    if len(sys.argv) < 2:
        print("No mode provided")
    if "-t" in sys.argv:
        runMatrixTest()


if __name__ == '__main__':
    main()
