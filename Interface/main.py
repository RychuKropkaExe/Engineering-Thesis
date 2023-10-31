from src.stadiumBlocks import rotateMatrix
from test.testRotateMatrix import runMatrixTest
from src.trackmaniaConnector import start
import sys

def main():
    if len(sys.argv) < 2:
        print("No mode provided")
    if "-t" in sys.argv:
        runMatrixTest()
    elif "-r" in sys.argv:
        start()


if __name__ == '__main__':
    main()
