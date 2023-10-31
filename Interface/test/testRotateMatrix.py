from Interface.src.stadiumBlocks import rotateMatrix

def main():
    a = [[1,2], [3,4], [5,6]]
    #a = [[5, 3, 1], [6, 4, 2]]
    a = rotateMatrix(a)
    print(a)


if __name__ == '__main__':
    main()