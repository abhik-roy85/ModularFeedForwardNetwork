import numpy as np


def testFun(a=None, b=None):
    print(a, b)


if __name__ == "__main__":
    print("Hello")
    x = list()
    print(x)

    testFun()

    for item in reversed(range(5)):
        print(item)

    lst = [1, 2, 3, 4, 5, 6]
    lst = np.concatenate((lst, []))
    print(lst)
    lst1 = [7, 8, 9, 0]
    lst2 = np.concatenate((lst, lst1))
    print(lst2)
    lst3 = [11, 12, 13, 14]
    lst4 = np.concatenate((lst2, lst3))
    print(lst4)
    print(len(lst4))

    for i in reversed(range(2)):
        print(i)

    X = 5
    print(X+2 if X < 5 else X-2)

    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(lst[3:5])

    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(np.array(X))


# That's all
