import numpy
import itertools
import mnist

def getCombinations(numInputs: int, degrees: int):
    result = []
    for i in range(degrees+1):
        comb = itertools.combinations_with_replacement(range(numInputs), i)
        result += list(comb)
    return result

def getSum(inputs: list[list[int]], combination):
    sum = 0
    for n in range(len(inputs)):
        additional = 1
        for x in combination:
            additional *= inputs[n][x]
        sum += additional
    return sum

def partition(combination):
    result = []
    for pattern in itertools.product([True,False],repeat=len(combination)):
        part1 = [x[1] for x in zip(pattern,combination) if x[0]]
        part2 = [x[1] for x in zip(pattern,combination) if not x[0]]
        if len(part1) <= 2 and len(part2) <= 2 and [part1, part2] not in result:
            result.append([part1, part2])
    return result

def getA(inputs: list[list[int]], resultDimension: int, degrees: int):
    result =  numpy.zeros((resultDimension, resultDimension))
    for combinationDegree in range(degrees*degrees+1):
        # TODO: parallelize
        for combination in itertools.combinations_with_replacement(range(len(inputs[0])), combinationDegree):
            sum = getSum(inputs, combination)
            partitions = partition(combination)
            for partition in partitions:
                # TODO: put sum into appripriate spots
                # p1 = sum_partition with offset
                # p2 = sum_partition with offset
                # result[p1][p2] = sum
                input("meep")
    return result

def getB(inputs: list[list[int]], outputs: list[int], combinations: list):
    bDimension = len(combinations)
    numInputs = len(inputs)

    result = numpy.zeros(bDimension)
    for n in range(numInputs):
        for i in range(bDimension):
            additional = outputs[n]
            for x in combinations[i]:
                additional *= inputs[n][x]
            result[i] += additional

    return result

def getDeleteRows(inputs: list[list[int]], combinations: list, C):
    numInputs = len(inputs)
    numTerms = len(C)

    contributions = [0] * numTerms
    for i in range(numTerms):
        for n in range(numInputs):
            additional = C[i]
            for x in combinations[i]:
                additional *= inputs[n][x]
            contributions[i] += abs(additional)

    totalContributions = sum(contributions)
    result = []
    for i in range(numTerms):
        # TODO: Improve term removal logic that don't contribute 5% to equation
        divisor = len(combinations[i]) if len(combinations[i]) > 0 else 1
        if contributions[i] / divisor * 20 < totalContributions:
            result.append(i)
    return result

def getD(A, deleteRows):
    D = numpy.delete(A, deleteRows, 0)
    D = numpy.delete(D, deleteRows, 1)
    return D

def getE(B, deleteRows):
    return numpy.delete(B, deleteRows, 0)

def printOutput(F, combinations, inputs: list[list[int]], outputs: list[int]):
    outputElements = []
    for i in range(0, len(combinations)):
        outputElements.append(f'{F[i]:.{2}f}*x{combinations[i]}')
    print('----------------------------------------------------------')
    print('y_est = ' + ' + '.join(outputElements))
    print('----------------------------------------------------------')

    y_hat = [0] * len(outputs)
    for n in range(len(outputs)):
        for i in range(len(combinations)):
            additional = F[i]
            for x in combinations[i]:
                additional *= inputs[n][x]
            y_hat[n] += additional
    print ("{:<10} {:<10}".format('y','y_est'))
    for i in range(len(outputs)):
        print("{:<10} {:<10}".format(round(outputs[i],2), round(y_hat[i],2)))

def execute(x, y, degrees):
    combinations = getCombinations(len(x[0]), degrees)
    A = getA(x, len(combinations), degrees)
    # B = getB(x, y, combinations)
    # C = np.linalg.solve(A, B)

    # deleteRows = getDeleteRows(x, combinations, C)
    # combinations = [i for j, i in enumerate(combinations) if j not in deleteRows]
    # D = getD(A, deleteRows)
    # E = getE(B, deleteRows)
    # F = np.linalg.solve(D, E)

    # # TODO: Might do multiple reductions for higher degrees

    # printOutput(F, combinations, x, y)

if __name__ == "__main__":
    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    print('begin')
    # x = [[8,65],
    #     [-44,-51],
    #     [17,-33],
    #     [-75,48],
    #     [18,45],
    #     [-92,-41],
    #     [11,13],
    #     [1,17],
    #     [-80,-64],
    #     [1,-31],
    #     [-40,71],
    #     [28,51],
    #     [-60,-20],
    #     [-71,-71]]
    # r = [39,-84,-18,49,-29,16,73,-98,-57,-12,17,-43,1,-49]
    # y = [x[i][0]*x[i][0]/10 + 5*x[i][1] + x[i][0]*x[i][1]/6 + r[i] for i in range(len(x))]
    # degrees = 3
    # execute(x, y, degrees)
    
    x_train, t_train, x_test, t_test = mnist.load()
    degrees = 2
    execute(x_train[0:20], t_train[0:20], degrees)

    # TODO: how to handle missing data
    print('end')
