import random, math

__author__ = 'samminns'


def main():
    f = open('ex_symb_regdata', 'w')
    f.write("@RELATION extendSymbolicRegression\n\n")
    f.write("@ATTRIBUTE x\n@ATTRIBUTE class{1,2}\n\n@DATA\n")
    for i in range(800):
        x = random.uniform(-100, 100)
        if x > 0:
            res = 1 / x + math.sin(x)
            f.write(str(x) + "\t" + str(res)+"\n")
        elif x <= 0:
            res = 2 * x + math.pow(x, 2) + 3.0
            f.write(str(x) + "\t" + str(res)+"\n")
    f.close()


if __name__ == '__main__':
    main()