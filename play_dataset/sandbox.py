import numpy as np


class yrange:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return yrange_iter(self.n)


class yrange_iter:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i

        else:
            raise StopIteration()



if __name__ == "__main__":

    a = yrange(10)

    for i in a:
        print(i)
