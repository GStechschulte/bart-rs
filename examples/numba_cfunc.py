from numba import cfunc



@cfunc("float64(float64, float64)")
def add(x, y):
    return x + y

def main():
    print(add.ctypes(4.0, 5.0))

if __name__ == "__main__":
    main()