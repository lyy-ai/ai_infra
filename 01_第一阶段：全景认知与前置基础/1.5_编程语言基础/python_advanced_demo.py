# 2.1 编程语言基础：Python 进阶演示
#
# 运行：
#   /data/qwen35_env/bin/python 1.5_编程语言基础/python_advanced_demo.py

import functools
import time
from multiprocessing import Pool


def timing(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        print(f"  {fn.__name__}: {(time.perf_counter() - t0) * 1e3:.2f}ms")
        return r
    return wrapper


def squares(n):
    for i in range(n):
        yield i * i


def heavy(x):
    return sum(i * i for i in range(200000))


@timing
def serial(n):
    return [heavy(i) for i in range(n)]


@timing
def parallel(n):
    with Pool(4) as p:
        return p.map(heavy, range(n))


def main():
    print("=== 装饰器计时 ===")
    serial(4)
    parallel(4)

    print("\n=== 生成器（惰性求值）===")
    g = squares(5)
    print("  next:", [next(g) for _ in range(5)])

    print("\n=== pybind11 最小示例（源码，需编译环境）===")
    print("""  #include <pybind11/pybind11.h>
  int add(int a, int b) { return a + b; }
  PYBIND11_MODULE(my_ops, m) { m.def("add", &add); }""")


if __name__ == "__main__":
    main()
