# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 9:33 AM
# @Author  : Zhong Lei
# @File    : LearnMultiprocess.py
import threading
import multiprocessing
import time
# from queue import Queue

def job(q):
    res = 0
    for _ in range(2):
        for i in range(1000000000):
            res += i
    q.put(res)


def nomal():
    res = 0
    for _ in range(2):
        for i in range(1000000000):
            res += i
    print("nomal:", res)


def multithread():
    q = multiprocessing.Queue()

    td1 = threading.Thread(target=job, args=(q, ))
    td2 = threading.Thread(target=job, args=(q, ))
    td1.start()
    td2.start()
    td1.join()
    td2.join()
    res1 = q.get()
    res2 = q.get()
    print("threading:", res2+res1)


def multicore():
    q = multiprocessing.Queue()
    mp1 = multiprocessing.Process(target=job, args=(q, ))
    mp2 = multiprocessing.Process(target=job, args=(q, ))
    mp1.start()
    mp2.start()
    mp1.join()
    mp2.join()
    res1 = q.get()
    res2 = q.get()
    print("core:", res1+res2)


# 进程池
def job_2(x):
    return x*x


def multipool():
    pool = multiprocessing.Pool(processes=2)
    res = pool.map(job_2, range(10))
    print(res)
    res = pool.apply_async(job_2, (2,))
    print(res.get())
    multi_res = [pool.apply_async(job_2, (i,)) for i in range(10)]
    print([res.get() for res in multi_res])


# 共享内存和锁。
def job_lock(value, num, lock):
    lock.acquire()
    for _ in range(10):
        time.sleep(0.1)
        value.value += num
        print(value.value)
    lock.release()


def multilock():
    value = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    p1 = multiprocessing.Process(target=job_lock, args=(value, 1, lock))
    p2 = multiprocessing.Process(target=job_lock, args=(value, 3, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    # 使用的多进程的queue和现成中的队列不同。
    # st = time.time()
    # nomal()
    # st1 = time.time()
    # print("normal time:", st1 - st)
    #
    # multicore()
    # st2 = time.time()
    # print("multi core:", st2 - st)
    #
    # multithread()
    # st3 = time.time()
    # print("multi thread:", st3 - st)

    # 进程池，将任务分给多个程序
    # multipool()

    # 共享内存和变量命名。
    multilock()