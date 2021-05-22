import threading
import time
from queue import Queue
import copy


def thread_job():
    # print("this is an added threading, number is %s"% threading.current_thread())
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finished\n")


def thread_job2():
    print("T2 start\n")
    print("T2 finished\n")


def main():
    added_thread = threading.Thread(target=thread_job, name="T1")
    added_thread2 = threading.Thread(target=thread_job2(), name="T2")
    added_thread.start()
    # added_thread.join()
    added_thread2.start()
    added_thread2.join()
    print("all done\n")
    # print(threading.active_count())
    # print(threading.enumerate())
    # print(threading.current_thread())


def job(l, q):
    data = [x ** 2 for x in l]
    q.put(data)


def job_2(l, q):
    q.put(sum(l))


def multithreading():
    # 用来存放每个线程返回的结果。
    q = Queue()
    threads = []
    data = [[1, 2, 3], [3, 4, 5], [4, 4, 4], [5, 5, 5]]

    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q))
        t.start()
        threads.append(t)
    # 并发
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)


def multithreading_2(l):
    q = Queue()
    threads = []
    for i in range(4):
        t = threading.Thread(target=job_2, args=(copy.copy(l), q),
                             name="T%i"% i)
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    total = 0
    for _ in range(4):
        total += q.get()
    print(total)


def normal(l):
    total = sum(l)
    print(total)


if __name__ == '__main__':
    # main()
    # multithreading()
    l = list(range(1000000))
    s_t = time.time()
    normal(l*4)
    print("normal", time.time()-s_t)

    s_t = time.time()
    multithreading_2(l)
    print("multi", time.time()-s_t)