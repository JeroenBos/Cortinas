from threading import Thread
from queue import Queue, Empty
from operator import itemgetter


class ParallelMap:

    _error = object()
    _queue = Queue()
    _threads = []

    @staticmethod
    def set_num_threads(count):
        assert count >= 1
        while len(ParallelMap._threads) > count:
            ParallelMap._threads.pop()
        while len(ParallelMap._threads) < count:
            thread = Thread(target=ParallelMap.worker, args=(ParallelMap._queue,))
            thread.setDaemon(True)
            ParallelMap._threads.append(thread)

    @staticmethod
    def worker(q: Queue):
        try:
            while True:
                t = q.get(block=False)
                element, f, self, i = t
                self.local_worker(f, element, i)
                q.task_done()
        except Empty:
            return

    # noinspection PyBroadException
    def local_worker(self, f, element, i):
        try:
            result = f(element)
        except Exception:
            self._set_failed_result(i)
        else:
            self.__results.put((result, i))

    def _set_failed_result(self, i):
        self.__results.put((ParallelMap._error, i))

    def __init__(self, sequence, f, num_threads=None):
        if num_threads is not None:
            ParallelMap.set_num_threads(num_threads)

        self.__results = Queue()

        sequence_length = 0
        for element in sequence:
            ParallelMap._queue.put((element, f, self, sequence_length))
            sequence_length += 1

        self.__expected_length = sequence_length

        for thread in ParallelMap._threads:
            thread.run()

    @property
    def results(self):
        ParallelMap._queue.join()  # TODO: this assumes there is only one parallel map being computed at the same time
        assert self.__expected_length == self.__results.qsize()

        return [result for result, i in sorted(list(self.__results.queue), key=itemgetter(1))]


ParallelMap.set_num_threads(4)
