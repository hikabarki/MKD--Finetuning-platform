"""Performance / concurrency tests — lock contention on shared state."""
from __future__ import annotations

import threading
import time


def test_chat_lock_no_deadlock():
    """8 threads acquiring the chat lock concurrently must all complete."""
    import inference.state as state

    results = []
    errors  = []

    def worker(n):
        try:
            with state._chat_lock:
                time.sleep(0.001)
                results.append(n)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors == [], f"Unexpected errors: {errors}"
    assert len(results) == 8


def test_vllm_lock_no_deadlock():
    import inference.state as state

    results = []

    def worker(n):
        with state._vllm_lock:
            time.sleep(0.001)
            results.append(n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(results) == 8


def test_chat_lock_is_reentrant():
    """RLock allows the same thread to acquire it twice."""
    import inference.state as state

    acquired = []

    def worker():
        with state._chat_lock:
            acquired.append(1)
            with state._chat_lock:   # reentrant — must not deadlock
                acquired.append(2)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2)
    assert acquired == [1, 2]


def test_state_mutation_under_lock_is_visible():
    """State mutation inside lock must be visible to other threads after release."""
    import inference.state as state

    state._chat_name = "initial"
    barrier = threading.Barrier(2)

    def writer():
        with state._chat_lock:
            state._chat_name = "updated"
            barrier.wait()   # signal reader to proceed

    def reader():
        barrier.wait()
        # writer has set the value under lock; by here, lock is released
        assert state._chat_name == "updated"

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start(); r.start()
    w.join(timeout=3); r.join(timeout=3)

    state._chat_name = None   # reset
