from collections import deque

class queue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.popleft()
        else:
            raise IndexError("empty queue")

    def front(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            raise IndexError("empty queue")

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)
    
    
n, k = map(int, input().split())
q = queue()

for i in range(1, n+1):
    q.enqueue(i)

for i in range(k):
    for j in range(k-1):
        q.enqueue(q.front())
        q.dequeue()
    print(q.front(), end=" ")
    q.dequeue()

print(q.front(),end=" ")