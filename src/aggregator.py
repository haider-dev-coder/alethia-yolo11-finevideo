import math
from collections import Counter


class SecondAggregator:
    """Per-second aggregation. Frames must arrive in timestamp order.

    add_frame(ts, objects, people) -> list of finished second-rows
    finish() -> remaining rows (call once at end of video)
    """

    def __init__(self, video_id):
        self.video_id = video_id
        self.t0 = None
        self.cur = None          # current second index
        self.n = 0               # frames in current second
        self.obj_counts = Counter()
        self.people = []
        self.next_out = 0        # next second index to emit

    def _row(self, sec, objs, people):
        return {"video": self.video_id, "second": sec,
                "objects": sorted(objs), "people": people}

    def _close(self):
        if self.cur is None or self.n == 0:
            return []
        rows = []
        while self.next_out < self.cur:      # gap seconds with no frames
            rows.append(self._row(self.next_out, [], 0))
            self.next_out += 1
        objs = [o for o, c in self.obj_counts.items() if 2 * c >= self.n]
        p = sorted(self.people)
        m = len(p)
        med = p[m // 2] if m % 2 else (p[m // 2 - 1] + p[m // 2]) // 2
        rows.append(self._row(self.cur, objs, med))
        self.next_out = self.cur + 1
        self.obj_counts = Counter()
        self.people = []
        self.n = 0
        return rows

    def add_frame(self, ts, objects, people):
        if self.t0 is None:
            self.t0 = ts
        sec = max(0, int(math.floor(ts - self.t0)))
        rows = []
        if self.cur is not None and sec != self.cur:
            rows = self._close()
        self.cur = sec
        self.n += 1
        for o in set(objects):
            self.obj_counts[o] += 1
        self.people.append(people)
        return rows

    def finish(self):
        return self._close()