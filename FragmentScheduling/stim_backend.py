import stim
from .extraction import extract_schedule

__all__ = ["StimExtractor", "schedule_to_stim"]


class _Allocator:
    def __init__(self, start = 0, reuse = "deferred"):
        self.next_label = start
        self.reuse = reuse
        # recyclable right now
        self.free = []  
        # freed this column, ready at the next               
        self.pending = []              

    def take(self):
        if self.reuse != "off" and self.free:
            return self.free.pop()
        q = self.next_label
        self.next_label += 1
        return q

    def release(self, q):
        if self.reuse == "off":
            return
        (self.free if self.reuse == "immediate" else self.pending).append(q)

    def step(self):
        self.free.extend(self.pending)
        self.pending.clear()


class StimExtractor:
    def __init__(self, n_inputs, reuse = "deferred"):
        self.lines = []
        self.allocator = _Allocator(n_inputs, reuse=reuse)
        self.history = []
        self.n_inputs = n_inputs
        self.n_measurements = 0
        self.postselected = []
        self.measured = []

    def _op(self, name, *targets):
        self.lines.append((name, list(targets)))

    def _fresh(self, spider_type):
        q = self.allocator.take()
        self._op("R" if spider_type == "Z" else "RX", q)
        return q

    def _cnot(self, a, b, spider_type):
        if spider_type == "Z":
            self._op("CX", a, b)
        else:
            self._op("CX", b, a)

    def _measure(self, q, basis, expect="0"):
        self._op("MX" if basis == "X" else "M", q)
        self.allocator.release(q)
        self.postselected.append(self.n_measurements)
        self.measured.append((q, basis, expect))
        self.n_measurements += 1

    def add_gate(self, name, qubit):
        if name.upper() in ("HAD", "H"):
            self._op("H", qubit)
        else:
            raise ValueError(f"unsupported gate {name!r}")

    def apply_spider(self, qubit, spider_type, phase):
        k = int(round(float(phase) * 2)) % 4
        if k == 0:
            return
        table = {"Z": {1: "S", 2: "Z", 3: "S_DAG"},
                 "X": {1: "SQRT_X", 2: "X", 3: "SQRT_X_DAG"}}
        self._op(table[spider_type][k], qubit)

    def split_zero_to_many(self, n_outputs, spider_type, phase=0):
        hub = self.allocator.take()
        # the hub is prepared in the OTHER basis from the rest: |+> for a Z
        # cat, |0> for an X one
        self._op("RX" if spider_type == "Z" else "R", hub)
        qubits = [hub]
        for _ in range(n_outputs - 1):
            qubits.append(self._fresh(spider_type))
        for a, b in zip(qubits, qubits[1:]):
            self._cnot(a, b, spider_type)
        self.apply_spider(hub, spider_type, phase)
        self.history.append(("cat_state", tuple(qubits), spider_type, phase))
        return qubits

    def merge_many_to_one(self, qubits, spider_type):
        qubits = list(qubits)
        for i in range(len(qubits) - 1):
            self._cnot(qubits[i + 1], qubits[i], spider_type)
            self._measure(qubits[i], "Z" if spider_type == "Z" else "X")
        self.history.append(("merge", tuple(qubits), spider_type))
        return qubits[-1]

    def split_one_to_many(self, pivot, n_outputs, spider_type):
        qubits = [pivot]
        for _ in range(n_outputs - 1):
            qubits.append(self._fresh(spider_type))
        for a, b in zip(qubits, qubits[1:]):
            self._cnot(a, b, spider_type)
        self.history.append(("split", tuple(qubits), spider_type))
        return qubits

    def postselect(self, qubit, basis):
        self._measure(qubit, "X" if basis == "+" else "Z")

    def next_step(self):
        self._op("TICK")
        self.allocator.step()

    def get_circuit(self, detectors=False):
        c = stim.Circuit()
        for name, targets in self.lines:
            c.append(name, targets)
        if detectors:
            for i in self.postselected:
                back = i - self.n_measurements
                c.append("DETECTOR", [stim.target_rec(back)])
        return c

    def draw(self):
        print(self.get_circuit())


def schedule_to_stim(G, schedule, detectors = False, reuse = "deferred", **kw):
    # reuse: "off" | "deferred" | "immediate"
    # "deferred" is the one that agrees with occupancy_profile
    result = extract_schedule(G, schedule, extractor_cls = lambda n: StimExtractor(n, reuse = reuse), **kw)
    circuit = result.extractor.get_circuit(detectors = detectors)
    return circuit, result
