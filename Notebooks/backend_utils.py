import networkx as nx
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import (
    FakeJakartaV2, FakeLimaV2, FakeKolkataV2) #FakeSherbrookeV2
#)
from qiskit.providers.fake_provider import GenericBackendV2

# Supported fake devices
FAKE_DEVICES = {
    "FakeJakartaV2": FakeJakartaV2,
    "FakeLimaV2": FakeLimaV2,
    "FakeKolkataV2": FakeKolkataV2,
    "Generic": GenericBackendV2,
}

# Optional device layouts
DEVICE_LAYOUTS = {
    "FakeLimaV2": [0, 1, 3, 4, 2],
    "FakeKolkataV2": [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2],
  
}

def get_backend(use_ideal=True, backend_name="FakeJakartaV2", n_qubits=5):
    """
    Returns a backend for running circuits.

    Args:
        use_ideal (bool): If True, returns an ideal AerSimulator.
        backend_name (str): Fake backend name if use_ideal=False.
        n_qubits (int): Only used for GenericBackendV2.

    Returns:
        Qiskit Backend instance.
    """
    if use_ideal:
        return AerSimulator()

    if backend_name not in FAKE_DEVICES:
        raise ValueError(f"Unsupported backend name: {backend_name}")

    backend_class = FAKE_DEVICES[backend_name]
    return backend_class(n_qubits) if backend_name == "Generic" else backend_class()


def get_phys_qubits(n_qubits, backend_name="FakeLimaV2"):
    """
    Maps logical to physical qubit layout (hardware layout or fallback chain).

    Args:
        n_qubits (int): Number of logical qubits to allocate.
        backend_name (str): Backend to map layout from.

    Returns:
        List[int]: Physical qubit indices.
    """
    layout = DEVICE_LAYOUTS.get(backend_name)
    if layout is None or n_qubits > len(layout):
        return list(range(n_qubits))  # fallback to linear
    return layout[:n_qubits]


def get_computer_graph(n_qubits, layout_type="chain", backend_name=None, return_pattern=False):
    """
    Returns a connectivity graph and optionally a randomized benchmarking pattern.

    Args:
        n_qubits (int): Number of qubits.
        layout_type (str): "chain" or "hardware".
        backend_name (str): Required if layout_type is "hardware".
        return_pattern (bool): Also return a 2-qubit RB pattern.

    Returns:
        nx.DiGraph or (graph, pattern)
    """
    if layout_type == "hardware":
        if not backend_name:
            raise ValueError("Must specify backend_name for hardware layout.")
        qubits = get_phys_qubits(n_qubits, backend_name)
    else:
        qubits = list(range(n_qubits))

    graph = nx.DiGraph()
    pattern = []

    for i in range(len(qubits) - 1):
        q1, q2 = qubits[i], qubits[i + 1]
        graph.add_edge(q1, q2)
        graph.add_edge(q2, q1)
        if i % 2 == 0:
            pattern.append([q1, q2])

    if layout_type == "chain" and n_qubits % 2 == 1:
        pattern.append([qubits[-1]])

    return (graph, pattern) if return_pattern else graph
