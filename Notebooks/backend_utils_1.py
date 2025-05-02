import networkx as nx
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import (
    FakeJakartaV2,
    FakeLimaV2,
    FakeKolkataV2
)
from qiskit.providers.fake_provider import GenericBackendV2
import qiskit
from typing import List, Tuple
from mitiq.benchmarks import generate_rb_circuits, generate_mirror_circuit, generate_ghz_circuit

# Supported fake devices (excluding unsupported runtime-only fakes)
FAKE_DEVICES = {
    "FakeJakartaV2": FakeJakartaV2,
    "FakeLimaV2": FakeLimaV2,
    "FakeKolkataV2": FakeKolkataV2,
    
    "Generic": GenericBackendV2,
}

# Optional device layouts for physical qubit mapping
DEVICE_LAYOUTS = {
    "FakeLimaV2": [0, 1, 3, 4, 2],
    "FakeKolkataV2": [
        0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25,
        22, 19, 16, 14, 11, 8, 5, 3, 2
    ],
    # Add more layouts from qiskit.providers.fake_provider as needed
}

def get_backend(use_ideal: bool = True,
                backend_name: str = "FakeJakartaV2",
                n_qubits: int = 5):
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

def execute(
    circuits: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit],
    backend,
    shots: int,
    correct_bitstring: str,
    verbose: bool,
    ) -> List[float]:
    """Executes the input circuit(s) and returns ⟨A⟩, where A = |correct_bitstring⟩⟨correct_bitstring| for each circuit."""

    if not isinstance(circuits, list):
        circuits = [circuits]
    if verbose:
        # Calculate average number of CNOT gates per circuit.
        print(f"Executing {len(circuits)} circuit(s) on {backend}.")
        print(f"Average cnot count in circuits: {get_avg_cnot_count(circuits)}")

    # Store all circuits to run in list to be returned.
    to_run: list[qiskit.QuantumCircuit] = []

    for circuit in circuits:
        circuit_to_run = circuit.copy()
        circuit_to_run.measure_all()
        to_run.append(
            qiskit.transpile(
                circuit_to_run,
                backend=backend,
                initial_layout= get_phys_qubits(circuit.num_qubits),
                optimization_level=0,  # Otherwise RB circuits are simplified to empty circuits.
            )
        )

    if verbose:
        # Calculate average number of CNOT gates per compiled circuit.
        print(f"Average cnot count in compiled circuits: {get_avg_cnot_count(to_run)}")

    # Run and get counts.
    job = backend.run(
        to_run,
        # Reset qubits to ground state after each sample.
        init_qubits=True,
        shots=shots,
    )
    # IBMQ uses online queue for processing jobs.
    # if verbose and not use_noisy_simulator:
    #     time.sleep(3)
    #     while not job.in_final_state():
    #         print(f"Queue position: {job.queue_position()}")
    #         time.sleep(verbose_update_time)
    #     print()

    # print(f"Correct bitstring: {correct_bitstring}")
    if len(circuits) == 1:
        return [job.result().get_counts().get(correct_bitstring, 0.0) / shots]
    return [
        count.get(correct_bitstring, 0.0) / shots for count in job.result().get_counts()
    ]
    
def get_phys_qubits(n_qubits: int,
                    backend_name: str = "FakeLimaV2") -> List[int]:
    """
    Maps logical to physical qubit indices for a given backend layout.

    Args:
        n_qubits (int): Number of logical qubits to map.
        backend_name (str): Backend key for layout lookup.

    Returns:
        List[int]: Physical qubit indices.
    """
    layout = DEVICE_LAYOUTS.get(backend_name)
    if layout is None or n_qubits > len(layout):
        # Fallback to linear mapping
        return list(range(n_qubits))
    return layout[:n_qubits]


def get_computer_graph(n_qubits: int,
                       layout_type: str = "chain",
                       backend_name: str = None,
                       return_pattern: bool = False):
    """
    Builds a directed connectivity graph for qubits and an optional RB pattern.

    Args:
        n_qubits (int): Number of qubits.
        layout_type (str): 'chain' for linear chain or 'hardware' for physical layout.
        backend_name (str): Required if using 'hardware' layout.
        return_pattern (bool): Whether to return the RB 2-qubit pattern.

    Returns:
        networkx.DiGraph or (graph, pattern)
    """
    if layout_type == "hardware":
        if not backend_name:
            raise ValueError("Must specify backend_name for hardware layout.")
        qubits = get_phys_qubits(n_qubits, backend_name)
    else:
        qubits = list(range(n_qubits))

    graph = nx.DiGraph()
    pattern: List[List[int]] = []

    for i in range(len(qubits) - 1):
        q1, q2 = qubits[i], qubits[i + 1]
        graph.add_edge(q1, q2)
        graph.add_edge(q2, q1)
        if i % 2 == 0:
            pattern.append([q1, q2])

    if layout_type == "chain" and n_qubits % 2 == 1:
        pattern.append([qubits[-1]])

    return (graph, pattern) if return_pattern else graph


def get_circuit(circuit_type: str,
                n_qubits: int,
                depth: int,
                seed: int) -> Tuple[qiskit.QuantumCircuit, str]:
    """
    Create a benchmark circuit and return it along with the correct bitstring.

    Supports:
      - 'rb': Randomized benchmarking circuit
      - 'mirror': Mirror circuit
      - 'long cnot': Chain of CNOTs
      - 'ghz': GHZ state
    """
    typ = circuit_type.lower()
    if typ == "rb":
        circuit = generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            seed=seed,
            return_type="qiskit"
        )[0]
        return circuit, "00"

    elif typ == "mirror":
        computer = get_computer_graph(n_qubits)
        circuit, correct = generate_mirror_circuit(
            nlayers=depth,
            two_qubit_gate_prob=1.0,
            connectivity_graph=computer,
            two_qubit_gate_name="CNOT",
            seed=seed,
            return_type="qiskit",
        )
        return circuit, "".join(map(str, correct[::-1]))

    elif typ == "long cnot":
        circuit = qiskit.QuantumCircuit(n_qubits)
        circuit.x(0)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        return circuit, "1" * (n_qubits - 1)

    elif typ == "ghz":
        circuit = generate_ghz_circuit(
            n_qubits=n_qubits,
            return_type="qiskit"
        )
        return circuit, "0" * n_qubits

    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")


def get_num_cnot_count(circuit: qiskit.QuantumCircuit) -> int:
    """Return the number of CNOT gates in the circuit."""
    return circuit.count_ops().get('cx', 0)


def get_oneq_count(circuit: qiskit.QuantumCircuit) -> int:
    """Return the number of single-qubit gates in the circuit."""
    ops = circuit.count_ops()
    single = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']
    return sum(ops.get(g, 0) for g in single)


def get_avg_cnot_count(circuits: List[qiskit.QuantumCircuit]) -> float:
    """Average CNOT count over a list of circuits."""
    return sum(get_num_cnot_count(c) for c in circuits) / max(len(circuits), 1)
