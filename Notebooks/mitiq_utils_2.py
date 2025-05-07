import functools
import cirq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
import qiskit
from mitiq import zne, pec
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_global
from mitiq.lre import execute_with_lre
from backend_utils_updated_3 import get_backend, get_circuit, get_computer_graph


def get_cnot_error(edge: Tuple[int, int]) -> float:
    """
    Return a default CNOT error probability for a given edge.
    """
    # Default value when using fake/simulated backends
    return 0.01


def get_cnot_representation(edge: Tuple[int, int]) -> pec.OperationRepresentation:
    """
    Create an OperationRepresentation for a CNOT gate with local depolarizing noise.
    """
    circuit = cirq.Circuit(
        cirq.CNOT(
            cirq.NamedQubit(f"q_{edge[0]}"),
            cirq.NamedQubit(f"q_{edge[1]}")
        )
    )
    noise_level = 1 - np.sqrt(1 - get_cnot_error(edge))
    return pec.represent_operation_with_local_depolarizing_noise(
        circuit,
        noise_level=noise_level,
    )


def get_representations(computer_graph: 'nx.Graph') -> List[pec.OperationRepresentation]:
    """
    Generate OperationRepresentations for each edge in the connectivity graph.
    """
    return [get_cnot_representation(edge) for edge in computer_graph.edges()]


def run_zne_experiment(
    circuit: qiskit.QuantumCircuit,
    executor: Callable,
    scale_factors: List[float],
    fold_method: Callable = fold_global,
    factory: Optional[RichardsonFactory] = None
) -> float:
    """
    Execute Zero Noise Extrapolation (ZNE) on a circuit.

    Args:
        circuit: The circuit to mitigate.
        executor: A function taking a circuit and returning expectation.
        scale_factors: Noise scaling factors.
        fold_method: Noise folding function.
        factory: RichardsonFactory (optional).

    Returns:
        Mitigated expectation value.
    """
    if factory is None:
        factory = RichardsonFactory(scale_factors=scale_factors)
    return zne.execute_with_zne(
        circuit,
        executor,
        scale_noise=fold_method,
        factory=factory
    )


def run_pec_experiment(
    circuit: qiskit.QuantumCircuit,
    executor: Callable,
    representations: List[pec.OperationRepresentation],
    num_samples: int,
    random_state: Optional[int] = None
) -> float:
    """
    Execute Probabilistic Error Cancellation (PEC) on a circuit.

    Args:
        circuit: The circuit to mitigate.
        executor: A function taking a circuit and returning expectation.
        representations: OperationRepresentations for noise.
        num_samples: Number of samples for mitigation.
        random_state: Seed for sampling.

    Returns:
        Mitigated expectation value.
    """
    return pec.execute_with_pec(
        circuit,
        executor,
        representations=representations,
        num_samples=num_samples,
        random_state=random_state
    )


def run_lre_experiment(
    circuit: qiskit.QuantumCircuit,
    executor: Callable,
    polynomial_degree: int,
    fold_multiplier: float
) -> float:
    """
    Execute Linear Response Extrapolation (LRE) on a circuit.

    Args:
        circuit: The circuit to mitigate.
        executor: A function taking a circuit and returning expectation.
        polynomial_degree: Degree of polynomial fit.
        fold_multiplier: Factor by which to fold noise.

    Returns:
        Mitigated expectation value.
    """
    return execute_with_lre(
        circuit,
        executor,
        polynomial_degree,
        fold_multiplier
    )


def run_experiment(
    method: str,
    executor: Callable,
    circuit_type: str,
    n_qubits: int,
    depth: int,
    seed: int,
    backend_name: str = "Generic",
    use_ideal: bool = False,
    scale_factors: List[float] = [1.0, 2.0],
    fold_method: Callable = fold_global,
    pec_num_samples: int = 100,
    lre_degree: int = 1,
    lre_fold: float = 2.0,
    computer_layout: str = "chain",
    random_state: Optional[int] = None
) -> float:
    """
    Generic runner for mitigation experiments ('zne', 'pec', 'lre').

    Args:
        method: Mitigation method.
        executor: Execution function.
        circuit_type: Benchmark circuit type.
        n_qubits: Number of qubits.
        depth: Circuit depth.
        seed: Random seed.
        backend_name: Fake backend key.
        use_ideal: Use ideal simulator if True.
        scale_factors: ZNE scale factors.
        fold_method: ZNE fold method.
        pec_num_samples: PEC sample count.
        lre_degree: LRE polynomial degree.
        lre_fold: LRE fold multiplier.
        computer_layout: Layout for connectivity graph.
        random_state: Seed for PEC.

    Returns:
        Mitigated expectation value.
    """
    # Prepare backend and circuit
    backend = get_backend(use_ideal, backend_name, n_qubits)
    circuit, _ = get_circuit(circuit_type, n_qubits, depth, seed)
    # Build executor partial
    exec_fn = functools.partial(executor, backend=backend)

    method = method.lower()
    if method == "zne":
        return run_zne_experiment(
            circuit,
            exec_fn,
            scale_factors,
            fold_method,
            RichardsonFactory(scale_factors=scale_factors)
        )
    if method == "pec":
        graph = get_computer_graph(n_qubits, layout_type=computer_layout, backend_name=backend_name)
        reps = get_representations(graph)
        return run_pec_experiment(circuit, exec_fn, reps, pec_num_samples, random_state)
    if method == "lre":
        return run_lre_experiment(circuit, exec_fn, lre_degree, lre_fold)
    raise ValueError(f"Unsupported method: {method}")


def test_depths(
    method: str,
    executor: Callable,
    circuit_type: str,
    n_qubits: int,
    depths: List[int],
    seeds: List[int],
    **kwargs
) -> Dict[int, float]:
    """
    Benchmark a mitigation method over various depths.
    Returns a mapping: depth -> mean mitigated value.
    """
    results: Dict[int, float] = {}
    for d in depths:
        vals = [run_experiment(method, executor, circuit_type, n_qubits, d, s, **kwargs) for s in seeds]
        results[d] = float(np.mean(vals))
    return results


def test_widths(
    method: str,
    executor: Callable,
    circuit_type: str,
    widths: List[int],
    depth: int,
    seeds: List[int],
    **kwargs
) -> Dict[int, float]:
    """
    Benchmark a mitigation method over different qubit counts.
    Returns a mapping: width -> mean mitigated value.
    """
    results: Dict[int, float] = {}
    for w in widths:
        vals = [run_experiment(method, executor, circuit_type, w, depth, s, **kwargs) for s in seeds]
        results[w] = float(np.mean(vals))
    return results


def still_useful(results: Dict[int, float], threshold: float) -> Dict[int, float]:
    """
    Filter results that meet the threshold criterion.
    """
    return {param: val for param, val in results.items() if abs(val) <= threshold}


def plot_heatmap(
    matrix: np.ndarray,
    x_labels: List[int],
    y_labels: List[int],
    title: str
) -> plt.Figure:
    """
    Plot a heatmap of values.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect='auto')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Width')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig