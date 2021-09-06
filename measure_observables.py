#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reads in VQE parameters, prepares the state, and measures all observables

import numpy as np
import logging
import functools
import pickle           # for saving a dictionary

from qiskit import *

# local imports
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms.minimum_eigen_solvers import VQE
from qiskit.aqua.components.initial_states import Zero, VarFormBased
from qiskit.aqua.operators.legacy import TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.list_ops import ListOp
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel

from hamiltonian import get_matrix, pauli_operator, h_even_fname, h_odd_fname, h_full_op
from run_vqe import initial_params
from interpolators import get_interp_op_list
from schwinger_ansatz import SchwingerAnsatz
from name_conventions import opt_params_fname, measured_coeffs_fname, aux_ops_fname

from callbacks import simple_callback, analyze_optimizer

# setup logging
FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT, level=logging.ERROR)

sim_name = 'h_even_noisy_L2_J0p833_w0p500_m-0p750'
nshots = 1024*8
nsamples = 100

# takes in results from a single circuit
# with memory=True, and estimates
# the uncertanities on the probability of each outcome
# (currently with bootstrap)
# returns outcome:(pba, std) dictionary
def compute_stats(res, invert_qubit_order = True):
    # check that there is a single circuit in the results
    # (ambiguous otherwise)
    assert(len(res.results)==1)
    # generates all possible outcomes given the number of qubits
    def _generate_bitstrings(nqubits, invert_qubit_order):
        # (recursive, modifies all_strings in place)
        def _generate_bitstrings_rec(nqubits, all_strings, a_string, irecur):
            # base
            if irecur == nqubits:
                all_strings.append(''.join([bit for bit in a_string]))
                return
            # append 0
            _generate_bitstrings_rec(nqubits, all_strings, a_string + ['0'], irecur + 1)
            # append 1
            _generate_bitstrings_rec(nqubits, all_strings, a_string + ['1'], irecur + 1)
        all_strings = []
        _generate_bitstrings_rec(nqubits, all_strings, [], 0)
        if (invert_qubit_order):
        # pesky Qiskit messes up qubit ordering... this may or may not be necessary to translate the results...
            all_strings = [''.join(reversed(bitstring)) for bitstring in all_strings]
        return all_strings
    # bootstrap specific?
    # given ensembles of outcomes and a particular outcome
    # calculates the statistics of that outcome:
    # returns outcome:(pba, std) estimates for the outcome
    def _calc_outcome_stats(ensembles, nshots, outcome):
        cts = np.count_nonzero(ensembles==outcome, axis=0)
        pba = np.mean(cts)/nshots
        std = np.std(cts/nshots, ddof = 1) # use unbiased estimator
        return (pba, std)
    nqubits = len(list(opt_res.get_counts(0).keys())[0]) # nb if bitstring wasn't measured, qiskit won't return it all :o
    outcomes = _generate_bitstrings(nqubits, invert_qubit_order)
    mem = res.get_memory(0)
    nshots = sum(list(res.get_counts(0).values()))
    nens = nshots # choose number of ensemles = number of samples
    nsam = nshots
    ensembles = np.random.choice(mem, (nens, nsam))
    stats = map(lambda outcome: (outcome, _calc_outcome_stats(ensembles, nshots, outcome)), outcomes)
    return dict(stats)


# given a hamiltonian and a set of interpolators,
# makes a list of operators to evaluate with evaluate_gap
def construct_aux_ops(interp_op_list, h_full_op):
    # O_i \in interp_op_list, i, j = 0, ... , N - 1
    #
    # N_ij, C_ij, H \in aux_ops, a 1D list.
    #
    # N_ij = O_i O_j^\dagger,   indexed i + N*j
    # C_ij = O_i H O_j^\dagger, indexed i + N*j + N^2
    # H is indexed N^2 * 2
    n_ops = len(interp_op_list)
    n_ops_sq = n_ops**2
    aux_ops = np.zeros(2 * n_ops_sq+1, dtype=object)
    for i in range(n_ops):
        op_i = interp_op_list[i]
        for j in range(n_ops):
            op_j = interp_op_list[j]
            # NEED TO USE reduce() TO SIMPLIFY OPERATOR AFTER COMPOSITIONS
            aux_ops[i + n_ops * j] = ((~op_i)@(h_full_op@op_j)).reduce().to_legacy_op()
            aux_ops[i + n_ops * j + n_ops**2] = ((~op_i)@op_j).reduce().to_legacy_op()
    aux_ops[2*n_ops_sq] = h_full_op
    return aux_ops

if __name__ == '__main__':
    print("Loading variational parameters...")
    opt_params = np.loadtxt(opt_params_fname(sim_name))
    var_form_depth = int(len(opt_params)/3)
    print("Parsing & logging arguments...")
    h_operator = h_full_op
    backend_sim = Aer.get_backend("qasm_simulator")
    print("Loading IBMQ account...")
    provider=IBMQ.load_account()
    # configure noise model
    noise_backend = None
    noise_model = None
    coupling_map = None
    basis_gates = None
    print("Using a noise model...")
    noise_backend = provider.get_backend('ibmqx2')
    noise_model = NoiseModel.from_backend(noise_backend)
    coupling_map = noise_backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    print("Creating a variational form and an optimizer instance...")
    num_qubits = 3
    var_form = SchwingerAnsatz(num_qubits, var_form_depth)
    optimizer = qiskit.aqua.components.optimizers.SPSA()
    print("Load interpolating operators and constructing observables to be measured...")
    print("Creating a simulation instance...")
    n_ops = 2                                   # number of interpolators
    interp_ops = get_interp_op_list(n_ops)
    aux_ops = construct_aux_ops(interp_ops, h_full_op)
    print("Creating a simulation instance...")
    initial_point = initial_params
    group_paulis = True
    expectation = PauliExpectation(group_paulis)
    max_evals_grouped = 1
    sim = VQE(
            h_operator,
            var_form,
            optimizer,
            initial_point,
            expectation,
            max_evals_grouped,
            aux_ops,
            )
    my_quantum_instance = QuantumInstance(backend_sim,
                                          shots=nshots,
                                          # noise model
                                          basis_gates=basis_gates,
                                          coupling_map=coupling_map,
                                          noise_model=noise_model,
                                          # error mitigation
                                          measurement_error_mitigation_cls=None,
                                          cals_matrix_refresh_period=None,
                                          measurement_error_mitigation_shots=None
                                          )
    sim._quantum_instance = my_quantum_instance
    print("Building the optimal circuit...")
    sim._ret = {}
    sim._ret['opt_params'] = opt_params
    opt_circ = sim.get_optimal_circuit()
    print("Evaluating auxiliary operators...")
    mean = np.zeros(shape=(nsamples, len(aux_ops)))
    errs = np.zeros(shape=(nsamples, len(aux_ops)))
    for i in range(nsamples):
        sim._eval_aux_ops()
        print("Done!")
        print(sim._ret['aux_ops'])
        print(sim._ret['aux_ops_err'])
        mean[i] = np.array(sim._ret['aux_ops'])
        errs[i] = np.array(sim._ret['aux_ops_err'])
    print("Saving to %s" % aux_ops_fname(sim_name))
    np.savetxt(aux_ops_fname(sim_name),
               np.array((sim._ret['aux_ops'], sim._ret['aux_ops_err'])).T)
    np.savetxt("check_mean_%s.txt" % (sim_name, ),
            mean)
    np.savetxt("check_err_%s.txt" % (sim_name, ),
            errs)
    print("Running evaluatuations of all coefficients in computational basis...")
    sim.quantum_instance.set_config(shots=nshots, memory=True)
    opt_res = sim.quantum_instance.execute(opt_circ)
    c = ClassicalRegister(opt_circ.width(), name='c')
    q = find_regs_by_name(opt_circ, 'q')
    opt_circ.add_register(c)
    opt_circ.barrier(q)
    opt_circ.measure(q, c)
    sim.quantum_instance.set_config(shots=nshots, memory=True)
    opt_res = sim.quantum_instance.execute(opt_circ)
    print("Done. Computing statistics with bootrstrap")
    measured_coeffs = compute_stats(opt_res)
    print(measured_coeffs)
    print("Saving to %s" % measured_coeffs_fname(sim_name))
    with open(measured_coeffs_fname(sim_name), 'wb') as f:
        pickle.dump(measured_coeffs, f)
