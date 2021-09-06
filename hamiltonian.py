#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Defines a Hamiltonian for the VQE

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, identity
from scipy.sparse import block_diag

from qiskit import *
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import MatrixOp, PauliOp, PrimitiveOp
from qiskit.aqua.operators import MatrixOperator, TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.expectations import PauliExpectation

from name_conventions import data_dir_name

# specify parameters
# TODO have this read from a file that is shared with the Mathematica script too,
# so that params reside in a single place
J = 0.8333
w = 1.5
m = 0.0833


# change as necessary
dir_str = 'data_L2_J0p833_w0p500_m-0p750'
h_even_fname = "./%s/hEvenPMatrix_L2_lmd1_1_lmd2_3.mtx" % (dir_str)
h_odd_fname = "./%s/hOddPMatrix_L2_lmd1_1_lmd2_3.mtx" % (dir_str)

def get_matrix( fname = h_even_fname ):
    return csr_matrix(mmread(fname))

def subtract_trace(a_matrix):
    # assumes a sparse a matrix
    assert(a_matrix.shape[0] == a_matrix.shape[1])
    dim = a_matrix.shape[0]
    assert(dim > 0)
    trace = float(a_matrix.diagonal().sum())
    a_matrix_traceless = a_matrix - (trace/float(dim)) * identity(dim)
    return a_matrix_traceless, trace

def pauli_operator(a_matrix):
    # subtract trace and resize to fit qubits snugly
    a_matrix = csr_matrix(a_matrix)
    assert(a_matrix.shape[0] == a_matrix.shape[1])
    dim = a_matrix.shape[0]
    new_dim= int(2**np.ceil(np.log2(dim)))
    a_matrix.resize((new_dim, new_dim))
    # convert to Pauli-grouped operator
    H_MatrixOperator = MatrixOp(a_matrix)
    H_Operator = H_MatrixOperator.to_pauli_op()
    return H_Operator

# get matrices
h_even_mat = get_matrix(h_even_fname)
h_odd_mat = get_matrix(h_odd_fname)
# construct block-diagonal full Hamiltonian
# trace-subtract each subspace separately for consistent results
h_even_mat_traceless, even_trace = subtract_trace(h_even_mat)
h_odd_mat_traceless, odd_trace = subtract_trace(h_odd_mat)
h_full_mat_traceless = block_diag((h_even_mat_traceless, h_odd_mat_traceless), format='csr')
# make operators
h_even_op = pauli_operator(h_even_mat_traceless)
h_odd_op  = pauli_operator(h_odd_mat_traceless)
h_full_op = pauli_operator(h_full_mat_traceless)

if __name__ == '__main__':
    print("grouped Hamiltonians:")
    print("EVEN:")
    print("trace = %f" % even_trace)
    print(h_even_op.__str__())
    print("ODD:")
    print("trace = %f" % odd_trace)
    print(h_odd_op.__str__())
    print("DIRECT SUM:")
    print(h_full_op.__str__())
    print(h_full_mat_traceless)

