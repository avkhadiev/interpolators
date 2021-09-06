#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from qiskit import *

# local import
from qiskit_aqua.qiskit.aqua import QuantumInstance

nshots = 1024

if __name__ == '__main__':
    IBMQ.load_account()
    backend_sim = Aer.get_backend('qasm_simulator')
    my_quantum_instance = QuantumInstance(backend_sim, nshots)
