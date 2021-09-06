#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Defines SPSA optimizer

from qiskit import *
from qiskit.aqua.components.optimizers import SPSA

max_trials = 500
optimizer = qiskit.aqua.components.optimizers.SPSA(max_trials)

if __name__ == '__main__':
    print(optimizer.setting)

