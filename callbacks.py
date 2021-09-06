#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Defines call-back functions to run in a VQE

import numpy as np

# a callback that can access the intermediate data
# during the optimization.
# Internally, four arguments are provided as follows
# the index of evaluation, parameters of variational form,
# evaluated mean, evaluated standard deviation.
def simple_callback(eval_count, parameter_set, mean, std):
    if eval_count % 1 == 0:
        print('Energy evaluation %s returned %4f +/- %4f' % (eval_count,
            np.real(mean), np.real(std)))

def analyze_optimizer(eval_count, parameter_set, mean, std):
    print('%d %7f %7f' % (eval_count, np.real(mean), np.real(std)))

