#!/usr/bin/env python
# -*- coding: utf-8 -*-

def opt_params_fname(sim_name):
    return f'opt_params_{sim_name}.txt'

def vqe_log_fname(sim_name):
    return f'vqe_{sim_name}.log'

def measured_coeffs_fname(sim_name):
    return f'measured_coeff_{sim_name}.pkl'

def aux_ops_fname(sim_name):
    return f'aux_ops_{sim_name}.txt'

# converts a real numerical parameter value to a string with nfigs significant
# figures and "." replaced with "p": e.g., 0.08333 becomes "0p083" if nfigs = 2.
def convert_num_to_str(num):
    # nfigs will be 3 to agree with files output with Mathematica script
    outstr = ('%s' % float('%.3g' % num)).replace('.', 'p')
    return outstr

def data_dir_name(L, J, w, m):
    Jstr = convert_num_to_str(J)
    wstr = convert_num_to_str(w)
    mstr = convert_num_to_str(m)
    return "data_L%d_J%s_w%s_m%s" % (L, Jstr, wstr, mstr)
