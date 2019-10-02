# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module provides solvers for the unitary Schrodinger equation.
"""

__all__ = ['sesolve_moving']

import os
from copy import deepcopy
import types
from functools import partial
import numpy as np
import scipy.integrate
from scipy.linalg import norm as la_norm
import qutip.settings as qset
from qutip.qobj import Qobj, isket
from qutip.rhs_generate import rhs_generate
from qutip.solver import Result, Options, config, _solver_safety_check
from qutip.rhs_generate import _td_format_check, _td_wrap_array_str
from qutip.interpolate import Cubic_Spline
from qutip.superoperator import operator_to_vector, vector_to_operator, spre, mat2vec
from qutip.settings import debug
from qutip.cy.spmatfuncs import (cy_expect_psi, cy_ode_rhs,
                                 cy_ode_psi_func_td,
                                 cy_ode_psi_func_td_with_state,
                                 spmvpy_csr)
from qutip.cy.codegen import Codegen
from qutip.cy.utilities import _cython_build_cleanup

from qutip.ui.progressbar import (BaseProgressBar, TextProgressBar)
from qutip.cy.openmp.utilities import check_use_openmp, openmp_components
from qutip.operators import destroy, displace, qeye, commutator
from qutip.tensor import tensor
from qutip.expect import expect
import time

if qset.has_openmp:
    from qutip.cy.openmp.parfuncs import cy_ode_rhs_openmp

if debug:
    import inspect


def sesolve_moving(H, psi0, tlist, e_ops=[], args={}, options=None,
            progress_bar=None,
            _safe_mode=True):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (`psi0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, or a callback function for time-dependent
        Hamiltonians.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.
        Must be empty list operator evolution

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors corresponding to the
        times in `tlist` [if `e_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """
    # check initial state: must be a state vector


    if _safe_mode:
        if not isinstance(psi0, Qobj):
            raise TypeError("psi0 must be Qobj")
        if psi0.isket:
            pass
        elif psi0.isunitary:
            if not e_ops == []:
                raise TypeError("Must have e_ops = [] when initial condition"
                                " psi0 is a unitary operator.")
        else:
            raise TypeError("The unitary solver requires psi0 to be"
                            " a ket as initial state"
                            " or a unitary as initial operator.")
        _solver_safety_check(H, psi0, c_ops=[], e_ops=e_ops, args=args)

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    # convert array based time-dependence to string format
    H, _, args = _td_wrap_array_str(H, [], args, tlist)
    # check for type (if any) of time-dependent inputs
    n_const, n_func, n_str = _td_format_check(H, [])

    if options is None:
        options = Options()

    if (not options.rhs_reuse) or (not config.tdfunc):
        # reset config time-dependence flags to default values
        config.reset()

    # check if should use OPENMP
    check_use_openmp(options)

    if n_func > 0:
        res = _sesolve_list_func_td(H, psi0, tlist, e_ops, args, options,
                                    progress_bar)
        # return res

    elif n_str > 0:
        res = _sesolve_list_str_td(H, psi0, tlist, e_ops, args, options,
                                   progress_bar)

    elif isinstance(H, (types.FunctionType,
                        types.BuiltinFunctionType,
                        partial)):
        res = _sesolve_func_td(H, psi0, tlist, e_ops, args, options,
                               progress_bar)

    else:
        res = _sesolve_const(H, psi0, tlist, e_ops, args, options,
                             progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_list_func_td(H_list, psi0, tlist, e_ops, args, opt,
                          progress_bar):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state or oper
    #
    if psi0.isket:
        initial_vector = psi0.full().ravel()
        oper_evo = False
    elif psi0.isunitary:
        initial_vector = operator_to_vector(psi0).full().ravel()
        oper_evo = True
    else:
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    if opt.moving_mode_indices is None:
        n_modes = len(psi0.dims[0])
        opt.moving_mode_indices = np.arange(n_modes)

    #
    # construct liouvillian in list-function format
    #
    L_list = []
    if not opt.rhs_with_state:
        constant_func = lambda x, y: 1.0
    else:
        constant_func = lambda x, y, z: 1.0

    # add all hamitonian terms to the lagrangian list
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = constant_func

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

        else:
            print(h_spec)
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected callback function)")

        if oper_evo:
            L = -1.0j * spre(h)
        else:
            L = -1j * h
        L_list.append([L.data, h_coeff])

    L_list_and_args = [L_list, args]
    # return L_list_and_args

    processed_com_lists = []
    for com_list in opt.com_lists:
        processed_com_list = []
        for h_spec in com_list:

            if isinstance(h_spec, Qobj):
                h = h_spec
                h_coeff = constant_func

            elif isinstance(h_spec, list):
                h = h_spec[0]
                h_coeff = h_spec[1]

            else:
                print(h_spec)
                raise TypeError("Incorrect specification of time-dependent " +
                                "Hamiltonian (expected callback function)")

            if oper_evo:
                L = 1.0j * spre(h)
            else:
                L = 1j * h
            processed_com_list.append([L.data, h_coeff])
        processed_com_lists.append(processed_com_list)
    opt.processed_com_lists = processed_com_lists

    #
    # setup integrator
    #
    if oper_evo:
        initial_vector = operator_to_vector(psi0).full().ravel()
    else:
        new_psi0 = psi0
        initial_vector = new_psi0.full().ravel()
        if opt.moving_basis:
            initial_mode_displacements = np.zeros(len(opt.moving_mode_indices), dtype=complex)
            for mode_idx in opt.moving_mode_indices:
                identities = [qeye(dim) for dim in psi0.dims[0]]
                a_op_components = identities
                a_op_components[mode_idx] = destroy(psi0.dims[0][mode_idx])
                a_op = tensor(a_op_components)
                a_expect = expect(a_op, psi0)
                initial_mode_displacements[mode_idx] = a_expect
                displacement_components = identities
                displacement_components[mode_idx] = displace(psi0.dims[0][mode_idx], -a_expect)
                displacement_op = tensor(displacement_components)
                new_psi0 = displacement_op * new_psi0
            initial_vector = np.hstack([initial_vector, initial_mode_displacements])
    if not opt.rhs_with_state:
        r = scipy.integrate.ode(psi_list_td)
    else:
        if opt.moving_basis:
            r = scipy.integrate.ode(psi_list_td_with_state_moving_basis)
        else:
            r = scipy.integrate.ode(psi_list_td_with_state)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    if opt.moving_basis:
        r.set_f_params(L_list_and_args, opt, psi0.dims)
    else:
        r.set_f_params(L_list_and_args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, psi0, tlist, e_ops, opt, progress_bar,
                              dims=psi0.dims)


def psi_list_td_with_state_moving_basis(t, y, H_list_and_args, opt, dims):
    t0 = time.time()

    H_list = H_list_and_args[0]
    args = H_list_and_args[1]
    params = args

    n_moving_modes = len(opt.moving_mode_indices)
    n_modes = len(dims[0])
    psi = y[0:-n_modes]
    mode_displacements = y[-n_modes:]
    alpha = mode_displacements[0]
    args = [args] + [list(mode_displacements)]
    ddisplacements = np.zeros(n_modes, dtype=complex)

    for mode_idx in opt.moving_mode_indices:
        # operators = [qeye(dim) for dim in dims[0]]
        # operators[mode_idx] = destroy(dims[0][mode_idx])
        # a_op = tensor(operators)
        # a_op = destroy(params['c_levels'])
        # com0 = commutator(H, a_op)

        # com = params['eps']*1j + params['chi']*(2*alpha*np.abs(alpha)**2 + 4*a*np.abs(alpha)**2 + 2*a.dag()*alpha**2
        #                                       + 2*np.conjugate(alpha)*a*a + 4*alpha*a.dag()*a + 2*a.dag()*a*a)
        # com *= -2*np.pi
        # ddisplacement = 1j * np.sum(np.conjugate(psi) * com * psi)
        # print(com0)
        # print(com)
        # print(H)



        t3 = time.time()
        psi_right = np.zeros(psi.shape[0], dtype=complex)
        # G = opt.com_list[0][0]
        # coeff = opt.com_list[0][1]
        # spmvpy_csr(G.data, G.indices, G.indptr, psi, coeff(t, psi, *args), psi_right)
        # ddisplacement = -np.sum(np.conjugate(psi)*psi_right)
        # print(ddisplacement)

        for n in range(0, len(opt.processed_com_lists[mode_idx])):
            # for n in range(0, 1):
            G = opt.processed_com_lists[mode_idx][n][0]
            coeff = opt.processed_com_lists[mode_idx][n][1]
            spmvpy_csr(G.data, G.indices, G.indptr, psi, coeff(t, psi, *args), psi_right)

        ddisplacement = -np.sum(np.conjugate(psi) * psi_right)

        t4 = time.time()

        t5 = time.time()
        ddisplacements[mode_idx] = ddisplacement
        # displacement = mode_displacements[mode_idx]
        # a_op = destroy(params['c_levels'])
        # H_mode_rotation = 0.5*1j*displacement*np.conjugate(ddisplacement) + 1j*np.conjugate(ddisplacement)*a_op
        # H_mode_rotation += H_mode_rotation.dag()
        # H_rotation += H_mode_rotation
        t6 = time.time()
        # H_rotation += displace()
    # print(ddisplacements)
    # print(H_rotation)
    t7 = time.time()
    # dpsi0 = -1j*H*psi
    # dpsi1 = -1j*H_rotation*psi



    # print(H)
    # H = H.data
    # dpsi = np.zeros(psi.shape[0], dtype=complex)
    # spmvpy_csr(H.data, H.indices, H.indptr, psi, 1.0, dpsi)
    # dpsi = -1j*dpsi

    # dalpha0 = np.sum(np.conjugate(dpsi0)*a_op*psi) + np.sum(np.conjugate(psi)*a_op*dpsi0)
    # dalpha1 = np.sum(np.conjugate(dpsi1)*a_op*psi) + np.sum(np.conjugate(psi)*a_op*dpsi1)

    # H_rotation *= -dalpha0/dalpha1
    # ddisplacements *= -dalpha0/dalpha1

    # H_total = H + H_rotation
    # dpsi = -1j*H_total*psi

    args += [ddisplacements]
    dpsi = np.zeros(psi.shape[0], dtype=complex)
    for n in range(0, len(H_list)):
        # for n in range(0, 1):
        H = H_list[n][0]
        H_td = H_list[n][1]
        spmvpy_csr(H.data, H.indices, H.indptr, psi, H_td(t, psi, *args), dpsi)

    t8 = time.time()

    # print(ddisplacement, ddisplacement-dalpha0, ddisplacement+dalpha1)


    dy = np.hstack([dpsi, ddisplacements])

    total = time.time() - t0

    # print((t4-t3)/total, (t6-t5)/total, (t8-t7)/total, 'time = ', total)

    return dy


#
# evaluate dpsi(t)/dt according to the master equation using the
# [Qobj, function] style time dependence API
#
def psi_list_td(t, psi, H_list_and_args):
    H_list = H_list_and_args[0]
    args = H_list_and_args[1]

    H = H_list[0][0]
    H_td = H_list[0][1]
    out = np.zeros(psi.shape[0], dtype=complex)
    spmvpy_csr(H.data, H.indices, H.indptr, psi, H_td(t, args), out)
    for n in range(1, len(H_list)):
        #
        # args[n][0] = the sparse data for a Qobj in operator form
        # args[n][1] = function callback giving the coefficient
        #
        H = H_list[n][0]
        H_td = H_list[n][1]
        spmvpy_csr(H.data, H.indices, H.indptr, psi, H_td(t, args), out)

    return out


def psi_list_td_with_state(t, psi, H_list_and_args):
    start_time = time.time()

    H_list = H_list_and_args[0]
    args = [H_list_and_args[1]]

    H = H_list[0][0]
    H_td = H_list[0][1]
    out = np.zeros(psi.shape[0], dtype=complex)
    spmvpy_csr(H.data, H.indices, H.indptr, psi, H_td(t, psi, *args), out)
    for n in range(1, len(H_list)):
        #
        # args[n][0] = the sparse data for a Qobj in operator form
        # args[n][1] = function callback giving the coefficient
        #
        H = H_list[n][0]
        H_td = H_list[n][1]
        spmvpy_csr(H.data, H.indices, H.indptr, psi, H_td(t, psi, *args), out)

    # print(time.time()-start_time)

    return out


#
# evaluate dpsi(t)/dt [not used. using cython function is being used instead]
#
def _ode_psi_func(t, psi, H):
    return H * psi


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians.
#
def _sesolve_func_td(H_func, psi0, tlist, e_ops, args, opt, progress_bar):
    """!
    Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """
    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state or oper
    #
    if psi0.isket:
        oper_evo = False
    elif psi0.isunitary:
        oper_evo = True
    else:
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    if opt.moving_mode_indices is None:
        n_modes = len(psi0.dims[0])
        opt.moving_mode_indices = np.arange(n_modes)

    #
    # setup integrator
    #
    new_args = None

    if type(args) is dict:
        new_args = {}
        for key in args:
            if isinstance(args[key], Qobj):
                new_args[key] = args[key].data
            else:
                new_args[key] = args[key]

    elif type(args) is list or type(args) is tuple:
        new_args = []
        for arg in args:
            if isinstance(arg, Qobj):
                new_args.append(arg.data)
            else:
                new_args.append(arg)

        if type(args) is tuple:
            new_args = tuple(new_args)
    else:
        if isinstance(args, Qobj):
            new_args = args.data
        else:
            new_args = args

    if oper_evo:
        initial_vector = operator_to_vector(psi0).full().ravel()
        # Check that function returns superoperator
        if H_func(0, args).issuper:
            L_func = H_func
        else:
            L_func = lambda t, args: spre(H_func(t, args))

    else:
        new_psi0 = psi0
        initial_vector = new_psi0.full().ravel()
        L_func = H_func
        if opt.moving_basis:
            initial_mode_displacements = np.zeros(len(opt.moving_mode_indices), dtype=complex)
            for mode_idx in opt.moving_mode_indices:
                identities = [qeye(dim) for dim in psi0.dims[0]]
                a_op_components = identities
                a_op_components[mode_idx] = destroy(psi0.dims[0][mode_idx])
                a_op = tensor(a_op_components)
                a_expect = expect(a_op, psi0)
                initial_mode_displacements[mode_idx] = a_expect
                displacement_components = identities
                displacement_components[mode_idx] = displace(psi0.dims[0][mode_idx], -a_expect)
                displacement_op = tensor(displacement_components)
                new_psi0 = displacement_op * new_psi0
            initial_vector = np.hstack([initial_vector, initial_mode_displacements])

    if not opt.rhs_with_state:
        print('Using cython function.')
        r = scipy.integrate.ode(cy_ode_psi_func_td)
    else:
        if opt.moving_basis:
            r = scipy.integrate.ode(_ode_psi_func_td_with_state_moving_basis)
        else:
            r = scipy.integrate.ode(_ode_psi_func_td_with_state)

    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    if opt.moving_basis:
        r.set_f_params(L_func, new_args, opt, psi0.dims)
    else:
        r.set_f_params(L_func, new_args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, psi0, tlist, e_ops, opt, progress_bar,
                              dims=psi0.dims)


#
# evaluate dpsi(t)/dt for time-dependent hamiltonian
#
def _ode_psi_func_td(t, psi, H_func, args):
    H = H_func(t, args)
    return -1j * (H * psi)


def _ode_psi_func_td_with_state(t, psi, H_func, args):
    H = H_func(t, psi, args)
    dpsi = -1j * (H * psi)
    return dpsi


def _ode_psi_func_td_with_state_moving_basis(t, y, H_func, args, opt, dims):
    n_moving_modes = len(opt.moving_mode_indices)
    n_modes = len(dims[0])
    psi = y[0:-n_modes]
    mode_displacements = y[-n_modes:]
    # print(type(args), type(mode_displacements))
    args = [args] + list(mode_displacements)
    H = H_func(t, psi, *args)
    ddisplacements = np.zeros(n_modes, dtype=complex)
    H_rotation = 0
    for mode_idx in opt.moving_mode_indices:
        operators = [qeye(dim) for dim in dims[0]]
        operators[mode_idx] = destroy(dims[0][mode_idx])
        a_op = tensor(operators)
        com = commutator(H, a_op)
        ddisplacement = 1j * np.sum(np.conjugate(psi) * com * psi)
        ddisplacements[mode_idx] = ddisplacement
        displacement = mode_displacements[mode_idx]
        H_mode_rotation = 0.5 * 1j * displacement * np.conjugate(ddisplacement) + 1j * np.conjugate(
            ddisplacement) * a_op
        H_mode_rotation += H_mode_rotation.dag()
        H_rotation += H_mode_rotation
    H = H + H_rotation
    dpsi = -1j * (H * psi)
    dy = np.hstack([dpsi, ddisplacements])
    return dy


# -----------------------------------------------------------------------------
# Solve an ODE which solver parameters already setup (r). Calculate the
# required expectation values or invoke callback function at each time step.
#
def _generic_ode_solve(r, psi0, tlist, e_ops, opt, progress_bar, dims=None):
    """
    Internal function for solving ODEs.
    """
    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "sesolve"
    output.times = tlist

    n_modes = len(psi0.dims[0])

    if psi0.isunitary:
        oper_evo = True
        oper_n = dims[0][0]
        norm_dim_factor = np.sqrt(oper_n)
    else:
        oper_evo = False
        norm_dim_factor = 1.0

    if opt.store_states:
        output.states = []

    if isinstance(e_ops, types.FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False

        if n_expt_op == 0:
            # fallback on storing states
            output.states = []
            opt.store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    def get_curr_state_data():
        if oper_evo:
            return r.y.reshape([oper_n, oper_n]).T
        else:
            if opt.moving_basis:
                return r.y[0:-n_modes]
            else:
                return r.y

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    output.displacements = np.zeros([n_modes, len(tlist)], dtype=complex)

    dt = np.diff(tlist)
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)

        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")

        # get the current state / oper data if needed
        cdata = None
        if opt.store_states or opt.normalize_output or n_expt_op > 0:
            cdata = get_curr_state_data()

        if opt.normalize_output:
            # cdata *= _get_norm_factor(cdata, oper_evo)
            cdata *= norm_dim_factor / la_norm(cdata)
            if oper_evo:
                r.set_initial_value(cdata.ravel(), r.t)
            else:
                r.set_initial_value(cdata, r.t)

        if opt.store_states:
            output.states.append(Qobj(cdata, dims=dims))

        if opt.moving_basis:
            output.displacements[:, t_idx] = r.y[-n_modes:]

        if expt_callback:
            # use callback method
            e_ops(t, Qobj(cdata, dims=dims))

        for m in range(n_expt_op):
            output.expect[m][t_idx] = cy_expect_psi(e_ops[m].data, cdata, e_ops[m].isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if not opt.rhs_reuse and config.tdname is not None:
        try:
            os.remove(config.tdname + ".pyx")
        except:
            pass

    if opt.store_final_state:
        cdata = get_curr_state_data()
        if opt.normalize_output:
            cdata *= norm_dim_factor / la_norm(cdata)
        output.final_state = Qobj(cdata, dims=dims)

    return output





import numpy as np
import pandas as pd
from scipy.special import factorial
import scipy
import itertools
import copy
from qutip.tensor import tensor
from qutip.operators import qeye, destroy


def gen_c_matrix(operator, threshold=None):
    dims = operator.dims[0]
    elements = operator.data.tocoo()

    c_limit_a = dims[0]
    c_limit_b = dims[1]

    c_matrix = np.zeros([c_limit_a, c_limit_b, c_limit_a, c_limit_b], dtype=complex)

    for row, col, el in zip(elements.row, elements.col, elements.data):
        mi_l = np.unravel_index(row, dims, order='C')
        mi_r = np.unravel_index(col, dims, order='C')
        mi = np.hstack([mi_l, mi_r])
        for delta_a in range(0, c_limit_a - max(mi[0], mi[2])):
            for delta_b in range(0, c_limit_b - max(mi[1], mi[3])):
                coeff = el * (-1) ** (delta_a + delta_b)
                coeff /= factorial(delta_a) * factorial(delta_b)
                coeff /= np.sqrt(factorial(mi[0]) * factorial(mi[1]) * factorial(mi[2]) * factorial(mi[3]))
                multi_index = [mi[0] + delta_a, mi[1] + delta_b, mi[2] + delta_a, mi[3] + delta_b]
                c_matrix[mi[0] + delta_a, mi[1] + delta_b, mi[2] + delta_a, mi[3] + delta_b] += coeff

    if threshold is not None:
        mask = np.abs(c_matrix) > threshold
        c_matrix *= mask

    return c_matrix


def gen_amp_funcs(component_idx, weight):
    product = itertools.product(range(component_idx[0] + 1), range(component_idx[1] + 1),
                                range(component_idx[2] + 1), range(component_idx[3] + 1))

    generated_indices = []
    amp_funcs_list = []

    for generated_idx in list(product):
        binom_coeffs = [scipy.special.comb(c_idx, g_idx, exact=True)
                        for c_idx, g_idx in zip(component_idx, generated_idx)]
        coeff = weight * np.prod(binom_coeffs)
        powers = [(c_idx - g_idx) for c_idx, g_idx in zip(component_idx, generated_idx)]

        amp_func = amp_func_factory(coeff, powers)

        generated_indices.append(generated_idx)
        amp_funcs_list.append(amp_func)

    multi_index = pd.MultiIndex.from_tuples(generated_indices, names=['a_l', 'b_l', 'a_r', 'b_r'])
    amp_funcs_frame = pd.DataFrame(amp_funcs_list, index=multi_index)

    return amp_funcs_frame


def amp_func_factory(coeff_in, powers):
    def amp_func(disp=[0, 0]):
        coeff = coeff_in
        coeff *= np.conjugate(disp[0]) ** powers[0]
        coeff *= np.conjugate(disp[1]) ** powers[1]
        coeff *= disp[0] ** powers[2]
        coeff *= disp[1] ** powers[3]
        return coeff

    return amp_func


def combine(frame1, frame2):
    combined = copy.deepcopy(frame1)
    for idx in frame2.index:
        if idx not in combined.index:
            combined.loc[idx] = frame2.loc[idx]
        else:
            combined.loc[idx] = function_adder(frame1.loc[idx, 0], frame2.loc[idx, 0])
    return combined


def function_adder(func1, func2):
    def combined_func(x):
        total = func1(x) + func2(x)
        return total

    return combined_func


def gen_pairs(amp_dependence_funcs, dims, time_func):
    pairs = []
    a = tensor(destroy(dims[0]), qeye(dims[1]))
    b = tensor(qeye(dims[0]), destroy(dims[1]))
    for idx in amp_dependence_funcs.index:
        amp_func = amp_dependence_funcs.loc[idx].iloc[0]
        dependence_func = copy.deepcopy(function_multiplier(time_func, amp_func))
        component = a.dag() ** idx[0] * b.dag() ** idx[1] * a ** idx[2] * b ** idx[3]
        pairs.append([component, dependence_func])
    pairs = pd.DataFrame(pairs, columns=['component', 'func'], index=amp_dependence_funcs.index)
    return pairs


def function_multiplier(time_func, amp_func):
    def combined_func(t, psi, args, disp, vel):
        total = time_func(t, psi, args) * amp_func(disp)
        return total

    return combined_func



