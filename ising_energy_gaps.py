from __future__ import print_function, division
from quspin.operators import hamiltonian, commutator  # Hamiltonian and operators
from quspin.basis import spin_basis_1d, spinless_fermion_basis_1d  # Hilbert space spin basis
from quspin.tools.evolution import evolve
import numpy as np
import matplotlib.pyplot as plt
from fh_energy_gaps import get_energy_gap, alt_get_energy_gap

from sklearn.linear_model import LinearRegression


def rhs(tau, phi, h):
    """
    RHS of imaginary time propagation of phi
    :param tau: imaginary time
    :param phi: wavefunction
    :param h:   hamiltonian
    :return:
    """
    return -h.dot(phi)


if __name__ == "__main__":
    # Define plotting parameters
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Define model parameters
    L = 2  # system size
    J = 1  # spin zz interaction
    #h = np.sqrt(2)  # z magnetic field strength
    h = 1
    PBC = 1  # periodic boundary condition

    # Define spin model
    # Site-coupling lists
    h_field = [[-h, i] for i in range(L)]
    J_zz = [[-J, i, (i+1) % L] for i in range(L)]  # PBC

    # Define spin static and dynamic lists
    static_spin = [["zz", J_zz], ["x", h_field]]
    dynamic_spin = []

    # Construct spin basis
    basis_spin = spin_basis_1d(L=L)

    # Build spin Hamiltonian
    H_spin = hamiltonian(static_spin, dynamic_spin, basis=basis_spin)
    print("H:", H_spin.toarray())
    E_spin, V_spin = H_spin.eigh()
    print("E_0: ", E_spin[0])

    # Get exact gap and ground state
    exact_gap_sq = (E_spin[0] - E_spin[1]) ** 2
    gs = V_spin[:, 0]
    print("Check normalization of ground state: ")
    print(sum(abs(gs) ** 2))

    # Create random complex initial state for imaginary time propagation
    np.random.seed(1)
    psi0 = np.random.random(gs.shape[0]) + 1j * np.random.random(gs.shape[0])
    psi0 /= np.sqrt((sum(abs(psi0) ** 2)))
    print(psi0)
    print("Check normalization of psi0: ")
    print(sum(abs(psi0) ** 2))

    # Create tau list
    tau_inc = np.linspace(0, 15, 100)
    rhs_params = (H_spin,)
    psi_tau = evolve(psi0, tau_inc[0], tau_inc, rhs, f_params=rhs_params, imag_time=True, iterate=True)

    psi_list = []
    # Plot results
    for i, psi in enumerate(psi_tau):
        plt.plot(abs(gs) ** 2, 'rs', label='$|\\psi_0|^2$')
        plt.plot(abs(psi) ** 2, 'b-', label='$|\\psi|^2$')
        plt.xlabel("Sites")
        plt.ylim([-0.01, max(abs(gs) ** 2) + 0.01])
        plt.legend(numpoints=1)
        plt.draw()
        plt.pause(0.005)
        plt.clf()
        psi_list.append(psi)

    plt.close()

    # Plot final wf from imag time prop and the exact ground state
    print("Norm of difference between exact and calc GS: ")
    print(np.linalg.norm(abs(gs) ** 2 - abs(psi_list[-1]) ** 2), '\n')

    fig_num = 1
    plt.figure(fig_num)
    plt.title("Ground State")
    plt.plot(abs(gs) ** 2, label='Spin')
    plt.plot(abs(psi_list[-1]) ** 2, label='Imag Time Spin')
    plt.xlabel("Sites")
    plt.legend(numpoints=1)
    fig_num += 1

    # Calculate energy gaps
    lim_spin_int = [[-h, i] for i in range(1)]
    obs_spin_static = [["z", lim_spin_int]]#, ["I", lim_spin_int]]
    obs_spin = hamiltonian(obs_spin_static, dynamic_spin, basis=basis_spin)
    print("Observable: ", obs_spin.toarray())

    print("\n", "######## Spin ########")
    calc_gap_sq = get_energy_gap(H_spin, obs_spin, psi_list[-1])
    alt_calc_gap_sq = alt_get_energy_gap(H_spin, obs_spin, psi_list[-1])
    print("Actual gap squared: ", exact_gap_sq)
    print("Calculated gap squared: ", calc_gap_sq)
    print("Rel diff for 1st method: ", abs(exact_gap_sq - calc_gap_sq) / exact_gap_sq)
    print("Gap squared using 4th and 2nd commutators: ", alt_calc_gap_sq)
    print("Rel diff for 2nd method: ", abs(exact_gap_sq - alt_calc_gap_sq) / exact_gap_sq)

    sin_com = commutator(H_spin, obs_spin).expt_value(psi_list[-1])
    doub_com = commutator(H_spin, commutator(H_spin, obs_spin)).expt_value(psi_list[-1])
    print("Test: ", doub_com / sin_com)

    print("\n####### Other info for 1st method ########")
    print("Expectation value of denominator: ", commutator(H_spin, obs_spin).expt_value(psi_list[-1]))
    print("[H, O] eigs: ", np.linalg.eigh(commutator(H_spin, obs_spin).toarray())[0])
    print("Expectation value of numerator: ",
          commutator(H_spin, commutator(H_spin, commutator(H_spin, obs_spin))).expt_value(psi_list[-1]))
    spin_fst_exc = V_spin[:, 1] * np.sqrt(L)
    print("[H, O]_3 eigs: ", np.linalg.eigh(commutator(H_spin, commutator(H_spin, commutator(H_spin, obs_spin))).toarray())[0])
    print("Overlap of observable: ", spin_fst_exc.conj().T @ obs_spin.toarray() @ gs)


    # Calculate energy gap as a function of beta
    spin_energy_gaps_sq = [get_energy_gap(H_spin, obs_spin, _) for _ in psi_list]
    alt_spin_energy_gaps_sq = [alt_get_energy_gap(H_spin, obs_spin, _) for _ in psi_list]

    spin_rel_dif = abs(np.array(spin_energy_gaps_sq) - exact_gap_sq) / exact_gap_sq
    log_rel_err = np.log(spin_rel_dif * exact_gap_sq)
    print("Slope of rel err: ", (log_rel_err[26] - log_rel_err[0]) / (tau_inc[26] - tau_inc[0]))
    print("E2-E1 gap: ", E_spin[2] - E_spin[1])
    alt_spin_rel_dif = abs(np.array(alt_spin_energy_gaps_sq) - exact_gap_sq) / exact_gap_sq
    plt.figure(fig_num)
    plt.semilogy(tau_inc, spin_rel_dif, label="$M=1$")#label='$\\langle[H, O]_3\\rangle / \\langle[H, O]_1\\rangle$')
    plt.semilogy(tau_inc, alt_spin_rel_dif, label="$M=2$")#label='$\\langle[H, O]_4\\rangle / \\langle[H, O]_2\\rangle$')
    #plt.xlabel("Imaginary time, $\\tau$")
    plt.ylabel("Relative error, $\\epsilon$")
    plt.legend(numpoints=1)
    plt.savefig("Figs/spin_rel_error.png", format='png', dpi=300)
    fig_num += 1

    plt.figure(fig_num)
    plt.plot([abs(psi_list[0].conj().T @ _) ** 2 for _ in V_spin.T], label="psi_0")
    plt.plot([abs(psi_list[-1].conj().T @ _) ** 2 for _ in V_spin.T], label="psi_T")
    plt.xlabel("Eigenvector Index")
    plt.ylabel("Probability")
    plt.legend(numpoints=1)
    fig_num += 1

    print("Ground state energy: ", E_spin[0])
    #print("Sing comm: ", commutator(H_spin, obs_spin).toarray())
    #temp = H_spin.toarray() @ obs_spin.toarray()
    #print(temp @ temp.conj().T)
    #print("Final WF: ", psi_list[-1])
    print("<HO>: ", psi_list[-1].conj().T @ H_spin.toarray() @ obs_spin.toarray() @ psi_list[-1])
    print("<OH>: ", psi_list[-1].conj().T @ obs_spin.toarray() @ H_spin.toarray() @ psi_list[-1])
    #print("Eigvecs: ", np.linalg.eigh(temp)[1])

    print("##########################")
    sin_expt_vals = np.array([commutator(H_spin, obs_spin).expt_value(_) for _ in psi_list])
    tri_expt_vals = np.array([commutator(H_spin, commutator(H_spin, commutator(H_spin, obs_spin))).expt_value(_) for _ in psi_list])
    plt.figure(fig_num)
    plt.plot(tau_inc, sin_expt_vals.imag, label='single')
    plt.plot(tau_inc, tri_expt_vals.imag, label='triple')

    h_expt_vals = [H_spin.expt_value(_) for _ in psi_list]
    plt.figure(fig_num)
    plt.plot(tau_inc, h_expt_vals)
    plt.plot(tau_inc, E_spin[0]*np.ones(len(h_expt_vals)), 'k-')
    plt.xlabel("$\\tau$")
    plt.ylabel("<H>")
    fig_num += 1

    sing_y_vals = np.log(np.abs(tri_expt_vals))
    sing_reg = LinearRegression().fit(
        tau_inc.reshape(-1, 1),
        sing_y_vals.reshape(-1, 1)
    )
    plt.figure(fig_num)
    plt.title("[H, O]")
    plt.plot(tau_inc, sing_y_vals)
    plt.plot(tau_inc, sing_reg.predict(tau_inc.reshape(-1, 1)))
    print("slope: ", sing_reg.coef_)
    print("expected slope: ", E_spin[0] + E_spin[1])
    print("diff: ", E_spin[1] - E_spin[0])
    print("intercept: ", sing_reg.intercept_)

    fig_num += 1
    plt.figure(fig_num)
    plt.semilogy(tau_inc, np.abs(sin_expt_vals))

    sigma_z = 1/np.sqrt(2) * np.array([[1, 0], [0, -1]])
    print(np.kron(sigma_z, sigma_z))
    print(psi0)
    plt.show()
