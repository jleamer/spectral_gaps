from quspin.operators import hamiltonian, commutator, anti_commutator
from quspin.basis import spinful_fermion_basis_1d
from quspin.tools.evolution import evolve
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool


def rhs(tau, phi, h):
    """
    RHS of imaginary time propagation of phi
    :param tau: imaginary time
    :param phi: wavefunction
    :param h:   hamiltonian
    :return:
    """
    return -h.dot(phi)


def adiabatic_rhs(times, phi, h, u, period):
    return -1j * h.dot(phi) + -1j * times / period * u.dot(phi)


def get_energy_gap(ham, obs, wf):
    """
    Calculates the energy gap as <wf|[ham, obs]_3|wf>/<wf|[ham, obs]_1|wf>
    :param ham: hamiltonian
    :param obs: observable - just need to overlap gs and 1st excited state
    :param wf:  wavefunction from imaginary time propagation
    :return:    the energy gap
    """
    # Calculate commutators
    sing_comm = commutator(ham, obs)
    trip_comm = commutator(ham, commutator(ham, sing_comm))

    return trip_comm.expt_value(wf) / sing_comm.expt_value(wf)


def alt_get_energy_gap(ham, obs, wf):
    """
    Calculate the energy gap as <wf|[ham, obs]_4|wf>/<wf|[ham, obs]_2|wf>
    :param args:
    :return:
    """
    # Calculate commutators
    sing_comm = commutator(ham, obs)
    doub_comm = commutator(ham, sing_comm)
    quad_comm = commutator(ham, commutator(ham, doub_comm))

    return quad_comm.expt_value(wf) / doub_comm.expt_value(wf)


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
    L = 4  # system size
    J = 1  # hopping parameter
    U = np.sqrt(2)  # interaction parameter
    mu = 0  # chemical potential

    # Define basis with 3 states per site L fermions in lattice
    N_up = L // 2 + L % 2  # number of fermions with spin up
    N_down = L // 2  # number of fermions with spin down
    basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))

    # Define site-coupling lists
    hop_right = [[-J, i, (i+1) % L] for i in range(L-1)]  # APBC
    hop_left = [[+J, i, (i+1) % L] for i in range(L-1)]  # APBC
    print(hop_left)
    interact = [[U, i, i] for i in range(L)]  # U/2 \sum_j n_{j, up} n_{j_down}

    # Define static and dynamic lists
    static = [
        ['+-|', hop_left],      # up hops left
        ['-+|', hop_right],     # up hops right
        ['|+-', hop_left],      # down hops left
        ['|-+', hop_right],     # down hops right
        ['n|n', interact]       # up-down interaction
    ]

    dynamic = []

    # Build Hamiltonian
    H = hamiltonian(static, dynamic, basis=basis, check_symm=False)

    # Get eigenvalues and eigenvectors
    E, V = H.eigh()
    exact_gap_squared = (E[0] - E[1]) ** 2
    print(E)
    gs = V[:, 0]
    fst_exc = V[:, 1]
    print("=========")
    print("Check normalization of gs:")
    print(sum(abs(gs) ** 2))

    # Set psi0 as a linear combination of the first and second eigenvector
    #psi0 = V[:, 0] / np.sqrt(2) + V[:, 1] / np.sqrt(2)
    # Set psi0 to ones, then normalize
    np.random.seed(64)
    psi0 = np.random.random(gs.shape[0]) + 1j * np.random.random(gs.shape[0])
    #psi0 = np.ones(gs.shape[0])
    #psi0 = np.sin(np.linspace(0, np.pi, int(E.shape[0])))
    psi0 /= np.sqrt(np.sum(abs(psi0) ** 2))
    print("Check normalization of psi0:")
    print(sum(abs(psi0) ** 2))
    print("=========")
    tau_inc = np.linspace(0, 20, 100)
    rhs_params = (H,)

    # Evolve state
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

    fig_count = 1

    plt.figure(fig_count)
    plt.title("Exact vs Calc GS")
    plt.plot(abs(gs) ** 2, 'rs', label='$|\\psi_{gs}|^2$')
    plt.plot(abs(psi_list[-1]) ** 2, 'b-', label='$|\\psi|^2$')
    plt.xlabel("Sites")
    plt.ylim([-0.01, max(abs(gs) ** 2) + 0.01])
    plt.legend(numpoints=1)
    fig_count += 1

    # Create observable from the interaction term of 01
    fst_interact = [[1, 0, 1]]
    fst_interaction_static = [
        ["n|n", fst_interact]
    ]
    fst_interaction = hamiltonian(fst_interaction_static, dynamic, basis=basis, check_symm=False)

    # Calculate (E0-E1) ** 2 using expectations of [H, O]_3 and [H, O]_1
    fst_gap_squared = get_energy_gap(H, fst_interaction, psi_list[-1])

    print("Exact gap squared: ", exact_gap_squared)
    print("Calculated gap squared from (0, 1): ", fst_gap_squared)
    temp = get_energy_gap(H, fst_interaction, gs)
    print("temp: ", temp)

    # Check alternate calculation of gap
    alt_fst_gap_squared = alt_get_energy_gap(H, fst_interaction, psi_list[-1])
    print("Calculated gap squared for (0, 1) using [H, O]_4 and [H, O]_2: ", alt_fst_gap_squared)

    print("Overlap with fst obs: ", fst_exc.conj().T @ fst_interaction.toarray() @ gs, '\n')
    sing_comm = commutator(H, fst_interaction)
    trip_comm = commutator(H, commutator(H, sing_comm))
    print("Single Commutator: ", sing_comm.expt_value(psi_list[-1]))
    print("Triple Commutator: ", trip_comm.expt_value(psi_list[-1]))
    print(trip_comm.expt_value(psi_list[-1]) / sing_comm.expt_value(psi_list[-1]))


    # Now try another site - say 10 10
    snd_interact = [[1, 2, 1], [1, 2, 3]]
    snd_interaction_static = [
        ["n|n", snd_interact]
    ]
    snd_interaction = hamiltonian(snd_interaction_static, dynamic, basis=basis, check_symm=False)

    snd_gap_squared = get_energy_gap(H, snd_interaction, psi_list[-1])
    print("Calculated gap squared from (2, 3): ", snd_gap_squared)

    # Check alternate calculation of gap
    alt_snd_gap_squared = alt_get_energy_gap(H, snd_interaction, psi_list[-1])
    print("Calculated gap squared from (2, 3) using [H, O]_4 and [H, O]_2: ", alt_snd_gap_squared, '\n')

    print("Overlap with snd obs: ", fst_exc.conj().T @ snd_interaction.toarray() @ gs)

    # Calculate energy gap as a function of beta
    energy_gaps_sq = [get_energy_gap(H, fst_interaction, _) for _ in psi_list]
    alt_energy_gaps_sq = [alt_get_energy_gap(H, fst_interaction, _) for _ in psi_list]

    rel_dif = abs(np.array(energy_gaps_sq) - exact_gap_squared) / exact_gap_squared
    alt_rel_dif = abs(np.array(alt_energy_gaps_sq) - exact_gap_squared) / exact_gap_squared
    plt.figure(fig_count)
    plt.semilogy(tau_inc, rel_dif, label="$M=1$")#label='$\\langle[H, O]_3\\rangle / \\langle[H, O]_1\\rangle$')
    plt.semilogy(tau_inc, alt_rel_dif, label="$M=2$")#label='$\\langle[H, O]_4\\rangle / \\langle[H, O]_2\\rangle$')
    plt.xlabel("Imaginary time, $\\tau$")
    plt.ylabel("Relative error, $\\epsilon$")
    plt.legend(numpoints=1)
    plt.savefig("Figs/rel_error.png", format='png', dpi=300)
    fig_count += 1

    snd_energy_gaps_sq = [get_energy_gap(H, snd_interaction, _) for _ in psi_list]
    alt_snd_energy_gaps_sq = [alt_get_energy_gap(H, snd_interaction, _) for _ in psi_list]
    snd_rel_dif = abs(np.array(snd_energy_gaps_sq) - exact_gap_squared) / exact_gap_squared
    alt_snd_rel_dif = abs(np.array(alt_snd_energy_gaps_sq) - exact_gap_squared) / exact_gap_squared
    plt.figure(fig_count)
    plt.semilogy(tau_inc, snd_rel_dif, label="$<[H, O]_3> / <[H, O]_1>$")
    plt.semilogy(tau_inc, alt_snd_rel_dif, label="$<[H, O]_4> / <[H, O]_2>$")
    plt.xlabel("$\\tau$")
    plt.ylabel("Rel. Error")
    plt.title("$O = Un_2n_3 + Un_2n_1$")
    plt.legend(numpoints=1)
    plt.savefig("Figs/rel_error_snd.png", format="png", dpi=300)
    fig_count += 1

    plt.figure(fig_count)
    plt.semilogy([abs(psi_list[-1].conj().T @ _) ** 2 for _ in V.T], "bo-", label="End")
    plt.semilogy([abs(psi_list[0].conj().T @ _) ** 2 for _ in V.T], "rs-", label="Beginning")
    plt.xlabel("Index")
    plt.ylabel("Overlap")
    plt.legend(numpoints=1)
    fig_count += 1

    plt.figure(fig_count)
    plt.plot(gs.real, 'rs-', label="$\\mathrm{Re}(\\psi_0)$")
    plt.plot(psi_list[-1].real, 'bo-', label="$\\mathrm{Re}(\\psi_{t_f})$")
    plt.xlabel("Sites")
    plt.legend(numpoints=1)
    plt.savefig("Figs/fh_real.png", format='png', dpi=300)
    fig_count += 1

    plt.figure(fig_count)
    plt.plot(gs.imag, 'rs-', label="$\\mathrm{Im}(\\psi_0)$")
    plt.plot(psi_list[-1].imag, 'bo-', label="$\\mathrm{Im}(\\psi_{t_f})$")
    plt.xlabel("Sites")
    plt.legend(numpoints=1)
    plt.savefig("Figs/fh_imag.png", format='png', dpi=300)
    fig_count += 1

    # Try to get the gap by
    filtered_energy_gaps_sq = np.array(energy_gaps_sq)
    alt_filtered_energy_gaps_sq = np.array(alt_energy_gaps_sq)
    plt.figure(fig_count)
    plt.plot(tau_inc, filtered_energy_gaps_sq, label="$<[H, O]_3> / <[H, O]_1>$")
    plt.plot(tau_inc, alt_filtered_energy_gaps_sq, label="$<[H, O]_4> / <[H, O]_2>$")
    plt.xlabel("$\\beta$")
    plt.legend(numpoints=1)
    fig_count += 1

    """
    filtered_energy_gaps_sq = filtered_energy_gaps_sq[filtered_energy_gaps_sq > 0]
    x = (filtered_energy_gaps_sq - min(filtered_energy_gaps_sq)) / min(filtered_energy_gaps_sq)
    print(min(filtered_energy_gaps_sq))
    x = np.log(x)[10:80]
    #alt_filtered_energy_gaps_sq = alt_filtered_energy_gaps_sq[alt_filtered_energy_gaps_sq > 0]
    y = (alt_filtered_energy_gaps_sq[15:80] - min(alt_filtered_energy_gaps_sq[15:80])) / min(alt_filtered_energy_gaps_sq[15:80])
    print(min(alt_filtered_energy_gaps_sq))
    y = np.log(y)
    plt.figure(fig_count)
    plt.plot(tau_inc[10:80], x, label="$<[H, O]_3> / <[H, O]_1>$")
    plt.plot(tau_inc[15:80], y, label="$<[H, O]_4> / <[H, O]_2>$")
    plt.xlabel("$\\beta$")
    plt.legend(numpoints=1)

    print("from 3rd and 1st comm: ", (x[-1] - x[0]) / (tau_inc[80] - tau_inc[5]))
    print("from 4th and 2nd comm: ", (y[-1] - y[0]) / (tau_inc[80] - tau_inc[15]))
    """
    plt.figure(fig_count)
    plt.plot(E, '*-')
    plt.plot([E[0], E[1]], '*-')
    plt.xlabel("Index")
    plt.ylabel("Energy")
    plt.savefig("Figs/spectrum_ex.png", format='png', dpi=300)
    plt.show()
