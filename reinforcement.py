import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import os
import concurrent.futures
import time
import pandas as pd

dt = 0.0001
KT = 4.1
DEFAULT_TMAX = 300
N_repeats = 300
BASE_OUTPUT_DIR = "./simulation_results"

config = {
    'th': [9],
    'st': [1],
    'dis': [0.5],
    'em': [0.1387873043],
    'ca': [0.005],
    'fr': [1]
}


class Integrin:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 1
        self.k_off = 1e-40

    def update_koff(self, force):
        self.k_off = 1 / 150

    def dettach(self):
        self.state = 0


class Talin:
    def __init__(self):
        self.kunfold = 2.5e-3
        self.kfold = 1.2e5
        self.xunfold = 6.49
        self.xfold = 4.61
        self.reset()

    def reset(self):
        self.state = 0
        self.first_unfold = 0

    def unfold(self):
        self.state = 1

    def fold(self):
        self.state = 0

    def update_and_get_probs(self, F, RANDOM, Material):
        p_u = dt * self.kunfold * np.exp(F * self.xunfold / KT)
        p_f = dt * self.kfold * np.exp(-F * self.xfold / KT)
        if self.state == 0:
            if RANDOM < p_u:
                self.unfold()
                if self.first_unfold == 0:
                    Material.force_unfolding = F
                    self.first_unfold = 1
        elif self.state == 1:
            if RANDOM < p_f:
                self.fold()
        return p_u, p_f


class Vinculin:
    def __init__(self):
        self.kbind = 0
        self.concentration = 7.5
        self.A = (self.concentration ** 2) * 2.8e-4
        self.delta_KT = 0.09 / KT

    def update_kbind(self, force):
        self.kbind = self.A * np.exp(-self.delta_KT * force)
        return self.kbind


class Material:
    def __init__(self, sim_tmax):
        self.clutch_area = 0.002
        self.sim_tmax = sim_tmax
        self.steps = int(round(sim_tmax / dt)) + 5000
        self.stress0 = np.zeros(self.steps)
        self.stress = np.zeros(self.steps)
        self.h1 = np.zeros(self.steps)
        self.h2 = np.zeros(self.steps)
        self.h3 = np.zeros(self.steps)
        self.h4 = np.zeros(self.steps)
        self.h5 = np.zeros(self.steps)
        self.e_template = None
        self.e = None
        self.reset()

    def reset(self):
        self.F = 0
        self.flag = 0
        self.emax = 0
        self.i = 0
        if self.e_template is not None:
            self.e = np.copy(self.e_template)

    def set_simcond(self, stiffness_mult, dissipation_mult, fraction, target_emax, target_clutch_area):
        self.clutch_area = target_clutch_area
        a = 1.0 * dissipation_mult
        self.kl = 17 * stiffness_mult
        self.kv1 = 1.8370 * a
        self.tau1 = 0.1652
        self.kv2 = 5.1823 * a
        self.tau2 = 0.9160
        self.kv3 = 7.8067 * a
        self.tau3 = 8.8383
        self.kv4 = 0
        self.tau4 = 11.0380
        self.kv5 = 0
        self.tau5 = 94.2670

        def get_exp(tau):
            return np.exp(-dt / tau) if tau > 0 else 0

        self.exp_t1 = get_exp(self.tau1)
        self.exp_t2 = get_exp(self.tau2)
        self.exp_t3 = get_exp(self.tau3)
        self.exp_t4 = get_exp(self.tau4)
        self.exp_t5 = get_exp(self.tau5)

        n = self.sim_tmax / dt
        emax = target_emax
        len_e1 = int(max(1, fraction * n / 1000))
        len_e2 = max(0, self.steps - len_e1)

        e1 = np.linspace(0, emax, len_e1)
        e2 = np.linspace(emax, emax, len_e2)
        self.e_template = np.concatenate((e1, e2))
        self.e = np.copy(self.e_template)

    def updatemechanics(self, threshold):
        i = self.i
        if i >= len(self.e) - 1:
            return
        if self.F >= threshold and self.flag == 0:
            self.emax = self.e[i]
            self.flag = 1
            self.e[i:] = self.emax

        stress0 = self.stress0
        e = self.e
        stress0[i + 1] = e[i + 1] * self.kl
        d_stress = stress0[i + 1] - stress0[i]
        factor_kl = 1.0 / self.kl if self.kl > 0 else 0

        def fast_calc(h_prev, kv, exp_term, tau):
            if tau == 0 or kv == 0:
                return 0
            return exp_term * h_prev + (kv * factor_kl) * (1 - exp_term) / (dt / tau) * d_stress

        self.h1[i + 1] = fast_calc(self.h1[i], self.kv1, self.exp_t1, self.tau1)
        self.h2[i + 1] = fast_calc(self.h2[i], self.kv2, self.exp_t2, self.tau2)
        self.h3[i + 1] = fast_calc(self.h3[i], self.kv3, self.exp_t3, self.tau3)
        self.h4[i + 1] = fast_calc(self.h4[i], self.kv4, self.exp_t4, self.tau4)
        self.h5[i + 1] = fast_calc(self.h5[i], self.kv5, self.exp_t5, self.tau5)
        self.stress[i + 1] = stress0[i + 1] + self.h1[i + 1] + self.h2[i + 1] + self.h3[i + 1] + self.h4[i + 1] + \
                             self.h5[i + 1]
        self.F = self.stress[i + 1] * self.clutch_area * 1000
        self.i += 1


def run_simulation_batch(st, dis, th, current_emax, current_clutch_area, current_fraction, n_repeats, current_tmax,
                         capture_trace=True):
    np.random.seed(int(time.time() * 1000) % 2 ** 32 + os.getpid())
    binding_times = []
    trace_data = None
    integrin = Integrin()
    talin = Talin()
    vinculin = Vinculin()
    mat = Material(current_tmax)
    mat.set_simcond(st, dis, current_fraction, current_emax, current_clutch_area)

    max_steps = int(current_tmax / dt)
    n_points = max_steps // 1000 + 1
    trace_time = np.zeros(n_points)
    trace_F = np.zeros(n_points)
    trace_Pu = np.zeros(n_points)
    trace_Pr = np.zeros(n_points)
    trace_kb = np.zeros(n_points)

    for r in range(n_repeats):
        integrin.reset()
        talin.reset()
        mat.reset()
        is_trace_run = (r == 0) and capture_trace
        trace_idx = 0

        for step in range(max_steps):
            mat.updatemechanics(th)
            F = mat.F
            r1, r2, r3 = rnd.random(), rnd.random(), rnd.random()
            p_u, p_f = talin.update_and_get_probs(F, r2, mat)
            kb = vinculin.update_kbind(F)

            if is_trace_run and (step % 1000 == 0) and trace_idx < n_points:
                trace_time[trace_idx] = step * dt
                trace_F[trace_idx] = F
                trace_Pu[trace_idx] = p_u
                trace_Pr[trace_idx] = p_f
                trace_kb[trace_idx] = kb
                trace_idx += 1

            integrin.update_koff(F)
            if r1 < integrin.k_off * dt:
                integrin.dettach()
                if not is_trace_run:
                    break

            if integrin.state != 0 and talin.state == 1:
                if r3 < kb * dt:
                    binding_times.append(step * dt)
                    if not is_trace_run:
                        break

        if is_trace_run:
            trace_data = {
                'time': trace_time[:trace_idx], 'F': trace_F[:trace_idx],
                'Pu': trace_Pu[:trace_idx], 'Pr': trace_Pr[:trace_idx], 'kb': trace_kb[:trace_idx]
            }

    return binding_times, trace_data, current_tmax


def process_condition(params):
    th, st, dis, emax_val, c_area, frac, current_tmax = params
    mat_type = '10g'
    save_dir = os.path.join(BASE_OUTPUT_DIR, f"Threshold_{th}pN", mat_type)
    suffix_str = f"{mat_type}_St{st}_Dis{dis}_Emax{emax_val}_Area{c_area}_Frac{frac}"
    csv_path = os.path.join(save_dir, f"SuccessRate_{suffix_str}.csv")

    if os.path.exists(csv_path):
        return

    times, trace, used_tmax = run_simulation_batch(st, dis, th, emax_val, c_area, frac, N_repeats, current_tmax)
    plt.switch_backend('Agg')
    os.makedirs(save_dir, exist_ok=True)
    time_axis = np.arange(0, used_tmax + 1, 1)
    times_arr = np.array(times)
    rates = [np.sum(times_arr <= t) / N_repeats for t in time_axis]

    pd.DataFrame({'Time_s': time_axis, f'{mat_type}_SuccessRate': rates}).to_csv(csv_path, index=False)

    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(time_axis, rates, color='r', label=mat_type, linewidth=2)
    plt.title(f"{mat_type} Success Rate\n(Th={th}, Fr={frac}, Tmax={used_tmax}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    fig1.savefig(os.path.join(save_dir, f"SuccessRate_{suffix_str}.png"), dpi=300)
    plt.close(fig1)

    if trace:
        pd.DataFrame({
            'Time_s': trace['time'], f'{mat_type}_Force': trace['F'],
            f'{mat_type}_P_unfold': trace['Pu'], f'{mat_type}_P_refold': trace['Pr'],
            f'{mat_type}_kbind': trace['kb']
        }).to_csv(os.path.join(save_dir, f"Dynamics_{suffix_str}.csv"), index=False)

        fig2, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        fig2.suptitle(f"{mat_type} Trace (Fr={frac})\n(Tmax={used_tmax}s)")
        axs[0, 0].plot(trace['time'], trace['F'], 'r', label=mat_type)
        axs[0, 0].set_title('Force (F)')
        axs[0, 0].set_ylabel('pN')
        axs[0, 0].legend()
        axs[0, 1].plot(trace['time'], trace['Pu'], 'r')
        axs[0, 1].set_title('Unfold Prob')
        axs[0, 1].set_yscale('log')
        axs[1, 0].plot(trace['time'], trace['Pr'], 'r')
        axs[1, 0].set_title('Refold Prob')
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 1].plot(trace['time'], trace['kb'], 'r')
        axs[1, 1].set_title('Binding Rate')
        axs[1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig(os.path.join(save_dir, f"Dynamics_{suffix_str}.png"), dpi=300)
        plt.close(fig2)


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    tasks_list = []

    for st in config['st']:
        for dis in config['dis']:
            for em in config['em']:
                for ca in config['ca']:
                    for frac in config['fr']:
                        current_tmax = int(max(DEFAULT_TMAX, np.ceil(frac * 0.3 + 100)))
                        for th in config['th']:
                            task_params = (th, st, dis, em, ca, frac, current_tmax)
                            csv_path = os.path.join(BASE_OUTPUT_DIR, f"Threshold_{th}pN", "10g",
                                                    f"SuccessRate_10g_St{st}_Dis{dis}_Emax{em}_Area{ca}_Frac{frac}.csv")
                            if not os.path.exists(csv_path):
                                tasks_list.append(task_params)

    if tasks_list:
        max_workers = max(1, os.cpu_count() or 22)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_condition, tasks_list))
