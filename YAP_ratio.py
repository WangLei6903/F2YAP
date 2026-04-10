import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import numpy as np
import matplotlib as mpl

BASE_DIR = "./simulation_results"
TARGET_PARAMS = {'St': 1.0, 'Emax': 0.1387873043, 'Area': 0.005, 'Frac': 1.0}
TARGET_MAT = '10g'
TIME_POINT = 300

R_MIN = 0.8880
DELTA_R = 1.5999
OPTIMAL_K = 3.1431


def transform_to_yap(rate, k=OPTIMAL_K):
    dg_current = (k * rate + 1e-10) ** (-2)
    return R_MIN + DELTA_R * np.exp(-dg_current)


def set_cns_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.transparent': True
    })


def parse_filename(filename):
    try:
        params = {}
        params['St'] = float(re.search(r'_St([\d\.]+)', filename).group(1))
        params['Dis'] = float(re.search(r'_Dis([\d\.]+)', filename).group(1))
        params['Emax'] = float(re.search(r'_Emax([\d\.]+)', filename).group(1))
        params['Area'] = float(re.search(r'_Area([\d\.]+)', filename).group(1))
        frac_match = re.search(r'_Frac([\d\.]+)\.csv', filename)
        params['Frac'] = float(frac_match.group(1)) if frac_match else 1.0
        return params
    except:
        return None


def run_comparison_analysis(target_threshold, control_dis, exp_dis):
    folder_name = f"Comparison_{target_threshold}pN_Dis{control_dis}_vs_{exp_dis}"
    result_folder = os.path.join(BASE_DIR, folder_name)
    os.makedirs(result_folder, exist_ok=True)

    target_dir_pattern = os.path.join(BASE_DIR, f"Threshold_{target_threshold}pN", TARGET_MAT)

    if not os.path.exists(target_dir_pattern):
        for d in glob.glob(os.path.join(BASE_DIR, "Threshold_*pN")):
            m = re.search(r'Threshold_([\d\.]+)pN', d)
            if m and abs(float(m.group(1)) - target_threshold) < 1e-4:
                target_dir_pattern = os.path.join(d, TARGET_MAT)
                break

    data_points = []
    for csv_path in glob.glob(os.path.join(target_dir_pattern, "SuccessRate_*.csv")):
        params = parse_filename(os.path.basename(csv_path))
        if not params or any(abs(params.get(k, -999) - v) > 1e-4 for k, v in TARGET_PARAMS.items()):
            continue

        current_dis = params['Dis']
        if abs(current_dis - control_dis) < 1e-4:
            group_label, group_x_label, x_order = "Control", f"Control\n(Dis={control_dis})", 0
        elif abs(current_dis - exp_dis) < 1e-4:
            group_label, group_x_label, x_order = "Experimental", f"Experimental\n(Dis={exp_dis})", 1
        else:
            continue

        try:
            df_temp = pd.read_csv(csv_path)
            row = df_temp.iloc[(df_temp['Time_s'] - TIME_POINT).abs().argsort()[:1]]
            if not row.empty:
                rate = row[f"{TARGET_MAT}_SuccessRate"].values[0]
                data_points.append(
                    {'Group': group_label, 'GroupXLabel': group_x_label, 'Dis': current_dis, 'Order': x_order,
                     'SuccessRate': rate})
        except:
            pass

    if len(data_points) < 2:
        return

    df_plot = pd.DataFrame(data_points).sort_values('Order')
    col_yap = f'YAP_k{OPTIMAL_K:.2f}'
    df_plot[col_yap] = df_plot['SuccessRate'].apply(lambda r: transform_to_yap(r, OPTIMAL_K))

    excel_path = os.path.join(result_folder, f"Comparison_Data_{target_threshold}pN.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_plot.to_excel(writer, sheet_name='Raw_Data_All', index=False)
        df_plot[['Group', 'Dis', 'SuccessRate']].to_excel(writer, sheet_name='Compare_SuccessRate', index=False)
        df_plot[['Group', 'Dis', col_yap]].to_excel(writer, sheet_name='Compare_YAP', index=False)

    set_cns_style()

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(df_plot['GroupXLabel'], df_plot['SuccessRate'], marker='o', markersize=14, linestyle='-', linewidth=3,
            color='#1F77B4', markerfacecolor='white', markeredgewidth=2.5)
    for _, row in df_plot.iterrows():
        ax.annotate(f"{row['SuccessRate']:.3f}", xy=(row['GroupXLabel'], row['SuccessRate']), xytext=(0, 15),
                    textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#1F77B4')
    ax.set_title(f"Success Rate Comparison\n(Force = {target_threshold} pN)", pad=20)
    ax.set_ylabel("Success Rate ($P_r$)")
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='.85')
    sns.despine(trim=True, offset=10)
    plt.savefig(os.path.join(result_folder, f"Compare_SuccessRate_{target_threshold}pN.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(df_plot['GroupXLabel'], df_plot[col_yap], marker='s', markersize=14, linestyle='-', linewidth=3,
            color='#D62728', markerfacecolor='white', markeredgewidth=2.5)
    unified_ylim = (0.8, max(2.6, df_plot[col_yap].max() * 1.1))
    for _, row in df_plot.iterrows():
        ax.annotate(f"{row[col_yap]:.2f}", xy=(row['GroupXLabel'], row[col_yap]), xytext=(0, 15),
                    textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#D62728')
    ax.set_title(f"YAP Ratio ($k={OPTIMAL_K:.2f}$)\n(Force = {target_threshold} pN)", pad=20, color='black')
    ax.set_ylabel("Predicted YAP Ratio")
    ax.set_ylim(unified_ylim)
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='.85')
    sns.despine(trim=True, offset=10)
    plt.savefig(os.path.join(result_folder, f"Compare_YAP_{target_threshold}pN.png"))
    plt.close()


if __name__ == "__main__":
    run_comparison_analysis(9.0, 1.0, 0.5)