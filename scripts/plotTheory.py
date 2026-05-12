'''Analytical theory plots for the chain model.

Saves individual panel PNGs to figures/panels/ and a combined figure to
figures/theory_combined.png.

Panels:
  1  Rescue probability upper bound (Lambda) vs dose
  2  Mutation-supply ratio Lambda_BF/Lambda_WM vs b_c
  3  Core-delivery rate Gamma_core vs dose
  4  Drift-survival probability ln(1+a)/a vs alpha
  5  Establishment suppression: p_est(k) for BF Phase 1 vs WM
  6  Full approximate rescue ratio P_BF/P_WM vs dose (includes suppression)

Usage:
    cd /path/to/chain-model
    python3 scripts/plotTheory.py
    python3 scripts/plotTheory.py --outdir figures
'''
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # numpy < 2.0 compatibility

N0      = 1000
L       = 100
B_E     = 1.0
B_R     = 1.0
D_R     = 0.6
R_STAR  = 10
MU      = 1e-4
B_C_DEF = 0.2

DOSES = np.linspace(1.05, 2.85, 400)
S_E   = DOSES - 1.0

B_C_VALS   = [0.0, 0.05, 0.1, 0.2, 0.4]
CMAP_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, len(B_C_VALS)))




def lam_wm(s_e):
    return MU * B_E * N0 / s_e

def lam_bf(s_e, b_c):
    return (MU / s_e) * (b_c * (N0 - L)**2 / (2 * L) + B_E * N0)

def alpha(s_e, b_c):
    return b_c * (N0 - L) / (s_e * L)

def gamma_core_per_mu(s_e, b_c):
    return s_e * L * np.log1p(alpha(s_e, b_c))




def _gamblers_ruin_P_arr(b_e, b_r, d_r, l, r_star):
    '''Compute the "scale function" array P[k-1] = prod_{j=1}^{k-1} r_j,
    where r_j = d_j/u_j, for k = 1 ... r_star.'''
    P = np.ones(r_star)          # P[0] = 1 (k=1)
    for k in range(1, r_star):   # fill P[k] = P[k-1] * r_k
        r_k = (b_e * (l - k) / l + d_r) / (b_r * (l - k) / l)
        P[k] = P[k - 1] * r_k
    return P

def p_est_bf_arr(b_e=B_E, b_r=B_R, d_r=D_R, l=L, r_star=R_STAR):
    '''p_est(k) for k=1..r_star in the BF Phase-1 chain (WT resupply active).
    Returns array of length r_star; index k-1 holds p_est(k).'''
    P = _gamblers_ruin_P_arr(b_e, b_r, d_r, l, r_star)
    cs = np.cumsum(P)
    return cs / cs[-1]

def p_est_wm_arr(b_r=B_R, d_r=D_R, r_star=R_STAR):
    '''p_est(k) for k=1..r_star in the WM chain (simple BDP, constant rates).'''
    rho = d_r / b_r
    k   = np.arange(1, r_star + 1)
    return (1 - rho**k) / (1 - rho**r_star)



def p_est_bolus(tau, b_c, p_bf):
    '''p_est for a bolus that survived drift time tau and now enters the BF
    Phase-1 edge chain.  Bolus size ~ Geom(1/(1+b_c*tau)) conditioned on
    survival.  p_bf[k-1] = p_est(k) in the BF chain.'''
    r_star = len(p_bf)
    if b_c * tau < 1e-12:
        return p_bf[0]
    p = 1.0 / (1.0 + b_c * tau)
    q = 1.0 - p
    result = sum(p * q**(k - 1) * p_bf[k - 1] for k in range(1, r_star + 1))
    result += q**r_star   # P(size > r*) → already rescued
    return result

def P_rescue_core(s_e, b_c, p_bf, n_pts=400):
    '''Rescue probability contribution from the core-delivery pathway.
    Integrates over uniform birth depths d in [0, N0-L], accounting for
    drift-survival probability and bolus-delivery establishment.'''
    if b_c == 0:
        return 0.0
    d    = np.linspace(0, N0 - L - 1, n_pts)
    tau  = d / (s_e * L)
    psurv = 1.0 / (1.0 + b_c * tau)
    pest  = np.array([p_est_bolus(t, b_c, p_bf) for t in tau])
    return MU * b_c * np.trapezoid(psurv * pest, d)



def P_rescue_bf_approx(s_e, b_c):
    p_bf  = p_est_bf_arr()
    p_wm1 = p_est_wm_arr()[0]          # p_est(1) for WM
    # edge-born Phase 1 (with WT resupply suppression)
    L1    = MU * B_E * (N0 - L) / s_e
    # edge-born Phase 2 (no resupply, WM-like)
    L2    = MU * B_E * L / s_e
    return L1 * p_bf[0] + L2 * p_wm1 + P_rescue_core(s_e, b_c, p_bf)

def P_rescue_wm_approx(s_e):
    p_wm1 = p_est_wm_arr()[0]
    return MU * B_E * N0 / s_e * p_wm1



def draw_panel1(ax):
    '''P(rescue) upper bound (Lambda only) vs dose.'''
    ax.plot(DOSES, 1 - np.exp(-lam_wm(S_E)), 'k--', lw=2,
            label='well-mixed', zorder=5)
    for b_c, col in zip(B_C_VALS, CMAP_COLORS):
        ls    = ':' if b_c == 0 else '-'
        label = rf'BF $b_c={b_c}$' + (r' (dormant)' if b_c == 0 else '')
        ax.plot(DOSES, 1 - np.exp(-lam_bf(S_E, b_c)), color=col,
                ls=ls, lw=1.8, label=label)
    ax.axvline(1 + B_C_DEF * (N0 - L) / L, color='gray', ls=':', lw=1,
               label=rf'$s_e^*$ (default $b_c$)')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$ — upper bound $1-e^{-\Lambda}$')
    ax.set_title('Rescue Probability vs Dose')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

def draw_panel2(ax):
    '''Lambda ratio vs b_c.'''
    bc  = np.linspace(0, 0.6, 400)
    rat = 1 + bc * (N0 - L)**2 / (2 * B_E * N0 * L)
    ax.plot(bc, rat, 'C0', lw=2)
    ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.5,
               label='well-mixed baseline')
    ax.axvline(B_C_DEF, color='gray', ls=':', lw=1,
               label=rf'default $b_c={B_C_DEF}$')
    rat_def = 1 + B_C_DEF * (N0 - L)**2 / (2 * B_E * N0 * L)
    ax.scatter([B_C_DEF], [rat_def], color='C0', zorder=5, s=50)
    ax.annotate(f'{rat_def:.2f}×',
                xy=(B_C_DEF, rat_def),
                xytext=(B_C_DEF + 0.04, rat_def - 0.2), fontsize=9, color='C0')
    ax.set_xlabel(r'$b_c$ (core birth rate)')
    ax.set_ylabel(r'$\Lambda_{\rm BF}/\Lambda_{\rm WM}$')
    ax.set_title('Mutation-Supply Advantage')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

def draw_panel3(ax):
    '''Core-delivery rate Gamma_core vs dose.'''
    for b_c, col in zip(B_C_VALS[1:], CMAP_COLORS[1:]):
        ax.plot(DOSES, gamma_core_per_mu(S_E, b_c), color=col, lw=1.8,
                label=rf'$b_c={b_c}$')
        ax.axhline(b_c * (N0 - L), color=col, ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$\Gamma_{\rm core}/\mu = s_e l\,\ln(1+\alpha)$')
    ax.set_title('Core-delivery Rate vs Dose')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

def draw_panel4(ax):
    '''Drift-survival probability vs alpha.'''
    a_vals  = np.logspace(-2, 2.5, 500)
    p_surv  = np.log1p(a_vals) / a_vals
    ax.semilogx(a_vals, p_surv, 'C3', lw=2)
    ax.semilogx(a_vals, np.ones_like(a_vals), 'k--', lw=1, alpha=0.4)
    ax.semilogx(a_vals, np.log(np.maximum(a_vals, 1.01)) / a_vals,
                'k:', lw=1, alpha=0.4, label=r'$\ln\alpha/\alpha$')
    ref_se = 0.5   # dose 1.5
    for b_c, col in zip(B_C_VALS[1:], CMAP_COLORS[1:]):
        a = alpha(ref_se, b_c)
        ax.scatter([a], [np.log1p(a) / a], color=col, zorder=5, s=40,
                   label=rf'$b_c={b_c}$, dose 1.5')
    ax.axvline(1.0, color='gray', ls=':', lw=1.2)
    ax.text(1.05, 0.92, r'$\alpha=1$', fontsize=8, color='gray')
    ax.set_xlabel(r'$\alpha = b_c(N_0-l)/(s_e\,l)$')
    ax.set_ylabel(r'$\langle P_{\rm survive}\rangle = \ln(1+\alpha)/\alpha$')
    ax.set_title('Mean Drift-Survival Probability')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which='both')

def draw_panel5(ax):
    '''Establishment suppression: p_est(k) curves for BF vs WM.'''
    k_vals = np.arange(1, R_STAR + 1)
    p_bf   = p_est_bf_arr()
    p_wm   = p_est_wm_arr()

    ax.semilogy(k_vals, p_bf, 'C0-o', ms=5, lw=2,
                label=r'BF Phase 1 (Core Present)')
    ax.semilogy(k_vals, p_wm, 'k--s', ms=5, lw=2,
                label=r'WM')

    ax.axhline(p_bf[0], color='C0', ls=':', lw=1, alpha=0.5)
    ax.axhline(p_wm[0], color='k',  ls=':', lw=1, alpha=0.5)
    ax.text(1.15, p_bf[0] * 1.4, f'{p_bf[0]:.4f}', fontsize=8, color='C0')
    ax.text(1.15, p_wm[0] * 0.6, f'{p_wm[0]:.3f}', fontsize=8, color='k')
    ax.annotate('',
                xy=(0.8, p_bf[0] * 1.05), xytext=(0.8, p_wm[0] * 0.95),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    suppression = p_wm[0] / p_bf[0]
    # ax.text(0.3, np.sqrt(p_bf[0] * p_wm[0]),
    #         f'{suppression:.0f}×\nsuppression', fontsize=8, color='red',
    #         ha='center', va='center')
    ax.set_xlabel(r'R Cells in Edge ($k$)')
    ax.set_ylabel(r'$P_{\rm est}(k)$ — Prob.\ of Reaching $r^*=10$')
    ax.set_title('Establishment Suppression by WT Resupply\n'
                 r'$b_e=1,\,d_R=0.6,\,b_R=1,\,l=100$')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

def draw_panel6(ax):
    '''Full approximate P_BF/P_WM ratio vs dose (includes suppression).'''
    p_wm_arr = np.array([P_rescue_wm_approx(s) for s in S_E])

    for b_c, col in zip(B_C_VALS, CMAP_COLORS):
        p_bf_arr = np.array([P_rescue_bf_approx(s, b_c) for s in S_E])
        ratio    = p_bf_arr / p_wm_arr
        ls       = ':' if b_c == 0 else '-'
        label    = (rf'$b_c={b_c}$' +
                    (r' (dormant)' if b_c == 0 else ''))
        ax.semilogy(DOSES, ratio, color=col, ls=ls, lw=1.8, label=label)

    ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='BF = WM')
    ax.axvline(1 + B_C_DEF * (N0 - L) / L, color='gray', ls=':', lw=1)
    ax.fill_between(DOSES, 0.001, 1.0, alpha=0.05, color='red')
    ax.fill_between(DOSES, 1.0, 100, alpha=0.05, color='green')
    # ax.text(1.1, 0.002,  'BF < WM\n(suppressed)', fontsize=8, color='red')
    # ax.text(2.2, 5.0,    'BF > WM\n(enhanced)',   fontsize=8, color='green')
    ax.set_xlabel(r'Dose $(d_e)$')
    ax.set_ylabel(r'$P_{\rm BF}/P_{\rm WM}$ (approx.)')
    ax.set_title('Rescue Ratio BF / WM')
    ax.set_ylim(1e-3, 50)
    # ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, which='both')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    panels_dir = os.path.join(args.outdir, 'panels')
    os.makedirs(panels_dir, exist_ok=True)

    draw_fns = [
        draw_panel1, draw_panel2, draw_panel3,
        draw_panel4, draw_panel5, draw_panel6,
    ]
    names = [
        'rescue_upper_bound', 'lambda_ratio', 'delivery_rate',
        'drift_survival',     'suppression_pest', 'suppression_pratio',
    ]
    titles = [
        'Panel 1: Rescue Upper Bound',
        'Panel 2: Mutation-Supply Ratio',
        'Panel 3: Core-Delivery Rate',
        'Panel 4: Drift-Survival Probability',
        'Panel 5: Establishment Suppression',
        'Panel 6: Full Rescue Ratio',
    ]

    for fn, name in zip(draw_fns, names):
        print(f'  plotting {name}...')
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        plt.tight_layout()
        out = os.path.join(panels_dir, f'{name}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    saved {out}')

    print('  plotting combined figure...')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    # fig.suptitle('Analytical theory: chain model predictions', fontsize=14)
    for fn, ax, title in zip(draw_fns, axes.flat, titles):
        fn(ax)
        ax.set_title(title + '\n' + ax.get_title().split('\n')[0], fontsize=9)
    plt.tight_layout()
    out = os.path.join(args.outdir, 'theory_combined.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out}')


if __name__ == '__main__':
    main()
