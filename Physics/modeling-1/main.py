import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# константы
HBAR = 1.0545718e-34        # Дж·с
ELECTRON_M = 9.10938356e-31 # кг
EV_TO_J = 1.602176634e-19   # Дж

def create_plot(U_eV, x, wf_array, bound_energies, width, figname):
    plt.figure(figsize=(8,8))
    plt.plot(x*1e10, U_eV, color='black', lw=2, label='U(x) (эВ)')

    # масштабируем волновые функции для наглядности
    Vspan = U_eV.max() - U_eV.min()
    amp = 0.28 * Vspan if Vspan>0 else 1.0

    for n in range(wf_array.shape[1]):
        psi = wf_array[:, n]
        E = bound_energies[n]
        # нормируем амплитуду и сдвигаем на энергию
        ψ_scaled = psi / np.max(np.abs(psi)) * amp
        plt.plot(x*1e10, ψ_scaled + E, linewidth=1.6, label=f'n={n+1}, E={E:.4f} эВ')

    # границы области ямы (если внутри 0..width)
    plt.axvline(0.0*1e10, color='black', lw=1)
    plt.axvline(width*1e10, color='black', lw=1)

    plt.xlabel('x ($м \cdot 10^{-10}$)')
    plt.ylabel('Энергия (эВ)')
    plt.title('Связанные состояния(рефакторинг)')
    plt.grid(alpha=0.6)
    plt.legend(loc='lower left')

    # подгонка y-диапазона с запасом
    ymin = min(U_eV.min(), bound_energies.min() if bound_energies.size else U_eV.min())
    ymax = max(U_eV.max(), bound_energies.max() if bound_energies.size else U_eV.max())
    pad = 0.12 * (ymax - ymin if ymax>ymin else 1.0)
    plt.ylim(ymin - pad, ymax + pad)

    if figname:
        plt.savefig(figname, dpi=200)

def solve_tise(potential, width,
               padd=1.5e-10,
               points=900, mass=ELECTRON_M, n_levels=6, figname=None):
    # --- сетка ---
    x0 = -padd
    x1 = width + padd
    x = np.linspace(x0, x1, points)
    dx = x[1] - x[0]

    # --- формируем потенциал в эВ ---
    if callable(potential):
        U_eV = np.asarray(potential(x), dtype=float)
    else:
        U_eV = np.asarray(potential, dtype=float)

    # переводим в джоули (матрица Гамильтониана в СИ)
    U_J = U_eV * EV_TO_J

    # --- внутренние узлы (убираем крайние, задаём Dirichlet psi=0) ---
    x_inner = x[1:-1]
    U_inner = U_J[1:-1]
    N_inner = x_inner.size

    # --- дискретный лапласиан: (psi_{i-1} - 2 psi_i + psi_{i+1})/dx^2 ---
    lap_diag = np.full(N_inner, -2.0) / dx**2
    lap_off  = np.full(N_inner - 1, 1.0) / dx**2
    # кинетический оператор: -(hbar^2 / 2m) * d2/dx2  =>  -coef * lap
    coef = HBAR**2 / (2.0 * mass)
    T = -coef * diags([lap_off, lap_diag, lap_off], offsets=[-1,0,1], format='csr')

    # потенциальная часть (диагональная)
    V = diags(U_inner, 0, format='csr')

    # Гамильтониан
    H = T + V

    # --- решаем k наименьших собственных значений ---
    k = min(n_levels + 6, N_inner-2)  # немного больше, чтобы потом отобрать связанные
    try:
        evals, evecs = eigsh(H, k=k, which='SA', tol=1e-8, maxiter=10000)
    except Exception as exc:
        # если eigsh не сошёлся, упрощённая попытка с меньшим k
        k2 = min(n_levels, N_inner-2)
        evals, evecs = eigsh(H, k=k2, which='SA', tol=1e-8, maxiter=10000)

    # сортируем по возрастанию энергии
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # переводим в эВ
    energies_eV = evals / EV_TO_J

    # --- находим асимптотическое значение потенциала (вне ямы) в эВ ---
    U_asympt_eV = max(U_eV[0], U_eV[-1])

    # --- отбираем связанные состояния (E < U_asympt) ---
    eps = 1e-9
    bound_mask = energies_eV < (U_asympt_eV - eps)
    bound_indices = np.where(bound_mask)[0]

    # собираем волновые функции (включая края с нулями)
    wf_list = []
    bound_energies = []
    for idx in bound_indices[:n_levels]:
        psi_inner = evecs[:, idx]
        # нормировка по dx
        norm = np.sqrt(np.trapz(np.abs(psi_inner)**2, x_inner))
        psi_inner = psi_inner / norm
        psi_full = np.zeros_like(x)
        psi_full[1:-1] = psi_inner
        wf_list.append(psi_full)
        bound_energies.append(energies_eV[idx])

    bound_energies = np.array(bound_energies)
    wf_array = np.column_stack(wf_list) if wf_list else np.zeros((x.size,0))

    create_plot(U_eV, x, wf_array, bound_energies, width, figname)
   
    return {
        'x_m': x,
        'U_eV': U_eV,
        'energies_eV': energies_eV,
        'bound_energies_eV': bound_energies,
        'wf_bound': wf_array,
        'n_bound': wf_array.shape[1],
        'U_asympt_eV': U_asympt_eV
    }

def rect_well(U0_eV, width):
    def U(x):
        return np.where((x>=0) & (x<=width), -abs(U0_eV), 0.0)
    return U

def triang_well(U0_eV, a):
    def U(x):
        return np.where((x>=0) & (x<=a), -abs(U0_eV)*(1 - x/a), 0.0)
    return U

def parabolic_well(U0_eV, width):
    def U(x):
        inside = (x>=0) & (x<=width)
        y = np.zeros_like(x, dtype=float)
        xm = width/2.0
        y[inside] = -abs(U0_eV) * (1.0 - ((x[inside]-xm)/xm)**2)
        return y
    return U

# def from_table(x_table, U_table, width, padd=1.5e-10, points=900):
#     x_all = np.linspace(-padd, width + padd, points)
#     interp = np.interp(x_all, x_table, U_table, left=U_table[0], right=U_table[-1])
#     return interp  # возвращаем массив, который можно передать в solver

if __name__ == "__main__":
    U0 = 50.0            # глубина (эВ)
    a = 2.7e-10          # ширина (м)
    pad = 1.5e-10
    result = solve_tise(rect_well(U0, a), width=a, padd=pad,
                        points=900, mass=ELECTRON_M, n_levels=6,
                        figname="rectangle.png")
    
    result = solve_tise(parabolic_well(U0, a), width=a, padd=pad,
                        points=900, mass=ELECTRON_M, n_levels=6,
                        figname="porabolic.png")
    
    result = solve_tise(triang_well(U0, a), width=a, padd=pad,
                        points=900, mass=ELECTRON_M, n_levels=6,
                        figname="traingle.png")

    print("Найдено связанных состояний:", result['n_bound'])
    for i, E in enumerate(result['bound_energies_eV'], start=1):
        print(f" n={i}  E = {E:.6f} эВ (U_inf={result['U_asympt_eV']:.3f} эВ)")
