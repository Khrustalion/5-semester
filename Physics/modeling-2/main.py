import numpy as np
import matplotlib.pyplot as plt

# Модель Кронига–Пенни в предельном случае дельта-гребёнки.
# Дисперсионное соотношение:
#
#   cos(k a) = cos(alpha a) + P * sin(alpha a) / (alpha a),
#
# где alpha^2 = E   (мы выбрали единицы, в которых ħ^2 / (2m) = 1),
# a — период решётки, P — безразмерная величина, пропорциональная
# высоте и ширине барьеров (сила потенциала).
#
# Разрешённые энергии — те, для которых | RHS(E) | <= 1.


def kronig_penney_delta_rhs(E, P, a):
    # Правая часть дисперсионного соотношения для заданных E и параметра P.
    E = np.asarray(E, dtype=float)
    alpha = np.sqrt(np.maximum(E, 0.0))
    # избегаем деления на ноль при E -> 0
    x = alpha * a
    rhs = np.cos(x)
    # отдельная обработка точки x ~= 0
    small = np.abs(x) < 1e-8
    not_small = ~small
    rhs[not_small] += P * np.sin(x[not_small]) / x[not_small]
    rhs[small] += P  # предельное значение sin(x)/x -> 1
    return rhs


def find_bands(E_grid, rhs):
    # По сетке энергий и RHS(E) находим интервалы разрешённых зон.
    allowed = np.abs(rhs) <= 1
    bands = []
    if not np.any(allowed):
        return bands

    start = None
    for i, ok in enumerate(allowed):
        if ok and start is None:
            start = i
        if (not ok or i == len(allowed) - 1) and start is not None:
            end = i if not ok else i
            bands.append((E_grid[start], E_grid[end]))
            start = None
    return bands


def compute_band_structure(P, a=1.0, E_max=50.0, n_E=5000):
    # Возвращает точки (k, E) и интервалы разрешённых зон для заданного P.
    E_grid = np.linspace(0.0, E_max, n_E)
    rhs = kronig_penney_delta_rhs(E_grid, P, a)

    allowed_idx = np.where(np.abs(rhs) <= 1)[0]

    k_points = []
    E_points = []
    for idx in allowed_idx:
        val = np.clip(rhs[idx], -1.0, 1.0)
        ka = np.arccos(val)  # в [0, π]
        k_points.append(ka / a)
        E_points.append(E_grid[idx])

    bands = find_bands(E_grid, rhs)
    return np.array(k_points), np.array(E_points), bands


def plot_bands_for_P_list(P_list, a=1.0, E_max=50.0):
    # Рисует зонную структуру для нескольких значений параметра P.
    fig, ax = plt.subplots(figsize=(9, 6))

    for P in P_list:
        if P == 0:
            # Свободный электрон: E = k^2 (в наших единицах)
            k = np.linspace(-np.pi / a, np.pi / a, 400)
            E = k**2
            ax.plot(k, E, label="P = 0 (свободный)")
            continue

        k, E, bands = compute_band_structure(P, a=a, E_max=E_max)
        ax.scatter(k, E, s=2, label=f"P = {P:g}")
        ax.scatter(-k, E, s=2)

        # Выводим ширину первых нескольких запрещённых зон
        print(f"\nP = {P:g}")
        if len(bands) > 1:
            for i in range(len(bands) - 1):
                band_top = bands[i][1]
                next_band_bottom = bands[i + 1][0]
                gap = next_band_bottom - band_top
                print(f"  Щель между зоной {i+1} и {i+2}: ΔE ≈ {gap:.3f}")
        else:
            print("  В рассматриваемом диапазоне энергий только одна зона, щелей нет.")

    ax.set_xlabel("k · a")
    ax.set_ylabel("E (безразмерная)")
    ax.set_xlim(-np.pi / a, np.pi / a)
    ax.set_ylim(0, E_max)
    ax.set_title("Зонная структура в модели Кронига–Пенни (дельта-гребёнка)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = 1.0   # период решётки
    # P = 0  — свободный электрон;
    # большие P соответствуют «почти бесконечным» барьерам.
    P_values = [0.0, 1.0, 5.0, 20.0]

    plot_bands_for_P_list(P_values, a=a, E_max=50.0)
