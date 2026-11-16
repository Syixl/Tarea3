"""
Tarea 3 - Ejercicio 1 (Opcion A): ajuste de la temperatura del Recocido Simulado con DOE.

Diseno factorial completo 3 x 3:
  - Factor      : temperatura inicial T0 (3 niveles: bajo, medio, alto)
  - Bloque      : tamano del TSP, n in {5, 10, 15} (una instancia fija por tamano)
  - Replicas    : 10 corridas por punto de diseno (90 corridas en total)

Pasos: simulacion -> ANOVA de dos factores -> bootstrap de la mediana -> graficas -> PDF.

Uso:
    python tarea3_doe_temperatura.py 1234567        (tu matricula como argumento)
o bien edita la constante MATRICULA de abajo.
"""
import math
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.formula.api import ols

# ===================== CONFIGURACION =====================
MATRICULA = 2116120
REPO_URL = "https://github.com/Syixl/Tarea3"

SIZES = [5, 10, 15]                   # bloque: tamano del problema
REPLICAS = 10                         # replicas por punto de diseno
ITERS = 250                           # iteraciones de SA por corrida (iguales para todos). Presupuesto
                                      # limitado a proposito: con miles de iteraciones SA halla el optimo
                                      # siempre y no habria variacion que analizar
T_MIN, T_MAX = 1.0, 100.0             # rango de temperatura definido por mi (ver README / reporte)
T_FINAL = 0.05                        # temperatura final fija (solo se ajusta T0)
N_BOOT = 10_000                       # remuestreos bootstrap
ALPHA = 0.05
LEVEL_NAMES = ["bajo", "medio", "alto"]

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


# ===================== TSP + RECOCIDO SIMULADO =====================
def make_instance(n: int):
    """Una instancia fija por tamano (la semilla depende solo de n, no de la matricula)."""
    rng = random.Random(1000 + n)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return [[math.dist(a, b) for b in pts] for a in pts]


def tour_length(D, tour):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def simulated_annealing(D, T0, iters, rng):
    """SA con vecindad 2-opt, inicio aleatorio y enfriamiento exponencial T0 -> T_FINAL."""
    n = len(D)
    tour = list(range(n))
    rng.shuffle(tour)
    cost = tour_length(D, tour)
    best, best_tour = cost, tour[:]
    ratio = T_FINAL / T0
    for t in range(iters):
        T = T0 * ratio ** (t / (iters - 1))
        i = rng.randint(1, n - 2)
        k = rng.randint(i + 1, n - 1)
        a, b = tour[i - 1], tour[i]
        c, d = tour[k], tour[(k + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:k + 1] = tour[i:k + 1][::-1]
            cost += delta
            if cost < best - 1e-9:
                best, best_tour = cost, tour[:]
    return tour_length(D, best_tour)     # se recalcula exacto (evita ruido de redondeo acumulado)


# ===================== ANOVA DE DOS FACTORES =====================
def anova_two_way(df):
    """ANOVA de dos factores con interaccion: costo ~ temperatura * tamano."""
    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)
    tabla.index = ["Temperatura", "Tamaño", "Temperatura × Tamaño", "Error (residual)"]
    return (tabla.reset_index(names="Fuente")
                 .rename(columns={"sum_sq": "Suma cuad.", "df": "gl", "PR(>F)": "p-valor"}))


# ===================== BOOTSTRAP =====================
def bootstrap_median(x, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ===================== PDF =====================
def text_page(pdf, title, paragraphs, size=10):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, title, fontsize=15, fontweight="bold", va="top")
    y = 0.90
    for p in paragraphs:
        lines = []
        for chunk in p.split("\n"):
            lines += textwrap.wrap(chunk, 95) or [""]
        for ln in lines:
            fig.text(0.08, y, ln, fontsize=size, va="top", family="DejaVu Sans")
            y -= 0.0205
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df, note=None, col_widths=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", cellLoc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    if note:
        fig.text(0.05, 0.05, "\n".join(textwrap.wrap(note, 140)), fontsize=9, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    matricula = MATRICULA
    if len(sys.argv) > 1:
        matricula = int(sys.argv[1])
    if matricula is None:
        sys.exit("Falta tu matricula: ejecuta  python tarea3_doe_temperatura.py <matricula>  "
                 "o edita MATRICULA en el archivo.")

    # ---- 1) r se genera UNA sola vez y se usa en todo el experimento ----
    random.seed(matricula)
    r = random.random()
    fractions = [0.3 * r, 0.3 * r + 0.3, 0.3 * r + 0.6]                   # Opcion A
    temps = [T_MIN + f * (T_MAX - T_MIN) for f in fractions]
    print(f"Matrícula = {matricula}   r = {r:.6f}")
    for nm, f, T in zip(LEVEL_NAMES, fractions, temps):
        print(f"  nivel {nm:<6}: fracción = {f:.4f}  ->  T0 = {T:.3f}")

    # ---- 2) simulacion: 3 temperaturas x 3 tamanos x 10 replicas = 90 corridas ----
    instances = {n: make_instance(n) for n in SIZES}
    rows = []
    for si, n in enumerate(SIZES):
        for ti, (nm, T0) in enumerate(zip(LEVEL_NAMES, temps)):
            for rep in range(REPLICAS):
                seed = 100_000 * (si + 1) + 1_000 * (ti + 1) + rep     # semilla distinta por corrida
                cost = simulated_annealing(instances[n], T0, ITERS, random.Random(seed))
                rows.append(dict(tamano=n, temperatura=nm, T0=T0, replica=rep + 1, costo=cost))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tarea3_datos_corridas.csv", index=False)
    print(f"\n{len(df)} corridas realizadas ({ITERS:,} iteraciones cada una)")

    # ---- 3) ANOVA de dos factores ----
    anova = anova_two_way(df)
    anova.to_csv(RESULTS / "tarea3_anova.csv", index=False)
    print("\nANOVA de dos factores (alpha = 0.05):")
    print(anova.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    sig = {row["Fuente"]: row["p-valor"] < ALPHA for _, row in anova.iloc[:3].iterrows()}

    # ---- 4) bootstrap de la mediana por punto de diseno + prueba por tamano ----
    brng = np.random.default_rng(matricula)
    boot = {}
    for n in SIZES:
        for nm in LEVEL_NAMES:
            x = df[(df.tamano == n) & (df.temperatura == nm)].costo.to_numpy()
            boot[(n, nm)] = bootstrap_median(x, brng)
    per_size_p = {}
    for n in SIZES:
        groups = [df[(df.tamano == n) & (df.temperatura == nm)].costo.to_numpy() for nm in LEVEL_NAMES]
        # si todas las corridas dan lo mismo (p. ej. n = 5 siempre alcanza el optimo), no hay variacion
        per_size_p[n] = 1.0 if np.ptp(np.concatenate(groups)) == 0 else stats.f_oneway(*groups).pvalue

    boot_rows = [dict(tamano=n, nivel=nm, T0=temps[LEVEL_NAMES.index(nm)],
                      mediana_boot=boot[(n, nm)][0], ic_inf=boot[(n, nm)][1], ic_sup=boot[(n, nm)][2])
                 for n in SIZES for nm in LEVEL_NAMES]
    pd.DataFrame(boot_rows).to_csv(RESULTS / "tarea3_bootstrap_todos.csv", index=False)

    summary = []
    for n in SIZES:
        best = min(LEVEL_NAMES, key=lambda nm: (boot[(n, nm)][0], LEVEL_NAMES.index(nm)))
        med, lo, hi = boot[(n, best)]
        p = per_size_p[n]
        summary.append(dict(n=n, nivel=best, T0=temps[LEVEL_NAMES.index(best)], med=med, lo=lo, hi=hi,
                            p=p, signif=p < ALPHA))
    sdf = pd.DataFrame(summary)

    print("\nMejor temperatura por tamaño (menor mediana bootstrap del costo):")
    print(f"{'n':>3} {'nivel':<7}{'T0':>9}{'mediana boot.':>15}{'IC 95%':>24}{'p (ANOVA por n)':>17}")
    for _, s in sdf.iterrows():
        print(f"{int(s.n):>3} {s.nivel:<7}{s.T0:>9.2f}{s.med:>15.2f}{f'[{s.lo:.2f}, {s.hi:.2f}]':>24}{fmt_p(s.p):>17}")
    sdf.to_csv(RESULTS / "tarea3_resumen_optimo.csv", index=False)

    # ---- 5) graficas ----
    colors = {5: "tab:blue", 10: "tab:orange", 15: "tab:green"}
    xs = np.arange(3)
    xt = [f"{nm}\nT0={T:.1f}" for nm, T in zip(LEVEL_NAMES, temps)]

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    m_t = [df[df.temperatura == nm].costo.mean() for nm in LEVEL_NAMES]
    e_t = [1.96 * df[df.temperatura == nm].costo.sem() for nm in LEVEL_NAMES]
    axes[0].errorbar(xs, m_t, yerr=e_t, fmt="o-", capsize=4, color="tab:purple")
    axes[0].set_xticks(xs, xt)
    axes[0].set_title(f"Efecto principal: temperatura\n(ANOVA p = {fmt_p(anova.iloc[0]['p-valor'])})")
    axes[0].set_ylabel("Costo medio (± IC 95%)")
    m_s = [df[df.tamano == n].costo.mean() for n in SIZES]
    e_s = [1.96 * df[df.tamano == n].costo.sem() for n in SIZES]
    axes[1].errorbar(range(3), m_s, yerr=e_s, fmt="o-", capsize=4, color="tab:red")
    axes[1].set_xticks(range(3), [f"n = {n}" for n in SIZES])
    axes[1].set_title(f"Efecto principal: tamaño\n(ANOVA p = {fmt_p(anova.iloc[1]['p-valor'])})")
    axes[1].set_ylabel("Costo medio (± IC 95%)")
    for a in axes:
        a.grid(alpha=0.3)
    fig1.suptitle("Gráfica de efectos principales")
    fig1.tight_layout()
    fig1.savefig(RESULTS / "tarea3_efectos_principales.png", dpi=150)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n in SIZES:
        means = [df[(df.tamano == n) & (df.temperatura == nm)].costo.mean() for nm in LEVEL_NAMES]
        axes[0].plot(xs, means, "o-", color=colors[n], label=f"n = {n}")
        axes[1].plot(xs, np.array(means) / np.mean(means), "o-", color=colors[n], label=f"n = {n}")
    axes[0].set_title(f"Interacción temperatura × tamaño\n(ANOVA p = {fmt_p(anova.iloc[2]['p-valor'])})")
    axes[0].set_ylabel("Costo medio")
    axes[1].set_title("Misma gráfica, costo relativo a la media de cada n\n(para comparar la forma de las curvas)")
    axes[1].set_ylabel("Costo medio / media del tamaño")
    for a in axes:
        a.set_xticks(xs, xt)
        a.grid(alpha=0.3)
        a.legend()
    fig2.suptitle("Gráfica de interacción (líneas no paralelas = interacción)")
    fig2.tight_layout()
    fig2.savefig(RESULTS / "tarea3_interaccion.png", dpi=150)

    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, n in zip(axes, SIZES):
        med = [boot[(n, nm)][0] for nm in LEVEL_NAMES]
        lo = [boot[(n, nm)][0] - boot[(n, nm)][1] for nm in LEVEL_NAMES]
        hi = [boot[(n, nm)][2] - boot[(n, nm)][0] for nm in LEVEL_NAMES]
        ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", capsize=5, color=colors[n])
        for i, nm in enumerate(LEVEL_NAMES):
            vals = df[(df.tamano == n) & (df.temperatura == nm)].costo
            ax.scatter(np.full(len(vals), i) + 0.12, vals, s=8, color="gray", alpha=0.5)
        b = sdf[sdf.n == n].iloc[0]
        ax.set_title(f"n = {n}   (mejor: {b.nivel}, T0 = {b.T0:.1f})")
        ax.set_xticks(xs, xt)
        ax.set_ylabel("Costo")
        ax.grid(alpha=0.3)
    fig3.suptitle("Mediana bootstrap e IC 95% por punto de diseño (puntos grises = corridas individuales)")
    fig3.tight_layout()
    fig3.savefig(RESULTS / "tarea3_bootstrap_por_tamano.png", dpi=150)

    # ---- 6) reporte PDF ----
    names = ["Temperatura", "Tamaño", "Temperatura × Tamaño"]
    def verdict(nm):
        p = anova.loc[anova.Fuente == nm, "p-valor"].iloc[0]
        return f"{nm}: p = {fmt_p(p)} -> {'SIGNIFICATIVO' if p < ALPHA else 'no significativo'}"

    idx = [LEVEL_NAMES.index(l) for l in sdf.nivel]
    if len(set(idx)) == 1:
        trend = (f"El mejor nivel fue el mismo ({sdf.nivel.iloc[0]}) para los tres tamaños, por lo que no se "
                 "observa un cambio de T* con n en este experimento.")
    elif idx == sorted(idx):
        trend = "T* tiende a AUMENTAR con el tamaño del problema (n = 5 -> 10 -> 15)."
    elif idx == sorted(idx, reverse=True):
        trend = "T* tiende a DISMINUIR con el tamaño del problema (n = 5 -> 10 -> 15)."
    else:
        trend = "No hay una tendencia monótona clara entre n y T*."
    not_sig = [int(n) for n, s in zip(sdf.n, sdf.signif) if not s]
    caveat = ""
    if not_sig:
        caveat = (f" Para n = {', '.join(map(str, not_sig))} la ANOVA por tamaño no encontró diferencias "
                  "significativas entre temperaturas, así que el 'mejor' nivel de ese tamaño no se distingue "
                  "estadísticamente de los demás (sus IC se traslapan).")

    with PdfPages(RESULTS / "tarea3_reporte.pdf") as pdf:
        text_page(pdf, "Tarea 3: ajuste de temperatura del Recocido Simulado con DOE", [
            f"Repositorio: {REPO_URL}",
            "Enfoque elegido: Opción A, diseño factorial completo 3 × 3 con ANOVA de dos factores.",
            "Diseño: factor = temperatura inicial T0 (3 niveles); bloque = tamaño del TSP n ∈ {5, 10, 15} "
            f"(una instancia fija por tamaño); {REPLICAS} réplicas por punto de diseño con semillas distintas "
            f"= {len(df)} corridas. Cada corrida usa {ITERS:,} iteraciones, vecindad 2-opt, inicio aleatorio y "
            f"enfriamiento exponencial de T0 a T_final = {T_FINAL}.",
            f"Rango de temperatura definido: [T_mín, T_máx] = [{T_MIN}, {T_MAX}]. Justificación: las ciudades "
            "están en un cuadrado de 100 × 100, con arista media cercana a 50; T_máx ≈ 2× la arista media "
            "(acepta casi cualquier movimiento) y T_mín = 1 es un SA casi voraz.",
            f"Número aleatorio fijado una sola vez: random.seed({matricula}); r = {r:.6f}.",
            "Niveles (T = T_mín + fracción × (T_máx − T_mín)):\n" + "\n".join(
                f"   {nm}: fracción = {f:.4f}  ->  T0 = {T:.3f}" for nm, f, T in zip(LEVEL_NAMES, fractions, temps)),
            "Procedimiento: (1) simulación de las 90 corridas; (2) ANOVA de dos factores con interacción a "
            "α = 0.05; (3) mediana bootstrap (10,000 remuestreos, IC percentil 95%) del costo en cada punto de "
            "diseño; (4) el mejor T0 de cada tamaño es el nivel con menor mediana bootstrap, apoyado por una "
            "ANOVA de un factor dentro de cada n.",
        ])
        table_page(pdf, "ANOVA de dos factores (variable respuesta: costo del tour)",
                   anova.assign(**{"Suma cuad.": anova["Suma cuad."].map(lambda v: f"{v:.2f}"),
                                   "gl": anova["gl"].astype(int),
                                   "F": anova["F"].map(lambda v: "" if np.isnan(v) else f"{v:.3f}"),
                                   "p-valor": anova["p-valor"].map(lambda v: "" if np.isnan(v) else fmt_p(v))}),
                   note="  |  ".join(verdict(nm) for nm in names))
        for fig in (fig1, fig2, fig3):
            pdf.savefig(fig)
        tab = pd.DataFrame({
            "n": sdf.n.astype(int), "Mejor nivel": sdf.nivel, "Temperatura T0": sdf.T0.map(lambda v: f"{v:.2f}"),
            "Mediana bootstrap": sdf.med.map(lambda v: f"{v:.2f}"),
            "IC 95%": [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(sdf.lo, sdf.hi)],
            "p (ANOVA por n)": sdf.p.map(fmt_p),
            "¿Efecto de T significativo?": sdf.signif.map(lambda s: "Sí" if s else "No"),
        })
        table_page(pdf, "Temperatura óptima por tamaño", tab,
                   note="El mejor nivel es el de menor mediana bootstrap del costo. Si el efecto no es significativo, "
                        "las diferencias entre niveles no se distinguen del ruido del algoritmo.")
        text_page(pdf, "Análisis", [
            "ANOVA de dos factores (α = 0.05): la temperatura no es significativa (p = 0.5147); el tamaño sí "
            "(p ≈ 8.6e-77), algo esperable porque tours con más ciudades son más largos; la interacción "
            "temperatura × tamaño tampoco es significativa (p = 0.9465).",
            "Mejor temperatura por tamaño (menor mediana bootstrap): n = 5 -> nivel medio (T0 = 51.58); "
            "n = 10 -> nivel bajo (T0 = 21.88); n = 15 -> nivel medio (T0 = 51.58). La ANOVA por tamaño no "
            "encontró diferencias significativas entre temperaturas en ningún caso (p = 0.375, 0.159 y 0.819), "
            "por lo que estos niveles son ganadores nominales que no se distinguen estadísticamente de los demás.",
            "¿La temperatura óptima cambia con el tamaño? No hay evidencia de ello ni de una tendencia entre n y "
            "T*: el mejor nivel fue medio, bajo y medio, sin patrón monótono, y las diferencias no son "
            "significativas. Probablemente, con enfriamiento exponencial hasta una temperatura final fija, el "
            "algoritmo refina a temperaturas bajas sin importar T0; con n = 5 casi todas las corridas alcanzan "
            "el óptimo y con n = 15 la dispersión entre réplicas oculta cualquier diferencia.",
            "Limitaciones: una sola instancia por tamaño, 10 réplicas por punto de diseño y 250 iteraciones "
            "por corrida. Con más réplicas, más iteraciones o instancias mayores el efecto de la temperatura "
            "podría hacerse detectable.",
        ])
    print(f"\nReporte PDF y gráficas guardados en {RESULTS}")


if __name__ == "__main__":
    main()