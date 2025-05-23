from flask import Flask, render_template, request
from pulp import *
from shapely.geometry import Polygon, LineString
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ Usar backend no interactivo para evitar errores en Mac
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        objetivo = request.form['objetivo']
        coef_obj = list(map(float, request.form['coef_obj'].split()))
        restricciones = []
        rhs = []
        operadores = []

        for i in range(int(request.form['num_rest'])):
            coef = list(map(float, request.form[f'coef_rest_{i}'].split()))
            restricciones.append(coef)
            operadores.append(request.form[f'op_rest_{i}'])
            rhs.append(float(request.form[f'rhs_{i}']))

        resultado, variables, z, holguras, plot_url = resolver_programacion_lineal(
            coef_obj, restricciones, rhs, operadores, objetivo)
        return render_template('resultado.html', resultado=resultado, variables=variables, z=z, holguras=holguras, plot_url=plot_url)

    return render_template('index.html')


def resolver_programacion_lineal(coef_objetivo, restricciones, rhs, operadores, objetivo):
    model = LpProblem("PL_Dinamico", LpMaximize if objetivo == 'max' else LpMinimize)
    num_vars = len(coef_objetivo)
    variables = [LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)]

    model += lpSum([coef_objetivo[i] * variables[i] for i in range(num_vars)])

    for i, restriccion in enumerate(restricciones):
        expr = lpSum([restriccion[j] * variables[j] for j in range(num_vars)])
        if operadores[i] == "<=":
            model += expr <= rhs[i]
        elif operadores[i] == ">=":
            model += expr >= rhs[i]
        else:
            model += expr == rhs[i]

    model.solve()

    resultado = LpStatus[model.status]
    var_result = [(var.name, var.varValue) for var in variables]
    z = value(model.objective)
    holguras = [f"R{i+1}: {model.constraints[list(model.constraints.keys())[i]].slack}" for i in range(len(model.constraints))]

    plot_url = None
    if len(variables) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Región Factible y Solución Óptima", fontsize=14)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True)

        x = np.linspace(0, 25, 400)
        lines = []
        colors = ['r', 'g', 'b', 'orange', 'purple']

        for i, restriccion in enumerate(restricciones):
            a, b = restriccion
            if b != 0:
                y = (rhs[i] - a * x) / b
                ax.plot(x, y, label=f"R{i+1}", color=colors[i % len(colors)])
                lines.append(lambda xi, a=a, b=b, rhs=rhs[i]: (rhs - a*xi)/b)
            elif a != 0:
                xi = rhs[i] / a
                ax.axvline(x=xi, label=f"R{i+1}", color=colors[i % len(colors)])

        # Buscar intersecciones (puntos candidatos)
        puntos = []

        def interseccion(a1, b1, c1, a2, b2, c2):
            det = a1 * b2 - a2 * b1
            if det == 0:
                return None
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            return (x, y)

        for (i1, r1), (i2, r2) in combinations(enumerate(restricciones), 2):
            pt = interseccion(*r1, rhs[i1], *r2, rhs[i2])
            if pt and all(p >= 0 for p in pt):
                if all(eval_restriccion(pt, restricciones[j], rhs[j], operadores[j]) for j in range(len(restricciones))):
                    puntos.append(pt)

        # También agregar intersección con ejes (x=0, y=0)
        for i, r in enumerate(restricciones):
            a, b = r
            if b != 0:
                y = rhs[i] / b
                if y >= 0 and eval_restriccion((0, y), r, rhs[i], operadores[i]):
                    puntos.append((0, y))
            if a != 0:
                x = rhs[i] / a
                if x >= 0 and eval_restriccion((x, 0), r, rhs[i], operadores[i]):
                    puntos.append((x, 0))

        if puntos:
            poly = Polygon(puntos)
            x_poly, y_poly = poly.exterior.xy
            ax.fill(x_poly, y_poly, color='lightblue', alpha=0.4, label="Región factible")

        # Punto óptimo
        sol_x = var_result[0][1]
        sol_y = var_result[1][1]
        ax.plot(sol_x, sol_y, 'ko', label=f"Óptimo ({sol_x:.2f}, {sol_y:.2f})")

        ax.legend(loc="upper right")

        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return resultado, var_result, z, holguras, plot_url


def eval_restriccion(punto, coef, rhs, op):
    x, y = punto
    res = coef[0] * x + coef[1] * y
    if op == "<=":
        return res <= rhs + 1e-5
    elif op == ">=":
        return res >= rhs - 1e-5
    else:
        return abs(res - rhs) < 1e-5

if __name__ == "__main__":
    app.run(debug=True)
