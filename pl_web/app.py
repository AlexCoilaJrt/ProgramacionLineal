
from flask import Flask, render_template, request
from pulp import *
import plotly.graph_objs as go
import plotly
import json
from shapely.geometry import Polygon
from itertools import combinations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        objetivo = request.form['objetivo']
        coef_obj = list(map(float, request.form['coef_obj'].split()))
        restricciones, rhs, operadores = [], [], []
        num_rest = int(request.form['num_rest'])
        for i in range(num_rest):
            coef = list(map(float, request.form[f'coef_rest_{i}'].split()))
            restricciones.append(coef)
            operadores.append(request.form[f'op_rest_{i}'])
            rhs.append(float(request.form[f'rhs_{i}']))

        resultado, variables, z, holguras, plot_data = resolver_programacion_lineal(
            coef_obj, restricciones, rhs, operadores, objetivo)
        return render_template('resultado.html', resultado=resultado,
                               variables=variables, z=z,
                               holguras=holguras, plot_url=plot_data)
    return render_template('index.html')


def resolver_programacion_lineal(coef_objetivo, restricciones, rhs, operadores, objetivo):
    # Definir modelo
    model = LpProblem("PL_Dinamico", LpMaximize if objetivo == 'max' else LpMinimize)
    num_vars = len(coef_objetivo)
    vars_pl = [LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)]
    model += lpSum(coef_objetivo[i] * vars_pl[i] for i in range(num_vars))
    for i, rest in enumerate(restricciones):
        expr = lpSum(rest[j] * vars_pl[j] for j in range(num_vars))
        if operadores[i] == '<=':
            model += expr <= rhs[i]
        elif operadores[i] == '>=':
            model += expr >= rhs[i]
        else:
            model += expr == rhs[i]
    model.solve()

    # Resultados PL
    resultado = LpStatus[model.status]
    var_result = [(v.name, v.varValue) for v in vars_pl]
    z = value(model.objective)
    holguras = [f"R{i+1}: {model.constraints[k].slack}" for i, k in enumerate(model.constraints)]

    # Gráfico Plotly
    plot_data = None
    if num_vars == 2:
        # Calcular vértices de región factible
        pts = []
        # Intersecciones restricciones
        for (i1,r1),(i2,r2) in combinations(enumerate(restricciones),2):
            a1,b1 = r1; a2,b2 = r2
            det = a1*b2 - a2*b1
            if abs(det) < 1e-6: continue
            c1, c2 = rhs[i1], rhs[i2]
            x = (c1*b2 - c2*b1)/det
            y = (a1*c2 - a2*c1)/det
            if x>=0 and y>=0 and all((rest[0]*x+rest[1]*y <= rhs[j]+1e-6 if op=='<=' else rest[0]*x+rest[1]*y >= rhs[j]-1e-6)
                                      for j,(rest,op) in enumerate(zip(restricciones,operadores))):
                pts.append((x,y))
        # Intersecciones con ejes
        for i,rest in enumerate(restricciones):
            a,b = rest
            if b!=0:
                y = rhs[i]/b
                if y>=0 and all((rr[0]*0+rr[1]*y <= rhs[j]+1e-6 if op=='<=' else rr[0]*0+rr[1]*y >= rhs[j]-1e-6)
                                 for j,(rr,op) in enumerate(zip(restricciones,operadores))):
                    pts.append((0,y))
            if a!=0:
                x = rhs[i]/a
                if x>=0 and all((rr[0]*x+rr[1]*0 <= rhs[j]+1e-6 if op=='<=' else rr[0]*x+rr[1]*0 >= rhs[j]-1e-6)
                                 for j,(rr,op) in enumerate(zip(restricciones,operadores))):
                    pts.append((x,0))
        if pts:
            poly = Polygon(pts).convex_hull
            x_poly,y_poly = poly.exterior.xy
        # Punto óptimo
        x_opt,y_opt = var_result[0][1] or 0, var_result[1][1] or 0
        # Crear figura y trazar región primero
        fig = go.Figure()
        if pts:
            fig.add_trace(go.Scatter(
                x=list(x_poly), y=list(y_poly), fill='toself',
                fillcolor='rgba(173,216,230,0.3)', line=dict(color='rgba(173,216,230,0)'),
                name='Región factible', showlegend=True
            ))
        # Trazar restricciones encima
        colors = ['red','green','blue','orange','purple']
        x_range = [0, max(x for x,_ in pts+[ (x_opt,y_opt) ])+5]
        y_range = [0, max(y for _,y in pts+[ (x_opt,y_opt) ])+5]
        x_vals = list(range(int(x_range[0]), int(x_range[1])+1))
        for i,rest in enumerate(restricciones):
            a,b = rest
            y_vals = [(rhs[i] - a*xi)/b if b!=0 else None for xi in x_vals]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines', name=f'R{i+1}',
                line=dict(color=colors[i%len(colors)], width=3)
            ))
        # Punto óptimo encima
        fig.add_trace(go.Scatter(
            x=[x_opt], y=[y_opt], mode='markers+text',
            marker=dict(color='black', size=12),
            text=[f'({"{:.2f}".format(x_opt)},{ "{:.2f}".format(y_opt)})'], textposition='top center',
            name='Óptimo'
        ))
        fig.update_layout(
            title='PL: Restricciones y Región Factible',
            xaxis_title='x1', yaxis_title='x2',
            xaxis=dict(range=x_range), yaxis=dict(range=y_range),
            legend=dict(x=0.01, y=0.99), width=700, height=500
        )
        plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return resultado, var_result, z, holguras, plot_data

if __name__=='__main__':
    app.run(debug=True)

