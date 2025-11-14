import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Asset Risk Dashboard"
server = app.server

def logistic_curve(x, alpha, midpoint=0.5):
    """S-curve for modeling gradual then rapid changes"""
    return 1 / (1 + np.exp(-alpha * (x - midpoint)))

def calculate_lof(age_normalized, lof_alpha, min_lof=0.05):
    """
    Calculate Likelihood of Failure based on age
    - Starts at min_lof (infant mortality)
    - Increases as S-curve (bathtub curve concept)
    """
    return min_lof + (1 - min_lof) * logistic_curve(age_normalized, lof_alpha, midpoint=0.7)

def generate_cost_curves(lifespan, cost_curve_alpha, replacement_cost, cycle_length):
    """
    Generate cumulative cost curves for each strategy
    Cost = repairs + interventions over time
    """
    x = np.linspace(0, lifespan, 500)
    x_norm = x / lifespan
    
    # 1. NO FIX: Zero cost until catastrophic failure at end
    # Small maintenance but then huge replacement at end
    no_fix_cost = replacement_cost * 0.05 * x_norm  # minimal upkeep
    # Add catastrophic replacement at ~90% of lifespan
    failure_point = int(0.9 * len(x))
    no_fix_cost[failure_point:] += replacement_cost
    
    # 2. FIX IN PLAN (Proactive): Regular scheduled maintenance
    # Smaller interventions at fixed intervals
    fix_in_plan_cost = np.zeros_like(x)
    n_cycles = int(lifespan / cycle_length)
    cost_per_cycle = replacement_cost * 0.15  # each intervention costs 15% of replacement
    
    for i in range(n_cycles):
        cycle_time = (i + 1) * cycle_length
        if cycle_time < lifespan:
            idx = int((cycle_time / lifespan) * len(x))
            # Add maintenance cost at each cycle
            fix_in_plan_cost[idx:] += cost_per_cycle
    
    # Add gradual baseline maintenance
    fix_in_plan_cost += replacement_cost * 0.02 * x_norm
    
    # 3. FIX ON FAIL (Reactive): Emergency repairs when things break
    # Random failures with expensive emergency repairs
    np.random.seed(42)
    fix_on_fail_cost = replacement_cost * 0.01 * x_norm  # minimal preventive
    
    # Fewer, more dramatic failures (same pattern as risk curve)
    n_failures = max(3, int(2 + 3 * (x_norm[-1] ** 2)))
    
    # Failures more likely in second half of life
    failure_times = []
    for i in range(n_failures):
        fail_position = 0.3 + (i / n_failures) * 0.7 + np.random.uniform(-0.05, 0.05)
        fail_position = max(0.3, min(1.0, fail_position))
        fail_idx = int(fail_position * len(x))
        failure_times.append(fail_idx)
    
    failure_times = sorted(set(failure_times))
    
    for fail_idx in failure_times:
        if fail_idx < len(fix_on_fail_cost):
            # Emergency repair costs 30-50% of replacement (expensive!)
            repair_cost = replacement_cost * np.random.uniform(0.3, 0.5)
            fix_on_fail_cost[fail_idx:] += repair_cost
    
    # 4. FIX ON RISK (Optimized): Intervene based on risk threshold
    # Monitor and intervene before failure but not on fixed schedule
    fix_on_risk_cost = replacement_cost * 0.02 * x_norm  # baseline monitoring
    
    # Interventions when risk crosses threshold (we'll calculate this properly)
    # Estimate ~4 interventions over lifespan, each 20% of replacement cost
    n_risk_interventions = 4
    for i in range(n_risk_interventions):
        intervention_point = int((0.3 + i * 0.15) * len(x))  # spread across life
        if intervention_point < len(x):
            intervention_cost = replacement_cost * 0.20
            fix_on_risk_cost[intervention_point:] += intervention_cost
    
    return x, no_fix_cost, fix_in_plan_cost, fix_on_fail_cost, fix_on_risk_cost

def generate_risk_curves(lifespan, risk_alpha, min_lof, cof, cycle_length, threshold):
    """
    Generate risk curves: Risk = LOF × COF
    Each strategy affects how LOF changes over time
    """
    x = np.linspace(0, lifespan, 500)
    x_norm = x / lifespan
    
    # 1. NO FIX: LOF continuously increases, never resets
    lof_no_fix = calculate_lof(x_norm, risk_alpha, min_lof)
    risk_no_fix = lof_no_fix * cof
    
    # 2. FIX IN PLAN (Cyclic): LOF follows smooth wave pattern with scheduled maintenance
    # Risk gradually builds then gradually decreases - creates smooth waves
    lof_cyclic = np.zeros_like(x)
    n_cycles = int(lifespan / cycle_length)
    
    for i in range(len(x)):
        # Which cycle are we in?
        cycle_position = (x[i] % cycle_length) / cycle_length  # 0 to 1 within each cycle
        
        # Create smooth wave using cosine: starts low, peaks at middle, returns low
        # (1 - cos(2π × position)) / 2 creates a smooth 0→1→0 wave
        wave = (1 - np.cos(2 * np.pi * cycle_position)) / 2
        
        # Scale wave from min_lof to a peak value
        # Peak LOF is moderate since we're doing regular maintenance
        peak_lof = min_lof + (0.3 - min_lof) * (1 + 0.5 * x_norm[i])  # Peaks grow slightly with age
        lof_cyclic[i] = min_lof + (peak_lof - min_lof) * wave
    
    risk_cyclic = lof_cyclic * cof
    
    # 3. FIX ON FAIL: Risk stays low until sudden failure spike
    # Then emergency repair resets it
    lof_fail = np.ones_like(x) * min_lof * 2  # slightly elevated baseline
    
    # Fewer, more dramatic failures that get more frequent with age
    # Early life: maybe 1-2 failures, Late life: 3-4 failures
    np.random.seed(42)
    n_failures = max(3, int(2 + 3 * (x_norm[-1] ** 2)))  # quadratic increase with age
    
    # Failures more likely in second half of life
    failure_times = []
    for i in range(n_failures):
        # Bias failures toward later in life
        fail_position = 0.3 + (i / n_failures) * 0.7 + np.random.uniform(-0.05, 0.05)
        fail_position = max(0.3, min(1.0, fail_position))
        fail_idx = int(fail_position * len(x))
        failure_times.append(fail_idx)
    
    failure_times = sorted(set(failure_times))  # remove duplicates and sort
    
    # Build risk profile with failures
    last_failure = 0
    for fail_idx in failure_times:
        if fail_idx >= len(lof_fail):
            continue
            
        # Risk builds up gradually between failures
        buildup_range = range(last_failure, fail_idx)
        for j in buildup_range:
            if j >= len(lof_fail):
                break
            # Gradual buildup with S-curve
            progress = (j - last_failure) / max(1, (fail_idx - last_failure))
            # LOF increases from baseline to near 1.0
            lof_fail[j] = min_lof * 2 + (0.9 - min_lof * 2) * logistic_curve(np.array([progress]), risk_alpha * 2, 0.6)[0]
        
        # Catastrophic failure spike
        lof_fail[fail_idx] = 1.0
        
        # After emergency repair, reset to baseline (but add a few points for visibility)
        reset_length = min(5, len(lof_fail) - fail_idx - 1)
        for k in range(1, reset_length + 1):
            if fail_idx + k < len(lof_fail):
                lof_fail[fail_idx + k] = min_lof * 2
        
        last_failure = fail_idx + reset_length
    
    # Fill in any remaining time after last failure
    if last_failure < len(lof_fail):
        for j in range(last_failure, len(lof_fail)):
            progress = (j - last_failure) / max(1, (len(lof_fail) - last_failure))
            lof_fail[j] = min_lof * 2 + (0.9 - min_lof * 2) * logistic_curve(np.array([progress]), risk_alpha * 2, 0.6)[0]
    
    risk_fail = lof_fail * cof
    
    # 4. FIX ON RISK: LOF increases until threshold, then intervention resets it
    lof_risk = np.zeros_like(x)
    current_lof = min_lof
    # Threshold is 0-1 representing what fraction of max risk we tolerate
    # Max risk ≈ 1.0 * cof (if LOF reaches 100%)
    # So threshold of 0.4 means we intervene at 40% of max risk = 0.4 * cof
    threshold_risk_dollars = threshold * cof
    threshold_lof = threshold  # Since risk = LOF × COF, and threshold is fraction of COF
    
    # Calculate how fast LOF should grow (faster as asset ages)
    for i in range(len(x)):
        # Base growth rate increases with age
        age_normalized = x_norm[i]
        # Growth rate: starts slow, accelerates with age
        growth_rate = 0.001 * risk_alpha * (1 + 2 * age_normalized)
        
        # LOF increases linearly until threshold
        current_lof += growth_rate
        
        # If we cross threshold, intervene and reset
        if current_lof >= threshold_lof:
            current_lof = min_lof * 1.5  # reset to slightly above baseline
        
        # Store the LOF value
        lof_risk[i] = current_lof
    
    risk_risk = lof_risk * cof
    
    # Baseline acceptable risk (around 10-15% of max risk)
    baseline = np.ones_like(x) * (min_lof * cof * 5)
    
    return x, risk_no_fix, risk_cyclic, risk_fail, risk_risk, baseline

def calculate_total_costs(x, no_fix, fix_in_plan, fix_on_fail, fix_on_risk):
    """Calculate total cost over lifespan for each strategy"""
    return {
        "No Fix": no_fix[-1],
        "Fix in Plan": fix_in_plan[-1],
        "Fix on Fail": fix_on_fail[-1],
        "Fix on Risk": fix_on_risk[-1]
    }

def calculate_average_risk(x, risk_curves):
    """Calculate time-weighted average risk for each strategy"""
    return {name: np.mean(curve) for name, curve in risk_curves.items()}

app.layout = html.Div([
    html.H2("Asset Risk Management Dashboard", style={"textAlign": "center", "marginBottom": "30px"}),

    dcc.Tabs(id="tabs", value="tab-risk", children=[
        dcc.Tab(label="📊 Risk Over Time", value="tab-risk"),
        dcc.Tab(label="💰 Cumulative Cost", value="tab-cost"),
        dcc.Tab(label="⚖️ Cost vs Risk Trade-off", value="tab-tradeoff"),
        dcc.Tab(label="📈 Summary Dashboard", value="tab-summary"),
    ]),

    html.Div(id="controls", children=[
        html.Div([
            html.Div([
                html.Label("Lifespan (years):", style={"fontWeight": "bold"}),
                dcc.Slider(id="lifespan-slider", min=10, max=200, step=10, value=100,
                          marks={i: str(i) for i in range(10, 201, 30)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("Replacement Cost ($):", style={"fontWeight": "bold"}),
                dcc.Slider(id="replacement-cost-slider", min=100e6, max=1e9, step=50e6, value=500e6,
                          marks={int(i): f"${int(i/1e6)}M" for i in range(100_000_000, 1_000_000_001, 300_000_000)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("Risk Alpha (failure rate growth):", style={"fontWeight": "bold"}),
                dcc.Slider(id="risk-alpha-slider", min=1, max=10, step=0.5, value=5.0,
                          marks={i: str(i) for i in range(1, 11, 2)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),
        ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "40px"}),

        html.Div([
            html.Div([
                html.Label("Minimum LOF (baseline failure rate):", style={"fontWeight": "bold"}),
                dcc.Slider(id="min-lof-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                          marks={i/100: f"{i}%" for i in range(1, 21, 5)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("Cycle Length (years between scheduled maintenance):", style={"fontWeight": "bold"}),
                dcc.Slider(id="cycle-length-slider", min=5, max=50, step=5, value=20,
                          marks={i: str(i) for i in range(5, 55, 10)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("Risk Threshold (% of max risk - trigger for Fix on Risk intervention):", style={"fontWeight": "bold"}),
                dcc.Slider(id="threshold-slider", min=0.1, max=0.9, step=0.05, value=0.4,
                          marks={i/10: f"{int(i*10)}%" for i in range(1, 10, 2)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "20px"}),
        ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"width": "90%", "margin": "auto", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"}),

    html.Div(id="tabs-content", style={"padding": "20px"})
])

@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("lifespan-slider", "value"),
    Input("replacement-cost-slider", "value"),
    Input("risk-alpha-slider", "value"),
    Input("min-lof-slider", "value"),
    Input("cycle-length-slider", "value"),
    Input("threshold-slider", "value"),
)
def render_tab(tab, lifespan, replacement_cost, risk_alpha, min_lof, cycle_length, threshold):
    # Calculate COF as replacement cost (consequence of complete failure)
    cof = replacement_cost
    cost_curve_alpha = risk_alpha * 0.8  # Cost curve follows similar pattern to risk
    
    # Generate curves
    x, no_fix_cost, fix_in_plan_cost, fix_on_fail_cost, fix_on_risk_cost = \
        generate_cost_curves(lifespan, cost_curve_alpha, replacement_cost, cycle_length)
    
    x_r, risk_no_fix, risk_cyclic, risk_fail, risk_risk, baseline = \
        generate_risk_curves(lifespan, risk_alpha, min_lof, cof, cycle_length, threshold)

    if tab == "tab-risk":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_r, y=risk_no_fix, 
            name="No Fix", 
            line=dict(color="red", width=3),
            hovertemplate="Year: %{x:.0f}<br>Risk: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x_r, y=risk_cyclic, 
            name="Fix in Plan (Scheduled)", 
            line=dict(color="cyan", width=2.5),
            hovertemplate="Year: %{x:.0f}<br>Risk: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x_r, y=risk_fail, 
            name="Fix on Fail (Reactive)", 
            line=dict(color="orange", width=2.5),
            hovertemplate="Year: %{x:.0f}<br>Risk: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x_r, y=risk_risk, 
            name="Fix on Risk (Optimized)", 
            line=dict(color="blue", width=3),
            hovertemplate="Year: %{x:.0f}<br>Risk: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x_r, y=baseline, 
            name="Baseline Risk", 
            line=dict(color="gray", width=2, dash="dot"),
            hovertemplate="Year: %{x:.0f}<br>Risk: $%{y:.2s}<extra></extra>"
        ))
        
        # Calculate the actual risk threshold value in dollars
        threshold_risk_dollars = threshold * cof
        
        fig.add_hline(
            y=threshold_risk_dollars, 
            line_dash="dash", 
            line_color="black", 
            annotation_text=f"Risk Threshold = ${threshold_risk_dollars/1e6:.0f}M ({threshold*100:.0f}%)",
            annotation_position="right"
        )
        
        fig.update_layout(
            title={
                "text": "Risk Over Time by Maintenance Strategy<br><sub>Risk = Likelihood of Failure × Consequence of Failure</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Time (years)", 
            yaxis_title="Risk (Expected Loss $)",
            yaxis_tickprefix="$", 
            yaxis_tickformat=".2s",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
        )
        
        return dcc.Graph(figure=fig)

    elif tab == "tab-cost":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=no_fix_cost, 
            name="No Fix", 
            line=dict(color="red", width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)',
            hovertemplate="Year: %{x:.0f}<br>Total Cost: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=fix_in_plan_cost, 
            name="Fix in Plan (Scheduled)", 
            line=dict(color="cyan", width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0,255,255,0.1)',
            hovertemplate="Year: %{x:.0f}<br>Total Cost: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=fix_on_fail_cost, 
            name="Fix on Fail (Reactive)", 
            line=dict(color="orange", width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.1)',
            hovertemplate="Year: %{x:.0f}<br>Total Cost: $%{y:.2s}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=fix_on_risk_cost, 
            name="Fix on Risk (Optimized)", 
            line=dict(color="blue", width=3),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.1)',
            hovertemplate="Year: %{x:.0f}<br>Total Cost: $%{y:.2s}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                "text": "Cumulative Cost Over Time by Maintenance Strategy",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Time (years)", 
            yaxis_title="Cumulative Cost ($)",
            yaxis_tickprefix="$", 
            yaxis_tickformat=".2s",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
        )
        
        return dcc.Graph(figure=fig)

    elif tab == "tab-tradeoff":
        # Calculate metrics
        total_costs = calculate_total_costs(x, no_fix_cost, fix_in_plan_cost, fix_on_fail_cost, fix_on_risk_cost)
        avg_risks = calculate_average_risk(x_r, {
            "No Fix": risk_no_fix,
            "Fix in Plan": risk_cyclic,
            "Fix on Fail": risk_fail,
            "Fix on Risk": risk_risk
        })
        
        strategies = list(total_costs.keys())
        costs = [total_costs[s] for s in strategies]
        risks = [avg_risks[s] for s in strategies]
        colors = ["red", "cyan", "orange", "blue"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=costs,
            y=risks,
            mode='markers+text',
            marker=dict(size=[30, 25, 25, 30], color=colors, opacity=0.7),
            text=strategies,
            textposition="top center",
            textfont=dict(size=12, color="black"),
            hovertemplate="<b>%{text}</b><br>Total Cost: $%{x:.2s}<br>Avg Risk: $%{y:.2s}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                "text": "Cost vs Risk Trade-off Analysis<br><sub>Lower-left is better (low cost, low risk)</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Total Cost Over Lifespan ($)",
            yaxis_title="Average Risk (Expected Loss $)",
            xaxis_tickprefix="$",
            xaxis_tickformat=".2s",
            yaxis_tickprefix="$",
            yaxis_tickformat=".2s",
            template="plotly_white",
            height=600
        )
        
        # Add quadrant lines
        median_cost = np.median(costs)
        median_risk = np.median(risks)
        
        fig.add_vline(x=median_cost, line_dash="dash", line_color="lightgray", opacity=0.5)
        fig.add_hline(y=median_risk, line_dash="dash", line_color="lightgray", opacity=0.5)
        
        return dcc.Graph(figure=fig)

    elif tab == "tab-summary":
        # Create a comprehensive summary dashboard
        total_costs = calculate_total_costs(x, no_fix_cost, fix_in_plan_cost, fix_on_fail_cost, fix_on_risk_cost)
        avg_risks = calculate_average_risk(x_r, {
            "No Fix": risk_no_fix,
            "Fix in Plan": risk_cyclic,
            "Fix on Fail": risk_fail,
            "Fix on Risk": risk_risk
        })
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Total Cost by Strategy", "Average Risk by Strategy", 
                           "Risk Over Time (Detail)", "Cost Over Time (Detail)"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Bar chart: Total costs
        strategies = list(total_costs.keys())
        costs = [total_costs[s] for s in strategies]
        colors_list = ["red", "cyan", "orange", "blue"]
        
        fig.add_trace(
            go.Bar(x=strategies, y=costs, marker_color=colors_list, name="Total Cost",
                   text=[f"${c/1e6:.0f}M" for c in costs], textposition="outside"),
            row=1, col=1
        )
        
        # Bar chart: Average risks
        risks = [avg_risks[s] for s in strategies]
        fig.add_trace(
            go.Bar(x=strategies, y=risks, marker_color=colors_list, name="Avg Risk",
                   text=[f"${r/1e6:.0f}M" for r in risks], textposition="outside"),
            row=1, col=2
        )
        
        # Line chart: Risk detail (last 30 years)
        idx_start = max(0, len(x_r) - 150)
        fig.add_trace(go.Scatter(x=x_r[idx_start:], y=risk_no_fix[idx_start:], name="No Fix", line=dict(color="red")), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_r[idx_start:], y=risk_cyclic[idx_start:], name="Fix in Plan", line=dict(color="cyan")), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_r[idx_start:], y=risk_fail[idx_start:], name="Fix on Fail", line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_r[idx_start:], y=risk_risk[idx_start:], name="Fix on Risk", line=dict(color="blue")), row=2, col=1)
        
        # Line chart: Cost detail (last 30 years)
        idx_start_cost = max(0, len(x) - 150)
        fig.add_trace(go.Scatter(x=x[idx_start_cost:], y=no_fix_cost[idx_start_cost:], name="No Fix", line=dict(color="red"), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x[idx_start_cost:], y=fix_in_plan_cost[idx_start_cost:], name="Fix in Plan", line=dict(color="cyan"), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x[idx_start_cost:], y=fix_on_fail_cost[idx_start_cost:], name="Fix on Fail", line=dict(color="orange"), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x[idx_start_cost:], y=fix_on_risk_cost[idx_start_cost:], name="Fix on Risk", line=dict(color="blue"), showlegend=False), row=2, col=2)
        
        fig.update_xaxes(title_text="Strategy", row=1, col=1)
        fig.update_yaxes(title_text="Total Cost ($)", tickprefix="$", tickformat=".2s", row=1, col=1)
        
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Avg Risk ($)", tickprefix="$", tickformat=".2s", row=1, col=2)
        
        fig.update_xaxes(title_text="Time (years)", row=2, col=1)
        fig.update_yaxes(title_text="Risk ($)", tickprefix="$", tickformat=".2s", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (years)", row=2, col=2)
        fig.update_yaxes(title_text="Cost ($)", tickprefix="$", tickformat=".2s", row=2, col=2)
        
        fig.update_layout(
            title_text="Comprehensive Maintenance Strategy Summary",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.H4("Key Insights:", style={"marginTop": "20px"}),
                html.Ul([
                    html.Li(f"Lowest Total Cost: {min(total_costs, key=total_costs.get)} (${total_costs[min(total_costs, key=total_costs.get)]/1e6:.0f}M)"),
                    html.Li(f"Lowest Average Risk: {min(avg_risks, key=avg_risks.get)} (${avg_risks[min(avg_risks, key=avg_risks.get)]/1e6:.0f}M)"),
                    html.Li(f"Best Balance (Fix on Risk): Cost ${total_costs['Fix on Risk']/1e6:.0f}M, Risk ${avg_risks['Fix on Risk']/1e6:.0f}M"),
                ])
            ], style={"width": "80%", "margin": "auto", "padding": "20px", "backgroundColor": "#e9ecef", "borderRadius": "10px"})
        ])

if __name__ == "__main__":
    app.run(debug=True)