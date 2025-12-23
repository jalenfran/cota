"""
Professional visualizations for COTA executive presentations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# Professional color scheme
COTA_COLORS = {
    'primary': '#AF272F',    # COTA Red
    'secondary': '#00205B',  # COTA Blue
    'accent': '#007B4B',     # COTA Green
    'neutral': '#666666',
    'background': '#F5F5F5'
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([COTA_COLORS['primary'], COTA_COLORS['secondary'], COTA_COLORS['accent']])


def route_efficiency_dashboard(
    routes_df: pd.DataFrame,
    metrics: List[str] = ['directness', 'stops_per_km', 'service_span_hours']
) -> plt.Figure:
    """
    Create executive dashboard showing route efficiency metrics
    
    Args:
        routes_df: DataFrame with route metrics
        metrics: List of metric columns to display
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COTA Route Efficiency Analysis', fontsize=20, fontweight='bold', y=0.995)
    
    # 1. Route efficiency scatter
    ax = axes[0, 0]
    scatter = ax.scatter(
        routes_df['directness'],
        routes_df['stops_per_km'],
        s=routes_df['trips_per_day'] * 2,
        c=routes_df['service_span_hours'],
        cmap='RdYlGn_r',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Route Directness (lower = more direct)', fontsize=12)
    ax.set_ylabel('Stops per km', fontsize=12)
    ax.set_title('Route Efficiency Matrix', fontsize=14, fontweight='bold')
    ax.axhline(routes_df['stops_per_km'].median(), color='red', linestyle='--', 
               alpha=0.5, label='Median')
    ax.axvline(routes_df['directness'].median(), color='red', linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Service Span (hours)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Top/Bottom performers
    ax = axes[0, 1]
    sorted_routes = routes_df.sort_values('directness')
    top_5 = sorted_routes.head(5)
    bottom_5 = sorted_routes.tail(5)
    
    y_pos = np.arange(10)
    combined = pd.concat([top_5, bottom_5])
    colors = ['green'] * 5 + ['red'] * 5
    
    ax.barh(y_pos, combined['directness'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined['route_name'])
    ax.set_xlabel('Directness Score', fontsize=12)
    ax.set_title('Best & Worst Routes by Directness', fontsize=14, fontweight='bold')
    ax.axvline(routes_df['directness'].median(), color='black', linestyle='--', 
               alpha=0.5, label='System Median')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Service distribution
    ax = axes[1, 0]
    routes_df.groupby('route_type')['trips_per_day'].sum().plot(
        kind='pie',
        ax=ax,
        autopct='%1.1f%%',
        colors=[COTA_COLORS['primary'], COTA_COLORS['secondary'], COTA_COLORS['accent']],
        startangle=90
    )
    ax.set_title('Daily Trips by Route Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    
    # 4. Efficiency summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    SYSTEM PERFORMANCE SUMMARY
    ══════════════════════════════════
    
    Total Routes:              {len(routes_df)}
    Total Daily Trips:         {routes_df['trips_per_day'].sum():,.0f}
    
    EFFICIENCY METRICS
    ──────────────────────────────────
    Avg Route Directness:      {routes_df['directness'].mean():.2f}
    Median Stops/km:           {routes_df['stops_per_km'].median():.2f}
    Avg Service Span:          {routes_df['service_span_hours'].mean():.1f} hrs
    
    RECOMMENDATIONS
    ──────────────────────────────────
    Routes needing attention:  {len(routes_df[routes_df['directness'] > routes_df['directness'].quantile(0.75)])}
    Optimization potential:    High Priority
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor=COTA_COLORS['background'], alpha=0.8))
    
    plt.tight_layout()
    return fig


def coverage_heatmap(
    stops_df: pd.DataFrame,
    coverage_gaps: pd.DataFrame,
    proposed_stops: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize service coverage and gaps
    
    Args:
        stops_df: Existing stops with lat, lon
        coverage_gaps: Areas with poor coverage
        proposed_stops: Recommended new stop locations
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Coverage gaps (red = high priority)
    scatter1 = ax.scatter(
        coverage_gaps['lon'],
        coverage_gaps['lat'],
        s=coverage_gaps['population'],
        c=coverage_gaps['walk_time_minutes'],
        cmap='Reds',
        alpha=0.5,
        label='Coverage Gaps'
    )
    
    # Existing stops
    ax.scatter(
        stops_df['stop_lon'],
        stops_df['stop_lat'],
        s=20,
        c=COTA_COLORS['primary'],
        marker='s',
        alpha=0.7,
        label='Existing Stops',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Proposed new stops
    if proposed_stops is not None:
        ax.scatter(
            proposed_stops['lon'],
            proposed_stops['lat'],
            s=200,
            c=COTA_COLORS['accent'],
            marker='*',
            alpha=0.9,
            label='Proposed New Stops',
            edgecolors='black',
            linewidth=1
        )
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('COTA Service Coverage Analysis & Recommendations', 
                 fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Walk Time to Nearest Stop (minutes)', fontsize=10)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def backtest_results_plot(
    backtest_df: pd.DataFrame,
    metric_name: str = 'Ridership',
    intervention_name: str = 'Service Change'
) -> plt.Figure:
    """
    Visualize backtesting results with before/after comparison
    
    Args:
        backtest_df: DataFrame with date, actual, predicted, intervention columns
        metric_name: Name of metric being analyzed
        intervention_name: Name of intervention tested
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series with intervention marker
    ax = axes[0]
    intervention_date = backtest_df[backtest_df.get('intervention', False) == True]['date'].min()
    
    ax.plot(backtest_df['date'], backtest_df['actual'], 
            label='Actual', color=COTA_COLORS['primary'], linewidth=2)
    ax.plot(backtest_df['date'], backtest_df['predicted'], 
            label='Predicted (No Intervention)', color=COTA_COLORS['secondary'],
            linewidth=2, linestyle='--')
    
    if pd.notna(intervention_date):
        ax.axvline(intervention_date, color='green', linestyle=':', linewidth=2,
                   label=f'{intervention_name} Implementation')
        ax.fill_between(
            backtest_df[backtest_df['date'] >= intervention_date]['date'],
            backtest_df[backtest_df['date'] >= intervention_date]['actual'].min(),
            backtest_df[backtest_df['date'] >= intervention_date]['actual'].max(),
            alpha=0.1, color='green', label='Post-Intervention Period'
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Backtesting Results: {intervention_name} Impact on {metric_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Residual analysis
    ax = axes[1]
    residuals = backtest_df['actual'] - backtest_df['predicted']
    ax.scatter(backtest_df['date'], residuals, alpha=0.5, 
               c=COTA_COLORS['neutral'], s=30)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(residuals.std(), color='red', linestyle=':', alpha=0.5, 
               label=f'±1 Std Dev ({residuals.std():.1f})')
    ax.axhline(-residuals.std(), color='red', linestyle=':', alpha=0.5)
    
    if pd.notna(intervention_date):
        ax.axvline(intervention_date, color='green', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Model Residuals', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def roi_comparison_chart(proposals: List[dict]) -> plt.Figure:
    """
    Compare ROI of different optimization proposals
    
    Args:
        proposals: List of dicts with keys: name, cost, benefit, roi, confidence_interval
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    df = pd.DataFrame(proposals).sort_values('roi', ascending=False)
    
    # ROI bar chart
    ax = axes[0]
    colors = [COTA_COLORS['accent'] if roi > 0 else COTA_COLORS['primary'] 
              for roi in df['roi']]
    bars = ax.barh(df['name'], df['roi'], color=colors, alpha=0.7)
    
    # Add confidence intervals if available
    if 'confidence_interval' in df.columns:
        for i, (name, row) in enumerate(df.iterrows()):
            ci = row['confidence_interval']
            ax.plot([ci[0], ci[1]], [i, i], color='black', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('ROI (%)', fontsize=12)
    ax.set_title('Return on Investment by Proposal', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Cost vs Benefit scatter
    ax = axes[1]
    scatter = ax.scatter(
        df['cost'],
        df['benefit'],
        s=abs(df['roi']) * 10,
        c=df['roi'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Diagonal line (break-even)
    max_val = max(df['cost'].max(), df['benefit'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Break-even')
    
    for _, row in df.iterrows():
        ax.annotate(row['name'], (row['cost'], row['benefit']), 
                    fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Annual Cost ($)', fontsize=12)
    ax.set_ylabel('Annual Benefit ($)', fontsize=12)
    ax.set_title('Cost-Benefit Analysis', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ROI (%)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def executive_summary_page(
    key_findings: List[str],
    recommendations: List[str],
    metrics: Dict[str, any]
) -> plt.Figure:
    """
    Generate one-page executive summary visualization
    
    Args:
        key_findings: List of bullet points
        recommendations: List of recommended actions
        metrics: Dict of key performance indicators
    """
    fig = plt.figure(figsize=(11, 8.5))  # Letter size
    
    # Title
    fig.text(0.5, 0.95, 'COTA OPTIMIZATION ANALYSIS', 
             ha='center', fontsize=24, fontweight='bold',
             color=COTA_COLORS['primary'])
    fig.text(0.5, 0.92, 'Executive Summary', 
             ha='center', fontsize=16, color=COTA_COLORS['secondary'])
    
    # Key metrics boxes
    metrics_y = 0.82
    metrics_x = [0.15, 0.4, 0.65]
    for i, (key, value) in enumerate(list(metrics.items())[:3]):
        x = metrics_x[i]
        fig.add_artist(plt.Rectangle((x-0.08, metrics_y-0.04), 0.16, 0.08,
                                     facecolor=COTA_COLORS['background'],
                                     edgecolor=COTA_COLORS['primary'],
                                     linewidth=2, transform=fig.transFigure))
        fig.text(x, metrics_y+0.01, str(value), 
                ha='center', fontsize=20, fontweight='bold',
                color=COTA_COLORS['primary'])
        fig.text(x, metrics_y-0.025, key, 
                ha='center', fontsize=10, color=COTA_COLORS['neutral'])
    
    # Key findings
    findings_y = 0.68
    fig.text(0.1, findings_y, 'KEY FINDINGS', 
             fontsize=14, fontweight='bold', color=COTA_COLORS['secondary'])
    
    for i, finding in enumerate(key_findings[:5]):
        y = findings_y - 0.06 - (i * 0.05)
        fig.text(0.12, y, f'• {finding}', fontsize=10, 
                verticalalignment='top', wrap=True)
    
    # Recommendations
    rec_y = 0.38
    fig.text(0.1, rec_y, 'RECOMMENDATIONS', 
             fontsize=14, fontweight='bold', color=COTA_COLORS['secondary'])
    
    for i, rec in enumerate(recommendations[:5]):
        y = rec_y - 0.06 - (i * 0.05)
        fig.text(0.12, y, f'{i+1}. {rec}', fontsize=10, 
                verticalalignment='top', wrap=True,
                color=COTA_COLORS['accent'], fontweight='bold')
    
    # Footer
    fig.text(0.5, 0.05, 'Detailed analysis and methodology available in full report',
             ha='center', fontsize=8, style='italic', color=COTA_COLORS['neutral'])
    
    return fig

