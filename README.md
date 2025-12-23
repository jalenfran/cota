# COTA Bus System Optimization

**Professional quantitative analysis** of Columbus COTA bus system with **actionable, backtested recommendations** for transit officials. Generates executive-ready reports (PDF + Excel) with specific proposals for new stops, schedule adjustments, and budget scenarios.

## Data Sources

**Static GTFS** (Schedule data)
- 42 routes, 2,986 stops, 5,476 trips
- 364K stop_times, 100K shape points
- Feed valid: Sept 2025 - Jan 2026

**Realtime GTFS-RT** (Live feeds from COTA servers)
- VehiclePositions - GPS tracking (live)
- TripUpdates - Schedule-based delay computation (live)
- Alerts - Service disruptions (live)

**Census Data** (Population & demographics)
- ACS 5-year estimates (2022) - Franklin County block groups
- TIGER/Line shapefiles - Geographic boundaries
- Auto-downloaded and cached on first use

## Research Questions

1. **Route Efficiency**: Which routes underperform? Dead zones?
2. **Schedule Optimization**: Headway gaps, bunching patterns
3. **Demand Prediction**: Ridership forecasting by time/location
4. **Resource Allocation**: Budget impact scenarios
5. **Network Topology**: Connection optimality, transfer efficiency

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install CBC solver (required for MIP optimization)
# macOS:
brew install cbc

# Linux:
# sudo apt-get install coinor-cbc

# Generate professional reports
jupyter notebook notebooks/02_professional_recommendations.ipynb
```

**Output**: 
- `output/reports/executive_summary.pdf` - 2-page summary for decision-makers
- `output/reports/recommendations.xlsx` - Detailed implementation plan

## Data Collection

For continuous real-time data collection:

```bash
# Run locally
python scripts/simple_collector.py

# Or with custom intervals
python scripts/simple_collector.py 2 15  # 2 min snapshots, 15 min alerts
```

Data collected to `data/realtime_history/`:
- Vehicles & delays: Every 2 minutes (configurable)
- Alerts: Every 15 minutes (alerts change infrequently)

## Key Capabilities

### Analysis & Optimization
- Route efficiency scoring (directness, stop density, service span)
- Coverage gap identification with real census population data
- New stop placement optimization (integer programming)
- Schedule optimization with demand matching
- Cost-benefit analysis with ROI calculation

### Backtesting & Validation
- Time-series cross-validation for forecasts
- Before/after analysis for schedule changes
- Confidence intervals for all estimates
- Sensitivity analysis for robustness

### Professional Deliverables
- Executive summary PDF (auto-generated)
- Detailed recommendations Excel workbook
- Professional dashboards and visualizations
- Implementation guide for COTA officials

