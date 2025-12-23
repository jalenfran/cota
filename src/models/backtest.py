"""
Backtesting framework for validating optimization proposals
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Results from backtesting an intervention"""
    baseline_metric: float
    treatment_metric: float
    effect_size: float
    effect_pct: float
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    
    def __str__(self) -> str:
        return (f"Effect: {self.effect_pct:+.1f}% "
                f"(95% CI: [{self.confidence_interval[0]:.1f}%, {self.confidence_interval[1]:.1f}%], "
                f"p={self.p_value:.3f})")


class Backtester:
    """Framework for backtesting transit system changes"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Args:
            historical_data: DataFrame with date, route, metric columns
        """
        self.data = historical_data.copy()
        self.data['date'] = pd.to_datetime(self.data.get('date', self.data.index))
        
    def time_series_split(self, train_pct: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally for backtesting"""
        cutoff = self.data['date'].quantile(train_pct)
        train = self.data[self.data['date'] <= cutoff]
        test = self.data[self.data['date'] > cutoff]
        return train, test
    
    def validate_demand_forecast(
        self, 
        model: Callable, 
        feature_cols: list,
        target_col: str,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Time-series cross-validation for demand forecasting
        
        Returns:
            metrics: MAE, RMSE, MAPE
        """
        results = []
        
        # Sort by date
        data = self.data.sort_values('date')
        n = len(data)
        test_size = n // (n_splits + 1)
        
        for i in range(n_splits):
            train_end = n - (n_splits - i) * test_size
            test_end = train_end + test_size
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Train model
            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values
            X_test = test_data[feature_cols].values
            y_test = test_data[target_col].values
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = np.abs(y_test - y_pred).mean()
            rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
            mape = (np.abs((y_test - y_pred) / y_test) * 100).mean()
            
            results.append({'mae': mae, 'rmse': rmse, 'mape': mape})
        
        # Average across folds
        return {
            'mae': np.mean([r['mae'] for r in results]),
            'rmse': np.mean([r['rmse'] for r in results]),
            'mape': np.mean([r['mape'] for r in results]),
        }
    
    def test_schedule_change(
        self,
        route_id: str,
        change_date: pd.Timestamp,
        metric_col: str,
        window_days: int = 30
    ) -> BacktestResult:
        """
        Analyze impact of historical schedule change using before/after comparison
        
        Args:
            route_id: Route identifier
            change_date: Date of schedule change
            metric_col: Metric to evaluate (e.g., 'ridership', 'delay')
            window_days: Days before/after to compare
        """
        route_data = self.data[self.data['route_id'] == route_id].copy()
        
        # Before period
        before_start = change_date - pd.Timedelta(days=window_days)
        before_data = route_data[
            (route_data['date'] >= before_start) & 
            (route_data['date'] < change_date)
        ]
        
        # After period
        after_end = change_date + pd.Timedelta(days=window_days)
        after_data = route_data[
            (route_data['date'] >= change_date) & 
            (route_data['date'] < after_end)
        ]
        
        # Calculate effect
        baseline = before_data[metric_col].mean()
        treatment = after_data[metric_col].mean()
        effect = treatment - baseline
        effect_pct = (effect / baseline) * 100 if baseline > 0 else 0
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        effects = []
        for _ in range(n_bootstrap):
            before_sample = before_data[metric_col].sample(len(before_data), replace=True).mean()
            after_sample = after_data[metric_col].sample(len(after_data), replace=True).mean()
            effects.append(((after_sample - before_sample) / before_sample) * 100)
        
        ci_lower, ci_upper = np.percentile(effects, [2.5, 97.5])
        
        # Two-sample t-test (simplified)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(after_data[metric_col], before_data[metric_col])
        
        return BacktestResult(
            baseline_metric=baseline,
            treatment_metric=treatment,
            effect_size=effect,
            effect_pct=effect_pct,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            sample_size=len(before_data) + len(after_data)
        )
    
    def simulate_new_stop(
        self,
        route_id: str,
        stop_location: Tuple[float, float],
        population_nearby: int,
        similar_stops_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Estimate ridership at proposed new stop using similar stops
        
        Args:
            route_id: Route for new stop
            stop_location: (lat, lon) of proposed stop
            population_nearby: Population within 400m
            similar_stops_data: Historical data from comparable stops
            
        Returns:
            estimates: predicted daily boardings with confidence bounds
        """
        # Use similar stops as comparables
        ridership_per_capita = similar_stops_data['daily_boardings'] / similar_stops_data['population_nearby']
        
        # Estimate for new stop
        mean_rate = ridership_per_capita.mean()
        std_rate = ridership_per_capita.std()
        
        predicted_boardings = mean_rate * population_nearby
        lower_bound = (mean_rate - 1.96 * std_rate) * population_nearby
        upper_bound = (mean_rate + 1.96 * std_rate) * population_nearby
        
        return {
            'predicted_daily_boardings': predicted_boardings,
            'lower_95ci': max(0, lower_bound),
            'upper_95ci': upper_bound,
            'comparable_stops': len(similar_stops_data)
        }
    
    def rolling_forecast_validation(
        self,
        model: Callable,
        features: pd.DataFrame,
        targets: pd.Series,
        initial_train_size: int,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Walk-forward validation for time series forecasting
        
        Returns:
            DataFrame with predictions, actuals, errors
        """
        results = []
        
        for i in range(initial_train_size, len(features) - horizon):
            # Expanding window
            X_train = features.iloc[:i]
            y_train = targets.iloc[:i]
            X_test = features.iloc[i:i+horizon]
            y_test = targets.iloc[i:i+horizon]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results.append({
                'date': features.index[i],
                'actual': y_test.iloc[0],
                'predicted': y_pred[0],
                'error': y_test.iloc[0] - y_pred[0]
            })
        
        return pd.DataFrame(results)


class ImpactEstimator:
    """Estimate impact of proposed changes"""
    
    @staticmethod
    def headway_ridership_elasticity(
        current_headway_min: int,
        proposed_headway_min: int,
        current_ridership: float,
        elasticity: float = -0.4
    ) -> Dict[str, float]:
        """
        Estimate ridership change from headway adjustment
        
        Typical elasticity: -0.4 (10% reduction in wait time -> 4% ridership increase)
        """
        wait_time_change_pct = (proposed_headway_min - current_headway_min) / current_headway_min
        ridership_change_pct = elasticity * wait_time_change_pct
        new_ridership = current_ridership * (1 + ridership_change_pct)
        
        return {
            'current_ridership': current_ridership,
            'projected_ridership': new_ridership,
            'change_pct': ridership_change_pct * 100,
            'additional_riders': new_ridership - current_ridership
        }
    
    @staticmethod
    def cost_benefit_analysis(
        annual_riders_added: float,
        fare_per_ride: float,
        annual_operating_cost: float,
        capital_cost: float = 0,
        discount_rate: float = 0.05,
        horizon_years: int = 10
    ) -> Dict[str, float]:
        """Calculate ROI for service improvements"""
        annual_revenue = annual_riders_added * fare_per_ride
        annual_net_benefit = annual_revenue - annual_operating_cost
        
        # NPV calculation
        npv = -capital_cost
        for year in range(1, horizon_years + 1):
            npv += annual_net_benefit / ((1 + discount_rate) ** year)
        
        # Payback period
        if annual_net_benefit > 0:
            payback_years = capital_cost / annual_net_benefit
        else:
            payback_years = float('inf')
        
        return {
            'annual_revenue': annual_revenue,
            'annual_operating_cost': annual_operating_cost,
            'annual_net_benefit': annual_net_benefit,
            'npv_10yr': npv,
            'roi_pct': (npv / capital_cost * 100) if capital_cost > 0 else float('inf'),
            'payback_years': payback_years
        }

