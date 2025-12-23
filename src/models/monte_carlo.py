"""
Monte Carlo simulation framework for uncertainty quantification in transit optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    mean: float
    std: float
    percentiles: Dict[float, float]
    confidence_interval: Tuple[float, float]
    n_simulations: int
    
    def __str__(self) -> str:
        return (f"Mean: {self.mean:.2f}, Std: {self.std:.2f}\n"
                f"95% CI: [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}]")


class MonteCarloSimulator:
    """Monte Carlo simulation for transit optimization under uncertainty"""
    
    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        """
        Args:
            n_simulations: Number of Monte Carlo iterations
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_ridership(
        self,
        base_ridership: float,
        uncertainty_pct: float = 0.2,
        distribution: str = 'normal'
    ) -> np.ndarray:
        """
        Simulate ridership with uncertainty.
        
        Args:
            base_ridership: Base ridership estimate
            uncertainty_pct: Coefficient of variation (e.g., 0.2 = 20% std)
            distribution: 'normal' or 'lognormal'
            
        Returns:
            Array of simulated ridership values
        """
        if distribution == 'normal':
            std = base_ridership * uncertainty_pct
            return np.random.normal(base_ridership, std, self.n_simulations)
        elif distribution == 'lognormal':
            mu = np.log(base_ridership) - 0.5 * (uncertainty_pct ** 2)
            sigma = uncertainty_pct
            return np.random.lognormal(mu, sigma, self.n_simulations)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def simulate_roi(
        self,
        annual_riders: Callable[[], float],
        fare_per_ride: float = 2.0,
        annual_cost: float = 0,
        capital_cost: float = 0
    ) -> MonteCarloResult:
        """
        Simulate ROI with uncertain ridership.
        
        Args:
            annual_riders: Function that returns simulated annual riders
            fare_per_ride: Fare per ride
            annual_cost: Annual operating cost
            capital_cost: One-time capital cost
            
        Returns:
            MonteCarloResult with ROI statistics
        """
        rois = []
        
        for _ in range(self.n_simulations):
            riders = annual_riders()
            revenue = riders * fare_per_ride
            net_benefit = revenue - annual_cost
            
            if capital_cost > 0:
                roi = ((net_benefit - capital_cost * 0.1) / capital_cost) * 100
            else:
                roi = (net_benefit / abs(annual_cost)) * 100 if annual_cost != 0 else 0
            
            rois.append(roi)
        
        rois = np.array(rois)
        
        return MonteCarloResult(
            mean=rois.mean(),
            std=rois.std(),
            percentiles={
                5: np.percentile(rois, 5),
                25: np.percentile(rois, 25),
                50: np.percentile(rois, 50),
                75: np.percentile(rois, 75),
                95: np.percentile(rois, 95)
            },
            confidence_interval=(np.percentile(rois, 2.5), np.percentile(rois, 97.5)),
            n_simulations=self.n_simulations
        )
    
    def sensitivity_analysis(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameters: Dict[str, Tuple[float, float]],
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Sensitivity analysis using Monte Carlo sampling.
        
        Args:
            objective_function: Function that takes parameter dict and returns objective value
            parameters: Dict of parameter_name -> (mean, std) tuples
            n_samples: Number of samples
            
        Returns:
            DataFrame with parameter correlations to objective
        """
        samples = []
        objectives = []
        
        for _ in range(n_samples):
            param_values = {}
            for param_name, (mean, std) in parameters.items():
                param_values[param_name] = np.random.normal(mean, std)
            
            samples.append(param_values)
            objectives.append(objective_function(param_values))
        
        df = pd.DataFrame(samples)
        df['objective'] = objectives
        
        correlations = df.corr()['objective'].drop('objective').abs().sort_values(ascending=False)
        
        return pd.DataFrame({
            'parameter': correlations.index,
            'correlation': correlations.values,
            'importance_rank': range(1, len(correlations) + 1)
        })
    
    def simulate_stop_proposal(
        self,
        population_nearby: int,
        ridership_rate_mean: float = 0.02,
        ridership_rate_std: float = 0.005,
        fare: float = 2.0,
        cost_per_stop: float = 10000,
        operating_cost_pct: float = 0.1
    ) -> MonteCarloResult:
        """
        Simulate ROI for a proposed new stop with uncertain ridership.
        
        Args:
            population_nearby: Population within 400m
            ridership_rate_mean: Mean ridership rate (riders/population)
            ridership_rate_std: Std of ridership rate
            fare: Fare per ride
            cost_per_stop: Implementation cost
            operating_cost_pct: Annual operating cost as % of capital
            
        Returns:
            MonteCarloResult with ROI statistics
        """
        def annual_riders():
            rate = np.random.normal(ridership_rate_mean, ridership_rate_std)
            rate = max(0, rate)
            daily_riders = population_nearby * rate
            return daily_riders * 260
        
        return self.simulate_roi(
            annual_riders=annual_riders,
            fare_per_ride=fare,
            annual_cost=cost_per_stop * operating_cost_pct,
            capital_cost=cost_per_stop
        )
    
    def simulate_schedule_change(
        self,
        current_headway_min: float,
        proposed_headway_min: float,
        current_ridership: float,
        elasticity_mean: float = -0.4,
        elasticity_std: float = 0.1,
        fare: float = 2.0,
        vehicle_cost_per_hour: float = 100,
        service_hours_per_day: float = 16,
        cycle_time_min: float = 60
    ) -> MonteCarloResult:
        """
        Simulate ROI for schedule change with uncertain elasticity.
        
        Args:
            current_headway_min: Current headway
            proposed_headway_min: Proposed headway
            current_ridership: Current daily ridership
            elasticity_mean: Mean ridership elasticity
            elasticity_std: Std of elasticity
            fare: Fare per ride
            vehicle_cost_per_hour: Operating cost per vehicle-hour
            service_hours_per_day: Service hours per day
            cycle_time_min: Route cycle time
            
        Returns:
            MonteCarloResult with net benefit statistics
        """
        def annual_benefit():
            elasticity = np.random.normal(elasticity_mean, elasticity_std)
            
            wait_time_change_pct = (proposed_headway_min - current_headway_min) / current_headway_min
            ridership_change_pct = elasticity * wait_time_change_pct
            new_ridership = current_ridership * (1 + ridership_change_pct)
            
            additional_riders = new_ridership - current_ridership
            
            vehicles_current = np.ceil(cycle_time_min / current_headway_min)
            vehicles_proposed = np.ceil(cycle_time_min / proposed_headway_min)
            additional_vehicles = vehicles_proposed - vehicles_current
            
            additional_hours = additional_vehicles * service_hours_per_day
            annual_cost = additional_hours * 260 * vehicle_cost_per_hour
            annual_revenue = additional_riders * 260 * fare
            
            return annual_revenue - annual_cost
        
        benefits = np.array([annual_benefit() for _ in range(self.n_simulations)])
        
        return MonteCarloResult(
            mean=benefits.mean(),
            std=benefits.std(),
            percentiles={
                5: np.percentile(benefits, 5),
                25: np.percentile(benefits, 25),
                50: np.percentile(benefits, 50),
                75: np.percentile(benefits, 75),
                95: np.percentile(benefits, 95)
            },
            confidence_interval=(np.percentile(benefits, 2.5), np.percentile(benefits, 97.5)),
            n_simulations=self.n_simulations
        )

