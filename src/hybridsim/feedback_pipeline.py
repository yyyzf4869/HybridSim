import os
import json
import logging
import pandas as pd
import asyncio
from typing import Dict, Any, List
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackPipeline:
    """
    Feedback iteration pipeline
    Responsible for running simulation multiple times and updating feedback data after each round
    """
    
    def __init__(self, args):
        self.args = args
        self.scaling_factor = None
        self.feedback_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback', 'feedback_data.csv')
        self.simulated_behavior_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback', 'simulated_behavior_15min.json')
        # Backup original feedback data
        self._backup_feedback_data()
        self._backup_simulated_behavior_data()
        
    def _backup_feedback_data(self):
        """Backup original feedback data file"""
        if os.path.exists(self.feedback_data_path):
            backup_path = self.feedback_data_path + ".bak"
            shutil.copy2(self.feedback_data_path, backup_path)
            logger.info(f"Backed up original feedback data to: {backup_path}")

    def _backup_simulated_behavior_data(self):
        """Backup original simulated behavior data file"""
        if os.path.exists(self.simulated_behavior_path):
            backup_path = self.simulated_behavior_path + ".bak"
            shutil.copy2(self.simulated_behavior_path, backup_path)
            logger.info(f"Backed up original simulated behavior data to: {backup_path}")

    def _restore_feedback_data(self):
        """Restore original feedback data file (optional, if cleanup needed)"""
        backup_path = self.feedback_data_path + ".bak"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, self.feedback_data_path)
            logger.info(f"Restored original feedback data")
        
        behavior_backup_path = self.simulated_behavior_path + ".bak"
        if os.path.exists(behavior_backup_path):
            shutil.copy2(behavior_backup_path, self.simulated_behavior_path)
            logger.info(f"Restored original simulated behavior data")

    def normalize_data(self, consumption_data: List[float]) -> List[float]:
        """
        Normalize energy consumption data
        If it's the first round, calculate Scaling Factor (Max Value)
        Subsequent rounds use the same Factor
        """
        current_max = max(consumption_data) if consumption_data else 1.0
        
        if self.scaling_factor is None:
            self.scaling_factor = current_max
            logger.info(f"[Feedback Pipeline] Setting normalization factor (Scaling Factor): {self.scaling_factor:.4f}")
        else:
            logger.info(f"[Feedback Pipeline] Using existing normalization factor: {self.scaling_factor:.4f} (Current round max: {current_max:.4f})")
            
        # Avoid division by zero
        if self.scaling_factor == 0:
            return consumption_data
            
        return [x / self.scaling_factor for x in consumption_data]

    def update_feedback_csv(self, time_steps: List[str], normalized_consumption: List[float]):
        """
        Update feedback_data.csv
        Simplified logic: assume simulation runs only one day (within 24 hours), directly extract HH:MM for matching
        """
        try:
            # Read existing CSV
            df = pd.read_csv(self.feedback_data_path)
            
            # Create new data dictionary: extract HH:MM -> value
            new_data_map = {}
            for t, v in zip(time_steps, normalized_consumption):
                # Extract HH:MM
                # Format might be "Monday 09:00" or "09:00"
                if ' ' in t:
                    time_part = t.split(' ')[1]
                else:
                    time_part = t
                
                # Try to standardize to HH:MM
                try:
                    h, m = map(int, time_part.split(':'))
                    std_key = f"{h:02d}:{m:02d}"
                    new_data_map[std_key] = v
                except:
                    # Skip if format is incorrect
                    continue

            # Apply updates
            def get_new_val(row):
                t = row['Time']
                # Standardize CSV time
                try:
                    h, m = map(int, t.split(':'))
                    std_t = f"{h:02d}:{m:02d}"
                except:
                    std_t = t
                
                if std_t in new_data_map:
                    return new_data_map[std_t]
                return row['Normalized_LLM'] # Keep original value

            df['Normalized_LLM'] = df.apply(get_new_val, axis=1)
            
            # Save back to CSV
            df.to_csv(self.feedback_data_path, index=False)
            logger.info(f"Updated feedback_data.csv, wrote {len(new_data_map)} new data points (single-day mode)")
            
        except Exception as e:
            logger.error(f"Failed to update CSV: {e}")
            raise

    def update_simulated_behavior_json(self, behavior_stats: List[Dict[str, Any]]):
        """Update simulated behavior data JSON"""
        try:
            with open(self.simulated_behavior_path, 'w', encoding='utf-8') as f:
                json.dump(behavior_stats, f, indent=4, ensure_ascii=False)
            logger.info(f"Updated simulated behavior data: {self.simulated_behavior_path}")
        except Exception as e:
            logger.error(f"Failed to update simulated behavior data: {e}")

    async def run(self):
        """Run iteration pipeline"""
        from code.simulator import EnergyConsumptionSimulator
        
        iterations = getattr(self.args, 'feedback_iterations', 1)
        logger.info(f"Starting Feedback iteration pipeline, total rounds: {iterations}")
        
        for i in range(iterations):
            logger.info(f"\n{'='*20} Iteration Round {i+1}/{iterations} {'='*20}")
            
            # 1. Create simulator (needs to be recreated each time to reload CSV)
            simulator = EnergyConsumptionSimulator(
                llm_workers=self.args.llm_workers,
                dl_workers=self.args.dl_workers,
                architecture_type=self.args.architecture,
                dl_model_path=self.args.dl_model_path,
                confidence_threshold=self.args.confidence_threshold,
                time_step=self.args.simulation_step,
                enable_feedback=True, # Force enable Feedback
                environmental_condition=self.args.environmental_condition
            )
            
            # 2. Initialize and run
            await simulator.initialize_simulation(
                start_time=self.args.start_time,
                max_steps=self.args.max_steps,
                agent_count=self.args.agent_count
            )
            
            await simulator.run_simulation()
            
            # 3. Get results
            # chart_data.json has already been exported in run_simulation, but we need in-memory data
            # simulator.environment_agent.step_consumptions contains raw data
            # But we need the total like in chart_data.json
            
            step_keys = sorted(simulator.environment_agent.step_consumptions.keys())
            
            # Calculate total energy consumption (Appliance + Base)
            # Reuse logic from simulator.run_simulation
            consumption_stats = simulator.environment_agent.get_consumption_stats()
            step_base_consumptions = consumption_stats.get('step_base_consumptions', {})
            
            appliance_totals = [sum(simulator.environment_agent.step_consumptions[k]) for k in step_keys]
            base_consumption = [step_base_consumptions.get(k, 0.0) for k in step_keys]
            total_consumption = [a + b for a, b in zip(appliance_totals, base_consumption)]
            
            # 4. Normalize
            normalized_vals = self.normalize_data(total_consumption)
            
            # 5. Update CSV (if not last round)
            if i < iterations - 1:
                self.update_feedback_csv(step_keys, normalized_vals)
                
                # Get and update behavior statistics
                behavior_stats = simulator.environment_agent.get_behavior_stats()
                self.update_simulated_behavior_json(behavior_stats)
                
                logger.info(f"Round {i+1} completed, feedback data and behavior data updated, preparing for next round...")
            else:
                logger.info(f"Round {i+1} completed (last round), no need to update feedback data.")
                
            # Explicit cleanup
            del simulator
            import gc
            gc.collect()
