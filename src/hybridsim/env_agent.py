import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from .events import EnergyConsumptionEvent, ConsumptionRecordedEvent

logger = logging.getLogger(__name__)

class EnvAgent:
    """Environment Agent - Manages campus environment and energy consumption records"""
    
    def __init__(self, agent_id: str = "environment", total_agents: int = 100, time_step: float = 1.0, environmental_condition: str = "normal"):
        self.agent_id = agent_id
        self.name = "CampusEnvironment"
        self.total_agents = total_agents  # Total agent count
        self.time_step = time_step  # Time step (hours)
        self.environmental_condition = environmental_condition  # Environmental condition parameter
        
        # Environmental state
        self.total_consumption = 0.0
        self.student_consumptions = {}  # Energy consumption records for each student
        self.environmental_conditions = {'condition': environmental_condition}  # Initialize environmental conditions
        self.current_time = "08:00"
        self.simulation_state = "initialized"
        
        # Data collection
        self.consumption_history = []
        self.step_consumptions = {}  # Consumption by time step
        self.step_location_summary = {}  # Consumption summary by location per time step
        self.step_behavior_records = {}  # Detailed behavior records per time step (for generating behavior stats)
        
        logger.info(f"Environment Agent {self.name} created - agent count: {total_agents}, time step: {time_step} hours")
    
    async def initialize(self):
        """Initialize environment agent"""
        logger.info(f"Environment Agent {self.agent_id} initializing...")
        # Specific initialization logic can be added here
        logger.info(f"Environment Agent {self.agent_id} initialization completed")
    
    def _calculate_base_consumption(self, location: str, timestamp: str) -> float:
        """Calculate base consumption (includes random fluctuation factor)"""
        import random
        
        # Base consumption configuration (per 100 agents, per hour)
        base_consumption_config = {
            "dormitory": {"06-10": 0.85, "10-14": 1.25, "14-18": 1.05, "18-22": 1.45, "22-06": 0.65},
            "classroom": {"08-12": 1.25, "12-14": 0.85, "14-18": 1.35, "18-22": 1.15, "22-08": 0.55},
            "library": {"08-12": 1.15, "12-14": 0.75, "14-18": 1.25, "18-22": 1.05, "22-08": 0.45},
            "laboratory": {"08-12": 1.45, "12-14": 1.15, "14-18": 1.55, "18-22": 1.35, "22-08": 0.75},
            "canteen": {"06-08": 2.00, "08-10": 0.50, "10-13": 2.00, "13-16": 0.50, "16-19": 2.00, "19-06": 0.00},
            "other": {"06-10": 0, "10-14": 0, "14-18": 0, "18-22": 0, "22-06": 0}
        }
        
        # Extract hour from timestamp (assuming format "Monday 14:30")
        try:
            time_part = timestamp.split(' ')[1]  # Get "14:30"
            current_hour = int(time_part.split(':')[0])
        except:
            current_hour = 12  # Default to noon
        
        # Determine time slot
        if 6 <= current_hour < 10:
            time_slot = "06-10"
        elif 10 <= current_hour < 14:
            time_slot = "10-14"
        elif 14 <= current_hour < 18:
            time_slot = "14-18"
        elif 18 <= current_hour < 22:
            time_slot = "18-22"
        else:
            time_slot = "22-06"
        
        # Get base consumption value (per 100 agents, per hour)
        location_config = base_consumption_config.get(location, base_consumption_config["other"])
        base_value_per_100_agents_per_hour = location_config.get(time_slot, 0.5)
        
        # Adjust based on actual agent count and time step
        # Formula: (actual agent count / 100) * time step * base value
        adjusted_base_value = (self.total_agents / 100.0) * self.time_step * base_value_per_100_agents_per_hour
        
        # Add random fluctuation factor (between 0.7 and 1.3)
        random_factor = random.uniform(0.7, 1.3)
        
        # Calculate final base consumption
        final_base_consumption = adjusted_base_value * random_factor
        
        return round(final_base_consumption, 4)
    
    async def initialize_environment(self, event):
        """Initialize environment"""
        logger.info(f"Environment Agent initializing...")
        self.simulation_state = "running"
        
        # No longer loading environment configuration file, using default configuration
        self.environmental_conditions = {}
        logger.info("Environment Agent initialization completed, state: {}".format(self.simulation_state))
    
    async def record_consumption(self, event: EnergyConsumptionEvent):
        """Record student energy consumption"""
        import random
        
        student_id = event.student_id
        consumption = event.consumption
        location = event.location
        intention = getattr(event, 'intention', 'weak')
        appliance = event.appliance
        timestamp = event.timestamp
        model_type = getattr(event, 'model_type', 'llm')  
        confidence = getattr(event, 'confidence', 0.0)   
        student_type = getattr(event, 'student_type', 'undergraduate') 
        
        # Base consumption should be calculated by location, not by student
        # Here we only record appliance consumption, base consumption is managed by the environment
        
        # Canteen dining consumption
        # If location is canteen, it adds 0.2kW (simulating 200W) power consumption
        # Multiplied by a random fluctuation factor between 0.7 and 1.3
        canteen_consumption = 0.0
        if location == 'canteen':
            canteen_consumption = 0.2 * random.uniform(0.7, 1.3)
            
        total_consumption = consumption + canteen_consumption
        
        # Update total consumption
        self.total_consumption += total_consumption
        
        # Update student personal consumption
        if student_id not in self.student_consumptions:
            self.student_consumptions[student_id] = 0.0
        self.student_consumptions[student_id] += total_consumption
        
        # Record history (only recording appliance consumption)
        consumption_record = {
            'environmental_conditions': self.environmental_condition,  
            'timestamp': timestamp,
            'student_id': student_id,
            'student_type': student_type, 
            'consumption': total_consumption,  
            'appliance_consumption': consumption,
            'canteen_consumption': canteen_consumption, 
            'base_consumption': 0.0,  
            'location': location,
            'intention': intention,
            'appliance': appliance,
            'total_consumption': self.total_consumption,
            'model_type': model_type,   
            'confidence': confidence,     
            'previous_location': getattr(event, 'previous_location', location), 
            'previous_appliance': getattr(event, 'previous_appliance', appliance), 
            'thermal_preference': getattr(event, 'thermal_preference', 'normal'),  
            'occupational_category': getattr(event, 'occupational_category', 'engineering')  
        }
        self.consumption_history.append(consumption_record)
        
        # Record by time step
        time_key = timestamp  # Use full timestamp format as key
        if time_key not in self.step_consumptions:
            self.step_consumptions[time_key] = []
        self.step_consumptions[time_key].append(total_consumption)

        # Location distribution statistics (only statistics appliance consumption)
        if time_key not in self.step_location_summary:
            self.step_location_summary[time_key] = {
                'dormitory': 0.0,
                'classroom': 0.0,
                'library': 0.0,
                'laboratory': 0.0,
                'other': 0.0
            }
        if location in self.step_location_summary[time_key]:
            self.step_location_summary[time_key][location] += total_consumption
        
        logger.info(f"Recorded consumption - Student: {student_id}, Appliance Consumption: {consumption:.3f} kWh, Location: {location}, Appliance: {appliance}, Model: {model_type}, Confidence: {confidence:.3f}, Total Consumption: {self.total_consumption:.3f} kWh")
        
        # Return confirmation event
        return ConsumptionRecordedEvent(
            from_agent_id=self.agent_id,
            to_agent_id=student_id,
            student_id=student_id,
            consumption_amount=consumption,
            total_consumption=self.total_consumption
        )
    
    def _calculate_step_base_consumption(self, timestamp: str) -> float:
        """Calculate base consumption for a time step (based on all agents)"""
        # Locations list in base consumption configuration
        locations = ["dormitory", "classroom", "library", "laboratory", "canteen", "other"]
        
        # Calculate base consumption for each location and sum up
        total_base_consumption = 0.0
        for location in locations:
            base_consumption = self._calculate_base_consumption(location, timestamp)
            total_base_consumption += base_consumption
            
        return total_base_consumption
    
    def get_behavior_stats(self) -> List[Dict[str, Any]]:
        """Generate behavior statistics (for Feedback Pipeline)"""
        if not self.consumption_history:
            return []
            
        # Aggregate statistics
        stats_map = {}
        
        for record in self.consumption_history:
            timestamp = record['timestamp']
            student_type = record.get('student_type', 'undergraduate')
            location = record['location']
            appliance = record['appliance']
            
            # Generate unique key
            key = f"{timestamp}_{student_type}_{location}_{appliance}"
            
            if key not in stats_map:
                stats_map[key] = {
                    "timestamp": timestamp,
                    "student_type": student_type,
                    "location": location,
                    "appliance": appliance,
                    "energy": 0.0,
                    "count": 0
                }
            
            stats_map[key]["energy"] += record['consumption']
            stats_map[key]["count"] += 1
            
        return list(stats_map.values())

    def get_consumption_stats(self) -> Dict[str, Any]:
        """Get consumption statistics"""
        if not self.consumption_history:
            return {
                'total_consumption': 0.0,
                'student_count': 0,
                'average_per_student': 0.0,
                'location_summary': {},
                'appliance_summary': {},
                'base_consumption_summary': {},
                'total_base_consumption': 0.0,
                'total_appliance_consumption': 0.0,
                'step_base_consumptions': {}  
            }
        
        # Statistics by location (only appliance consumption)
        location_summary = {}
        appliance_summary = {}
        base_consumption_summary = {}
        thermal_preference_summary = {} 
        occupational_category_summary = {} 
        total_appliance_consumption = 0.0
        step_base_consumptions = {} 
        
        records = self.consumption_history
        grad_recs = []
        undergrad_recs = []
        for rec in records:
            # Get student_type
            stype = rec.get('student_type', 'undergraduate')
            if stype == 'graduate':
                grad_recs.append(rec)
            else:
                undergrad_recs.append(rec)
        
        # Calculate base consumption for each time step first
        unique_timestamps = set(record['timestamp'] for record in self.consumption_history)
        for timestamp in unique_timestamps:
            step_base_consumptions[timestamp] = self._calculate_step_base_consumption(timestamp)
        
        for record in self.consumption_history:
            location = record['location']
            appliance = record['appliance']
            appliance_consumption = record['appliance_consumption']
            thermal_preference = record.get('thermal_preference', 'normal')
            occupational_category = record.get('occupational_category', 'engineering')
            
            # Location statistics (only includes appliance consumption)
            location_summary[location] = location_summary.get(location, 0.0) + record['consumption']
            
            # Appliance consumption statistics (only real appliance consumption)
            if appliance != "none":
                appliance_summary[appliance] = appliance_summary.get(appliance, 0.0) + appliance_consumption
            
            # Thermal preference statistics
            thermal_preference_summary[thermal_preference] = thermal_preference_summary.get(thermal_preference, 0.0) + record['consumption']
            
            # Occupational category statistics
            occupational_category_summary[occupational_category] = occupational_category_summary.get(occupational_category, 0.0) + record['consumption']
            
            total_appliance_consumption += appliance_consumption
        
        # Calculate total base consumption (sum of base consumption for all time steps)
        total_base_consumption = sum(step_base_consumptions.values())
        
        # Simplify base consumption by location distribution statistics (calculated based on all locations)
        locations = ["dormitory", "classroom", "library", "laboratory", "canteen", "other"]
        for timestamp in unique_timestamps:
            # Calculate base consumption distribution for each location
            for location in locations:
                base_consumption = self._calculate_base_consumption(location, timestamp)
                base_consumption_summary[location] = base_consumption_summary.get(location, 0.0) + base_consumption
        
        # Calculate total consumption (base + appliance)
        total_consumption = total_base_consumption + total_appliance_consumption
        
        student_count = len(self.student_consumptions)
        
        return {
            'total_consumption': total_consumption, 
            'student_count': student_count,
            'average_per_student': total_consumption / student_count if student_count > 0 else 0.0,
            'location_summary': location_summary,
            'appliance_summary': appliance_summary,
            'base_consumption_summary': base_consumption_summary,  
            'total_base_consumption': total_base_consumption,
            'total_appliance_consumption': total_appliance_consumption,
            'step_base_consumptions': step_base_consumptions,  
            'consumption_history': self.consumption_history[-10:]  
        }
    
    def export_data(self, filename: str = None, output_dir: str = None):
        """Export data to file"""
        if output_dir is not None:
            try:
                import os
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create output directory: {e}")
        
        if filename is None:
            filename = f"energy_consumption_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        full_path = filename
        if output_dir is not None:
            try:
                import os
                full_path = os.path.join(output_dir, filename)
            except Exception:
                full_path = filename
        
        # Get latest statistics
        consumption_stats = self.get_consumption_stats()
        
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_consumption': consumption_stats['total_consumption'], 
                'student_count': len(self.student_consumptions),
                'simulation_state': self.simulation_state
            },
            'consumption_stats': consumption_stats,
            'student_consumptions': self.student_consumptions,
            'consumption_history': self.consumption_history,
            'step_consumptions': self.step_consumptions,
            'step_location_summary': self.step_location_summary
        }
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data export completed: {full_path}")
        except Exception as e:
            logger.error(f"Data export failed: {e}")
        
        return full_path
