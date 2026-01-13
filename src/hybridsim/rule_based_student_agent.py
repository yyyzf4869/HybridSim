"""
Rule-based Student Agent - makes decisions completely based on rule engine
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from .events import TimeEvent, EnergyConsumptionEvent
from .base_agent import BaseAgent
from .rule_engine import RuleEngine
import random

logger = logging.getLogger(__name__)

class RuleBasedStudentAgent(BaseAgent):
    """Rule-based Student Agent - makes decisions completely based on predefined rules"""
    
    def __init__(self, agent_id: str, profile_data: Dict[str, Any]):
        super().__init__(agent_id, profile_data.get('name', 'Unknown'))
        
        self.profile_data = profile_data
        self.occupation = profile_data.get('occupation', 'undergraduate')
        self.occupational_category = profile_data.get('occupational_category', 'engineering')
        self.age = profile_data.get('age', 20)
        self.work_intensity = profile_data.get('work_intensity', 'medium')
        self.energy_demand = profile_data.get('energy_demand', 'normal')
        self.thermal_preference = profile_data.get('thermal_preference', 'normal')
        
        # Student state
        self.current_location = "other"
        self.current_appliance = "none"
        self.energy_intention = "weak"
        self.total_consumption = 0.0
        
        # Rule engine
        self.rule_engine = RuleEngine()
        
        # Schedule (Simple simulation)
        self.class_schedule = self._generate_class_schedule()
        
        # Social state
        self.energy_awareness = self._calculate_initial_awareness()
        self.social_connections = []
        
        # Room state (for multi-person interaction)
        self.room_occupancy = 1  # Default 1 person in room
        
        logger.info(f"Rule-based Student Agent {self.name} created - "
                   f"Occupation: {self.occupation}, Category: {self.occupational_category}, "
                   f"Age: {self.age}, Work Intensity: {self.work_intensity}, "
                   f"Energy Demand: {self.energy_demand}, Thermal Preference: {self.thermal_preference}")
    
    def _generate_class_schedule(self) -> Dict[str, Any]:
        """Generate class schedule"""
        # Simplified schedule: classes on weekdays, no classes on weekends
        schedule = {
            'Monday': {'has_class': True, 'classes': ['morning', 'afternoon']},
            'Tuesday': {'has_class': True, 'classes': ['morning', 'afternoon']},
            'Wednesday': {'has_class': True, 'classes': ['morning', 'afternoon']},
            'Thursday': {'has_class': True, 'classes': ['morning', 'afternoon']},
            'Friday': {'has_class': True, 'classes': ['morning']},
            'Saturday': {'has_class': False, 'classes': []},
            'Sunday': {'has_class': False, 'classes': []}
        }
        return schedule
    
    def _calculate_initial_awareness(self) -> float:
        """Calculate initial energy saving awareness"""
        # Calculate awareness based on student characteristics
        awareness = 0.5  # Base awareness
        
        # Work intensity impact
        if self.work_intensity == 'high':
            awareness += 0.2
        elif self.work_intensity == 'low':
            awareness -= 0.2
        
        # Energy demand impact
        if self.energy_demand == 'low':
            awareness += 0.15
        elif self.energy_demand == 'high':
            awareness -= 0.15
        
        # Age impact (assume older implies more mature and higher awareness)
        age_factor = (self.age - 18) / 12  # Normalize 18-30
        awareness += age_factor * 0.1
        
        return max(0.0, min(1.0, awareness))  # Limit to 0-1 range
    
    def _calculate_energy_consumption(self, appliance: str, usage_hours: float, 
                                     intention: str = "weak") -> float:
        """Calculate energy consumption"""
        # Appliance power rating (kWh)
        power_ratings = {
            "laptop": 0.1,              # 100W
            "phone_charger": 0.05,      # 50W
            "light": 0.015,             # 15W
            "air_conditioner": 1.0,       # 1000W
            "experimental_equipment": 2.5,  # 2500W
            "none": 0.0
        }
        
        base_power = power_ratings.get(appliance, 0.05)
        
        # Adjust power based on intention strength
        intention_multipliers = {
            "weak": 0.9,      # weak equivalent to *0.9 power
            "moderate": 1.0,  # moderate equivalent to *1.0 power
            "strong": 1.1,    # strong equivalent to *1.1 power
            "none": 1.0       # usually no appliance used when none, but calculate normally if used
        }
        
        intention_multiplier = intention_multipliers.get(intention, 1.0)
        
        # Calculate appliance consumption = base power * usage time * intention strength
        consumption = base_power * usage_hours * intention_multiplier
        
        return round(consumption, 4)
    
    def _get_class_status(self, current_time: str) -> bool:
        """Get current class status"""
        try:
            weekday_str, _ = current_time.rsplit(' ', 1)
            day_schedule = self.class_schedule.get(weekday_str, {'has_class': True})
            return day_schedule['has_class']
        except:
            return True  # Default to having class
    
    def _generate_neighbors(self) -> list:
        """Generate neighbor list (for social interaction)"""
        # Simplified handling: return empty list, can get from environment in actual application
        return []
    
    def _process_social_interactions(self, social_message: Dict[str, Any]):
        """Process social interactions"""
        if not social_message:
            return
        
        message_type = social_message.get('type')
        awareness_change = social_message.get('awareness_change', 0)
        
        # Decide whether to accept influence based on acceptance rate
        acceptance_rate = self.rule_engine.social_rules['acceptance_rate'][f'{self.energy_awareness:.1f}_awareness']
        
        if random.random() < acceptance_rate:
            # Accept influence, adjust energy awareness
            old_awareness = self.energy_awareness
            self.energy_awareness += awareness_change
            self.energy_awareness = max(0.0, min(1.0, self.energy_awareness))
            
            logger.info(f"Student {self.name} influenced by social interaction: {message_type}, "
                       f"Energy awareness changed from {old_awareness:.2f} to {self.energy_awareness:.2f}")
    
    async def handle_time_event(self, event: TimeEvent) -> EnergyConsumptionEvent:
        """Handle time event - make decision based on rules"""
        current_time = event.current_time
        environmental_context = event.environmental_context
        
        logger.info(f"Rule-based Student {self.name} handling time event: {current_time}")
        
        # Get current class status
        has_class = self._get_class_status(current_time)
        
        # Generate neighbor list
        neighbors = self._generate_neighbors()
        
        # Build additional context
        additional_context = {
            'has_class': has_class,
            'room_occupancy': self.room_occupancy,
            'neighbors': neighbors,
            'energy_awareness': self.energy_awareness
        }
        
        # Make decision using rule engine
        decision = self.rule_engine.make_decision(
            current_time, self.profile_data, environmental_context, additional_context
        )
        
        # Extract decision results
        location = decision['location']
        intention = decision['intention']
        appliance = decision['appliance']
        social_message = decision.get('social_message')
        
        # Process social interaction
        if social_message:
            self._process_social_interactions(social_message)
        
        # Calculate usage time (equal to time step)
        usage_hours = event.time_step if appliance != "none" else 0
        
        # Calculate energy consumption
        consumption = self._calculate_energy_consumption(appliance, usage_hours, intention)
        
        # Update state
        self.previous_location = self.current_location
        self.previous_appliance = self.current_appliance
        self.current_location = location
        self.current_appliance = appliance
        self.energy_intention = intention
        self.total_consumption += consumption
        
        logger.info(f"Rule-based Student {self.name} decision: "
                   f"Location: {location}, Appliance: {appliance}, "
                   f"Usage Time: {usage_hours:.1f}h, Consumption: {consumption:.3f} kWh, "
                   f"Intention: {intention}")
        
        # Return energy consumption event
        return EnergyConsumptionEvent(
            from_agent_id=self.agent_id,
            to_agent_id="environment",
            student_id=self.agent_id,
            consumption=consumption,
            location=location,
            intention=intention,
            appliance=appliance,
            scenario="rule_based",
            timestamp=current_time,
            model_type="rule", 
            confidence=0.95, 
            thermal_preference=self.thermal_preference,
            occupational_category=self.occupational_category,
            previous_location=self.previous_location,
            previous_appliance=self.previous_appliance
        )
    
    async def initialize(self):
        """Initialize student agent"""
        logger.info(f"Rule-based Student Agent {self.name} initializing...")
        
        # Initialize social connections
        self.social_connections = []
        
        # Initialize energy awareness
        self.energy_awareness = self._calculate_initial_awareness()
        
        logger.info(f"Rule-based Student Agent {self.name} initialization completed - "
                   f"Initial energy awareness: {self.energy_awareness:.2f}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get agent state summary"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'current_location': self.current_location,
            'current_appliance': self.current_appliance,
            'energy_intention': self.energy_intention,
            'total_consumption': self.total_consumption,
            'energy_awareness': self.energy_awareness,
            'occupation': self.occupation,
            'work_intensity': self.work_intensity,
            'energy_demand': self.energy_demand
        }
