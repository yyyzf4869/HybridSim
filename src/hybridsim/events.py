from datetime import datetime
from typing import Dict, Any
import json

class Event:
    """Base Event Class"""
    
    def __init__(self, event_type: str, from_agent_id: str, to_agent_id: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.from_agent_id = from_agent_id
        self.to_agent_id = to_agent_id
        self.data = data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'from_agent_id': self.from_agent_id,
            'to_agent_id': self.to_agent_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

class TimeEvent(Event):
    """Time Event - Used to drive simulation time forward"""
    
    def __init__(self, from_agent_id: str, to_agent_id: str, current_time: str, time_step: float, environmental_context: Dict[str, Any]):
        super().__init__("TimeEvent", from_agent_id, to_agent_id)
        self.current_time = current_time
        self.time_step = time_step
        self.environmental_context = environmental_context
        
        self.data = {
            'current_time': current_time,
            'time_step': time_step,
            'environmental_context': environmental_context
        }

class EnergyConsumptionEvent(Event):
    """Energy Consumption Event - Student agent reports energy consumption to environment agent"""
    
    def __init__(self, from_agent_id: str, to_agent_id: str, student_id: str, consumption: float, 
                 location: str, intention: str, appliance: str, scenario: str, timestamp: str,
                 model_type: str = "llm", confidence: float = 0.0, thermal_preference: str = "normal",
                 occupational_category: str = "engineering", previous_location: str = "other", 
                 previous_appliance: str = "none", student_type: str = "undergraduate"):
        super().__init__("EnergyConsumptionEvent", from_agent_id, to_agent_id)
        self.student_id = student_id
        self.consumption = consumption
        self.location = location
        self.intention = intention
        self.appliance = appliance
        self.scenario = scenario
        self.timestamp = timestamp
        self.model_type = model_type
        self.confidence = confidence
        self.thermal_preference = thermal_preference
        self.occupational_category = occupational_category
        self.previous_location = previous_location
        self.previous_appliance = previous_appliance
        self.student_type = student_type
        
        self.data = {
            'student_id': student_id,
            'consumption': consumption,
            'location': location,
            'intention': intention,
            'appliance': appliance,
            'scenario': scenario,
            'timestamp': timestamp,
            'model_type': model_type,
            'confidence': confidence,
            'thermal_preference': thermal_preference,
            'occupational_category': occupational_category,
            'previous_location': previous_location,
            'previous_appliance': previous_appliance,
            'student_type': student_type
        }

class ConsumptionRecordedEvent(Event):
    """Consumption Recorded Event - Environment agent confirms receipt of student's energy consumption report"""
    
    def __init__(self, from_agent_id: str, to_agent_id: str, student_id: str, 
                 consumption_amount: float, total_consumption: float):
        super().__init__("ConsumptionRecordedEvent", from_agent_id, to_agent_id)
        self.student_id = student_id
        self.consumption_amount = consumption_amount
        self.total_consumption = total_consumption
        
        self.data = {
            'student_id': student_id,
            'consumption_amount': consumption_amount,
            'total_consumption': total_consumption
        }
