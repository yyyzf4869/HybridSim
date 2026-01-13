"""
Standalone Version of Campus Energy Consumption Simulation

This is a standalone campus energy consumption simulation system that simulates student energy usage behavior on campus.

Main modules:
- simulator: Main simulator class
- student_agent: Student agent
- env_agent: Environment agent
- events: Event system
- base_agent: Base agent class
"""

from .simulator import EnergyConsumptionSimulator
from .student_agent import StudentAgent
from .env_agent import EnvAgent
from .events import Event, TimeEvent, EnergyConsumptionEvent, ConsumptionRecordedEvent
from .base_agent import BaseAgent, LLMAgent

__version__ = "1.0.0"
__author__ = "Energy Simulation Team"

__all__ = [
    'EnergyConsumptionSimulator',
    'StudentAgent',
    'EnvAgent',
    'Event',
    'TimeEvent',
    'EnergyConsumptionEvent',
    'ConsumptionRecordedEvent',
    'BaseAgent',
    'LLMAgent'
]
