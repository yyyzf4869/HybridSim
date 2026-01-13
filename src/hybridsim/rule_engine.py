"""
Rule Engine Module - Implements pure rule-driven energy consumption simulation
Decisions are made based on student characteristics and environmental conditions using predefined rules
"""

import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class RuleEngine:
    """Pure Rule Engine - Rule-based decision system based on student characteristics and environmental conditions"""
    
    def __init__(self):
        # Schedule rules configuration
        self.schedule_rules = self._init_schedule_rules()
        
        # Appliance usage rules configuration
        self.appliance_rules = self._init_appliance_rules()
        
        # Social interaction rules configuration
        self.social_rules = self._init_social_rules()
        
        # Environmental condition mapping
        self.environmental_mapping = {
            'hot': {'temperature_range': (28, 40), 'description': 'Hot'},
            'warm': {'temperature_range': (22, 28), 'description': 'Warm'},
            'cool': {'temperature_range': (15, 22), 'description': 'Cool'},
            'cold': {'temperature_range': (0, 15), 'description': 'Cold'},
            'bad weather': {'temperature_range': (10, 25), 'description': 'Bad Weather'}
        }
        
        logger.info("Rule Engine initialization completed")
    
    def _init_schedule_rules(self) -> Dict[str, Any]:
        """Initialize schedule rules"""
        return {
            'wake_up': {
                'weekday_early': {'time': '07:00', 'condition': 'Weekday with class'},
                'weekend_late': {'time': '09:00', 'condition': 'Weekend or no class'},
                'work_intensity_factor': {
                    'high': -30,  # High work intensity wakes up 30 minutes earlier
                    'medium': 0,   # Medium work intensity normal time
                    'low': 30      # Low work intensity wakes up 30 minutes later
                }
            },
            'meal_times': {
                'breakfast': {'start': '07:00', 'end': '09:00'},
                'lunch': {'start': '11:30', 'end': '13:30'},
                'dinner': {'start': '17:30', 'end': '19:30'}
            },
            'study_times': {
                'undergraduate': {'start': '08:00', 'end': '17:00'},
                'graduate student': {'start': '09:00', 'end': '18:00'}
            },
            'sleep_times': {
                'high_intensity': {'start': '23:00', 'end': '06:00'},
                'medium_intensity': {'start': '23:30', 'end': '07:00'},
                'low_intensity': {'start': '00:30', 'end': '08:00'}
            }
        }
    
    def _init_appliance_rules(self) -> Dict[str, Any]:
        """Initialize appliance usage rules"""
        return {
            'light': {
                'on_rules': {
                    'high_awareness': {'time_threshold': '19:00', 'description': 'High energy awareness, turns on light after 19:00'},
                    'medium_awareness': {'time_threshold': '18:00', 'description': 'Medium energy awareness, turns on light after 18:00'},
                    'low_awareness': {'time_threshold': '17:00', 'description': 'Low energy awareness, turns on light after 17:00'}
                },
                'off_rules': {
                    'no_people': True,      # Force turn off when no one is present
                    'all_sleeping': True,   # Turn off when everyone is sleeping
                    'daylight_hours': {'start': '08:00', 'end': '17:00'}  # Daylight hours
                }
            },
            'air_conditioner': {
                'on_temperature_threshold': 26,  # Temperature threshold
                'temperature_settings': {
                    'day_preference': {'temperature': 24, 'condition': 'Daytime preference'},
                    'night_preference': {'temperature': 22, 'condition': 'Nighttime preference'}
                },
                'off_rules': {
                    'last_person_leave': True,  # Turn off AC when the last person leaves
                    'forget_probability': 0.3     # Probability of forgetting to turn off AC (low energy awareness)
                }
            },
            'computer': {
                'usage_probability': {
                    'high_work_intensity': 0.9,
                    'medium_work_intensity': 0.7,
                    'low_work_intensity': 0.4
                },
                'shutdown_behavior': {
                    'high_awareness': 'turn_off',      # High energy awareness completely shuts down
                    'low_awareness': 'standby'         # Low energy awareness uses standby
                }
            },
            'phone_charger': {
                'usage_probability': {
                    'evening': 0.8,      # Usage probability in the evening
                    'leisure_time': 0.6   # Usage probability during leisure time
                }
            }
        }
    
    def _init_social_rules(self) -> Dict[str, Any]:
        """Initialize social interaction rules"""
        return {
            'positive_influence': {
                'high_awareness': {
                    'messages': ['be greener!', 'stop wasting!'],
                    'influence_probability': 0.3,
                    'awareness_change': 0.1  # Magnitude of energy awareness increase
                }
            },
            'negative_influence': {
                'low_awareness': {
                    'threshold': 0.3,  # Energy awareness threshold
                    'messages': ['Become Wasting!'],
                    'influence_probability': 0.2,
                    'awareness_change': -0.1  # Magnitude of energy awareness decrease
                }
            },
            'acceptance_rate': {
                'high_awareness': 0.8,   # Probability of high energy awareness students accepting suggestions
                'medium_awareness': 0.5, # Probability of medium energy awareness students accepting suggestions
                'low_awareness': 0.2     # Probability of low energy awareness students accepting suggestions
            }
        }
    
    def parse_time(self, time_str: str) -> int:
        """Convert time string to minutes (minutes from 00:00)"""
        if ':' in time_str:
            hour, minute = map(int, time_str.split(':'))
            return hour * 60 + minute
        return 0
    
    def is_weekend(self, day_of_week: int) -> bool:
        """Determine if it is a weekend (5=Saturday, 6=Sunday)"""
        return day_of_week >= 5
    
    def is_meal_time(self, current_time_minutes: int, meal_type: str) -> bool:
        """Determine if it is meal time"""
        meal_config = self.schedule_rules['meal_times'][meal_type]
        start_minutes = self.parse_time(meal_config['start'])
        end_minutes = self.parse_time(meal_config['end'])
        return start_minutes <= current_time_minutes <= end_minutes
    
    def is_sleep_time(self, current_time_minutes: int, work_intensity: str) -> bool:
        """Determine if it is sleep time"""
        sleep_config = self.schedule_rules['sleep_times'][f'{work_intensity}_intensity']
        sleep_start = self.parse_time(sleep_config['start'])
        sleep_end = self.parse_time(sleep_config['end'])
        
        # Handle cross-day sleep time
        if sleep_start < sleep_end:
            return sleep_start <= current_time_minutes <= sleep_end
        else:
            # Cross-day situation (e.g., 23:00-06:00)
            return current_time_minutes >= sleep_start or current_time_minutes <= sleep_end
    
    def should_wake_up(self, current_time_minutes: int, day_of_week: int, 
                      has_class: bool, work_intensity: str) -> bool:
        """Determine if one should wake up"""
        # Base wake up time
        if self.is_weekend(day_of_week) or not has_class:
            base_wake_time = self.parse_time(self.schedule_rules['wake_up']['weekend_late']['time'])
        else:
            base_wake_time = self.parse_time(self.schedule_rules['wake_up']['weekday_early']['time'])
        
        # Adjust based on work intensity
        intensity_factor = self.schedule_rules['wake_up']['work_intensity_factor'][work_intensity]
        adjusted_wake_time = base_wake_time + intensity_factor
        
        return current_time_minutes >= adjusted_wake_time
    
    def get_energy_awareness_level(self, student_profile: Dict[str, Any]) -> str:
        """Determine energy awareness level based on student characteristics"""
        # Infer energy awareness based on work intensity, energy demand, etc.
        work_intensity = student_profile.get('work_intensity', 'medium')
        energy_demand = student_profile.get('energy_demand', 'medium')
        
        if work_intensity == 'high' and energy_demand == 'low':
            return 'high'
        elif work_intensity == 'low' and energy_demand == 'high':
            return 'low'
        else:
            return 'medium'
    
    def decide_light_usage(self, current_time_minutes: int, energy_awareness: str,
                          room_occupancy: int, is_daylight: bool) -> Tuple[bool, str]:
        """Decide on light usage"""
        light_rules = self.appliance_rules['light']
        
        # Check force off conditions
        if room_occupancy == 0:  # Force turn off when no one is present
            return False, "Force turn off when no one is present"
        
        if is_daylight:  # Daylight hours
            return False, "Utilize natural light during the day"
        
        # Decide turn on time based on energy awareness
        awareness_rules = light_rules['on_rules'][f'{energy_awareness}_awareness']
        threshold_time = self.parse_time(awareness_rules['time_threshold'])
        
        current_hour = current_time_minutes // 60
        threshold_hour = threshold_time // 60
        
        if current_hour >= threshold_hour:
            return True, f"{awareness_rules['description']}"
        
        return False, "Too early, no need to turn on lights"
    
    def decide_air_conditioner_usage(self, current_temperature: float, room_occupancy: int,
                                     energy_awareness: str, is_night: bool) -> Tuple[bool, float, str]:
        """Decide on air conditioner usage"""
        ac_rules = self.appliance_rules['air_conditioner']
        
        # Temperature threshold check
        if current_temperature < ac_rules['on_temperature_threshold']:
            return False, 0.0, "Temperature has not reached the threshold for turning on AC"
        
        # Decide temperature setting
        if is_night:
            target_temp = ac_rules['temperature_settings']['night_preference']['temperature']
            reason = ac_rules['temperature_settings']['night_preference']['condition']
        else:
            target_temp = ac_rules['temperature_settings']['day_preference']['temperature']
            reason = ac_rules['temperature_settings']['day_preference']['condition']
        
        # Check off conditions
        if room_occupancy <= 1:  # Last person leaving
            # Students with low energy awareness might forget to turn off AC
            if energy_awareness == 'low':
                forget_prob = ac_rules['off_rules']['forget_probability']
                if random.random() < forget_prob:
                    return True, target_temp, f"Low energy awareness, forgot to turn off AC ({reason})"
            
            return False, 0.0, "Last person leaving, turn off AC"
        
        return True, target_temp, f"AC adjustment needed ({reason})"
    
    def decide_computer_usage(self, work_intensity: str, is_study_time: bool,
                            is_evening: bool) -> Tuple[bool, str]:
        """Decide on computer usage"""
        computer_rules = self.appliance_rules['computer']
        
        # Base usage probability
        usage_prob = computer_rules['usage_probability'][f'{work_intensity}_work_intensity']
        
        # Adjust probability based on time and activity
        if is_study_time:
            usage_prob *= 1.5  # Increase usage probability during study time
        elif is_evening:
            usage_prob *= 1.2  # Increase usage probability in the evening
        
        # Random decision
        if random.random() < usage_prob:
            return True, f"Work intensity {work_intensity}, need to use computer"
        
        return False, "No need to use computer currently"
    
    def decide_phone_charger_usage(self, is_evening: bool, is_leisure_time: bool) -> Tuple[bool, str]:
        """Decide on phone charger usage"""
        charger_rules = self.appliance_rules['phone_charger']
        
        usage_prob = 0.0
        
        if is_evening:
            usage_prob = charger_rules['usage_probability']['evening']
        elif is_leisure_time:
            usage_prob = charger_rules['usage_probability']['leisure_time']
        
        if random.random() < usage_prob:
            return True, "Need to charge phone during evening/leisure time"
        
        return False, "No need to charge currently"
    
    def generate_social_message(self, student_profile: Dict[str, Any], 
                               neighbors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate social interaction message"""
        energy_awareness = self.get_energy_awareness_level(student_profile)
        social_rules = self.social_rules
        
        # Positive influence (High energy awareness)
        if energy_awareness == 'high':
            influence_config = social_rules['positive_influence']['high_awareness']
            if random.random() < influence_config['influence_probability']:
                message = random.choice(influence_config['messages'])
                return {
                    'type': 'positive',
                    'message': message,
                    'sender_awareness': energy_awareness,
                    'awareness_change': influence_config['awareness_change']
                }
        
        # Negative influence (Low energy awareness)
        elif energy_awareness == 'low':
            # Need to check if energy awareness is low enough
            # Simplified here, assuming low energy awareness is below threshold
            influence_config = social_rules['negative_influence']['low_awareness']
            if random.random() < influence_config['influence_probability']:
                message = random.choice(influence_config['messages'])
                return {
                    'type': 'negative',
                    'message': message,
                    'sender_awareness': energy_awareness,
                    'awareness_change': influence_config['awareness_change']
                }
        
        return None
    
    def process_social_influence(self, receiver_profile: Dict[str, Any], 
                                message: Dict[str, Any]) -> bool:
        """Process received social influence"""
        receiver_awareness = self.get_energy_awareness_level(receiver_profile)
        acceptance_rate = self.social_rules['acceptance_rate'][f'{receiver_awareness}_awareness']
        
        # Randomly decide whether to accept influence
        if random.random() < acceptance_rate:
            return True  # Accept influence
        
        return False  # Reject influence
    
    def decide_location(self, current_time_minutes: int, day_of_week: int,
                       student_profile: Dict[str, Any], has_class: bool) -> Tuple[str, str]:
        """Decide location (Rule-based)"""
        occupation = student_profile.get('occupation', 'undergraduate')
        work_intensity = student_profile.get('work_intensity', 'medium')
        occupational_category = student_profile.get('occupational_category', 'engineering')
        
        # Check sleep time
        if self.is_sleep_time(current_time_minutes, work_intensity):
            return 'dormitory', 'Sleeping time, in dormitory'
        
        # Check meal time
        if self.is_meal_time(current_time_minutes, 'breakfast'):
            return 'canteen', 'Breakfast time, go to canteen'
        elif self.is_meal_time(current_time_minutes, 'lunch'):
            return 'canteen', 'Lunch time, go to canteen'
        elif self.is_meal_time(current_time_minutes, 'dinner'):
            return 'canteen', 'Dinner time, go to canteen'
        
        # Study time decision
        study_config = self.schedule_rules['study_times'][occupation]
        study_start = self.parse_time(study_config['start'])
        study_end = self.parse_time(study_config['end'])
        
        is_study_time = study_start <= current_time_minutes <= study_end
        
        if is_study_time and has_class:
            # Decide study location based on occupational category and study time
            if occupation == 'graduate student':
                # Graduate students are more likely to be in the laboratory
                if occupational_category in ['engineering', 'science', 'medicine']:
                    if random.random() < 0.7:  # 70% probability to go to laboratory
                        return 'laboratory', f"Graduate student {occupational_category} major, in laboratory"
                    else:
                        return 'library', 'Graduate student studying in library'
                else:
                    if random.random() < 0.4:  # 40% probability to go to laboratory
                        return 'laboratory', f"Graduate student {occupational_category} major, in laboratory"
                    else:
                        return 'library', 'Graduate student studying in library'
            else:
                # Undergraduates are more likely to be in classrooms and libraries
                if random.random() < 0.6:  # 60% probability to go to classroom
                    return 'classroom', 'Undergraduate class time, in classroom'
                else:
                    return 'library', 'Undergraduate studying in library'
        elif is_study_time and not has_class:
            # Study time without class
            if random.random() < 0.7:  # 70% probability to go to library
                return 'library', 'Study time without class, go to library'
            else:
                return 'dormitory', 'Self-study in dormitory'
        
        # Leisure time
        current_hour = current_time_minutes // 60
        if current_hour >= 19:  # After 19:00
            if random.random() < 0.6:  # 60% probability in dormitory
                return 'dormitory', 'Evening leisure time, in dormitory'
            else:
                return 'library', 'Go to library in the evening'
        elif current_hour < 8:  # Before 8:00
            if self.should_wake_up(current_time_minutes, day_of_week, has_class, work_intensity):
                return 'dormitory', 'Morning in dormitory'
            else:
                return 'dormitory', 'Still sleeping'
        else:
            # Other leisure times
            if random.random() < 0.4:
                return 'dormitory', 'Leisure time in dormitory'
            elif random.random() < 0.7:
                return 'library', 'Leisure time go to library'
            else:
                return 'other', 'Activity in other places'
    
    def decide_intention(self, location: str, appliance: str, student_profile: Dict[str, Any],
                        environmental_condition: str) -> str:
        """Decide energy usage intention intensity"""
        work_intensity = student_profile.get('work_intensity', 'medium')
        energy_demand = student_profile.get('energy_demand', 'medium')
        
        # Base intention
        if work_intensity == 'high':
            base_intention = 'strong'
        elif work_intensity == 'medium':
            base_intention = 'moderate'
        else:
            base_intention = 'weak'
        
        # Adjust based on appliance type
        if appliance == 'none':
            return 'none'
        elif appliance == 'air_conditioner':
            # AC usage usually indicates strong energy demand
            if environmental_condition in ['hot', 'cold']:
                return 'strong'
            else:
                return base_intention
        elif appliance == 'experimental_equipment':
            # Experimental equipment usually indicates strong energy demand
            return 'strong'
        elif appliance == 'laptop':
            # Laptop depends on work intensity
            return base_intention
        elif appliance in ['light', 'phone_charger']:
            # Light and phone charger usually have moderate or weak demand
            if base_intention == 'strong':
                return 'moderate'
            else:
                return base_intention
        
        return base_intention
    
    def decide_appliance_usage(self, current_time_minutes: int, location: str,
                              student_profile: Dict[str, Any], environmental_context: Dict[str, Any],
                              room_occupancy: int = 1) -> Tuple[str, str]:
        """Decide appliance usage"""
        energy_awareness = self.get_energy_awareness_level(student_profile)
        work_intensity = student_profile.get('work_intensity', 'medium')
        occupational_category = student_profile.get('occupational_category', 'engineering')
        
        # Get environmental conditions
        environmental_condition = environmental_context.get('condition', 'normal')
        current_hour = current_time_minutes // 60
        
        # Check if it is daylight (8:00-17:00)
        is_daylight = 8 <= current_hour <= 17
        is_night = current_hour >= 19 or current_hour <= 6
        is_evening = 18 <= current_hour <= 23
        
        # Check if it is study time
        occupation = student_profile.get('occupation', 'undergraduate')
        study_config = self.schedule_rules['study_times'][occupation]
        study_start = self.parse_time(study_config['start'])
        study_end = self.parse_time(study_config['end'])
        is_study_time = study_start <= current_time_minutes <= study_end
        
        # Decide appliance usage based on location and environmental conditions
        appliances_to_consider = []
        
        # Light decision
        light_on, light_reason = self.decide_light_usage(
            current_time_minutes, energy_awareness, room_occupancy, is_daylight
        )
        if light_on:
            appliances_to_consider.append(('light', light_reason))
        
        # AC decision (only in specific locations)
        if location in ['dormitory', 'laboratory', 'library']:
            # Simulate current temperature (based on environmental conditions)
            temp_range = self.environmental_mapping.get(environmental_condition, 
                                                      self.environmental_mapping['warm'])
            current_temperature = random.uniform(temp_range['temperature_range'][0], 
                                                 temp_range['temperature_range'][1])
            
            ac_on, ac_temp, ac_reason = self.decide_air_conditioner_usage(
                current_temperature, room_occupancy, energy_awareness, is_night
            )
            if ac_on:
                appliances_to_consider.append(('air_conditioner', ac_reason))
        
        # Computer decision
        computer_on, computer_reason = self.decide_computer_usage(
            work_intensity, is_study_time, is_evening
        )
        if computer_on:
            appliances_to_consider.append(('laptop', computer_reason))
        
        # Experimental equipment decision (only graduate students in laboratory)
        if location == 'laboratory' and occupation == 'graduate student':
            if occupational_category in ['engineering', 'science', 'medicine']:
                if random.random() < 0.6:  # 60% probability to use experimental equipment
                    appliances_to_consider.append(('experimental_equipment', 'Graduate student conducting experiment in laboratory'))
        
        # Phone charger decision
        charger_on, charger_reason = self.decide_phone_charger_usage(is_evening, not is_study_time)
        if charger_on:
            appliances_to_consider.append(('phone_charger', charger_reason))
        
        # If no suitable appliance, return none
        if not appliances_to_consider:
            return 'none', 'No need to use any appliance currently'
        
        # Randomly select one appliance (avoid using multiple appliances simultaneously)
        selected_appliance, selected_reason = random.choice(appliances_to_consider)
        
        return selected_appliance, selected_reason
    
    def make_decision(self, current_time: str, student_profile: Dict[str, Any], 
                     environmental_context: Dict[str, Any], 
                     additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a complete decision based on rules
        
        Args:
            current_time: Current time (Format: "Monday 14:30")
            student_profile: Student profile
            environmental_context: Environmental context
            additional_context: Additional context (e.g., class schedule, room occupancy)
            
        Returns:
            Decision result dictionary
        """
        # Parse time
        weekday_str, time_part = current_time.rsplit(' ', 1)
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        try:
            day_of_week = weekdays.index(weekday_str)
        except ValueError:
            day_of_week = 0  # Default to Monday
        
        current_time_minutes = self.parse_time(time_part)
        
        # Get additional context
        has_class = additional_context.get('has_class', True) if additional_context else True
        room_occupancy = additional_context.get('room_occupancy', 1) if additional_context else 1
        
        # Make location decision
        location, location_reason = self.decide_location(
            current_time_minutes, day_of_week, student_profile, has_class
        )
        
        # Make appliance usage decision
        appliance, appliance_reason = self.decide_appliance_usage(
            current_time_minutes, location, student_profile, environmental_context, room_occupancy
        )
        
        # Make intention decision
        intention = self.decide_intention(location, appliance, student_profile, 
                                         environmental_context.get('condition', 'normal'))
        
        # Generate social message (optional)
        social_message = None
        if additional_context and 'neighbors' in additional_context:
            social_message = self.generate_social_message(student_profile, additional_context['neighbors'])
        
        # Build decision result
        decision = {
            'location': location,
            'intention': intention,
            'appliance': appliance,
            'reason': {
                'location': location_reason,
                'appliance': appliance_reason
            },
            'social_message': social_message,
            'rule_engine_version': '1.0',
            'timestamp': current_time
        }
        
        logger.info(f"Rule Engine Decision - Student: {student_profile.get('name', 'Unknown')}, "
                   f"Location: {location}, Appliance: {appliance}, Intention: {intention}")
        
        return decision