import asyncio
import json
import random
from datetime import datetime
from typing import Dict, Any, List
import logging
from .events import TimeEvent, EnergyConsumptionEvent
from .base_agent import LLMAgent
from .model_router import ModelRouter

logger = logging.getLogger(__name__)

class StudentAgent(LLMAgent):
    
    def __init__(self, agent_id: str, profile_data: Dict[str, Any], model_config: Dict[str, Any], 
                 architecture_type: str = "hybrid", dl_model_path: str = None, 
                 confidence_threshold: float = 0.8, direct_confidence_threshold: float = 0.6):
        # Build system prompt
        sys_prompt = self._build_system_prompt(profile_data)
        
        # Call parent class constructor
        super().__init__(agent_id, profile_data.get('name', 'Unknown'), sys_prompt, model_config)
        
        self.profile_data = profile_data
        self.occupation = profile_data.get('occupation', 'undergraduate')
        self.occupational_category = profile_data.get('occupational_category', 'engineering')  # New: occupational category
        self.age = profile_data.get('age', 20)
        self.work_intensity = profile_data.get('work_intensity', 'medium')
        self.energy_demand = profile_data.get('energy_demand', 'normal')
        self.thermal_preference = profile_data.get('thermal_preference', 'normal')
        
        # Architecture type: 'pure_llm', 'hybrid', 'pure_dl', 'pure_rule', 'hybrid_direct', 'pure_direct_dl'
        self.architecture_type = architecture_type
        
        # Student attributes
        self.current_location = "other"
        self.energy_intention = "weak"
        self.current_appliance = "none"
        self.total_consumption = 0.0
        
        # New: state tracking for DL models
        self.previous_location = "other"  # Previous location
        self.previous_appliance = "none"  # Previous device
        
        # Model router (selected based on architecture type)
        self.model_router = None
        if architecture_type in ["hybrid", "pure_dl", "pure_llm"]:
            self.model_router = ModelRouter(dl_model_path, confidence_threshold)
        elif architecture_type in ["hybrid_direct", "pure_direct_dl"]:
            # Use direct prediction model router (adapted for new no-intention cascade model)
            from .direct_model_router import DirectModelRouter
            # Direct prediction model now uses new no-intention cascade model path
            direct_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dl_model_no_intention', 'cascade_energy_model_no_intention.pth')
            self.model_router = DirectModelRouter(direct_model_path, direct_confidence_threshold)
        
        logger.info(f"Student agent {self.name} creation completed - Occupation: {self.occupation}, Category: {self.occupational_category}, Age: {self.age}, Work intensity: {self.work_intensity}, Energy demand: {self.energy_demand}, Thermal preference: {self.thermal_preference}, Architecture: {architecture_type}")
    
    def _build_system_prompt(self, profile_data: Dict[str, Any]) -> str:
        """Build system prompt"""
        return """
        Reply ONLY with a JSON object under an "action" key:
        {
          "action": {
            "location": "dormitory|classroom|library|laboratory|canteen|other",
            "intention": "strong|moderate|weak|none",
            "appliance": "laptop|light|phone_charger|air_conditioner|experimental_equipment|none"
          }
        }
        No explanation, no extra text.
        """
    
    def _build_llm_input(self, time_event) -> str:
        """Build LLM input"""
        env_context = time_event.environmental_context
        
        feedback_section = ""
        # Check if feedback_plan exists
        if 'feedback_plan' in env_context and env_context['feedback_plan']:
            feedback_data = env_context['feedback_plan']
            
            # 1. Determine student category (undergraduate / graduate)
            # Note: self.occupation may be "undergraduate" or "graduate student"
            category_key = "graduate" if "graduate" in self.occupation and "undergraduate" not in self.occupation else "undergraduate"
            if self.occupation == "graduate student":
                 category_key = "graduate"
            
            # 2. Try to get corresponding plan from feedback_data
            if isinstance(feedback_data, dict) and category_key in feedback_data:
                category_plan = feedback_data[category_key]
                
                # 3. Get detailed dimension
                dimension = category_plan.get('dimension')
                suggestions = category_plan.get('suggestions', {})
                
                # 4. Get student's value on this dimension
                student_dim_value = "unknown"
                if dimension == "energy_demand":
                    student_dim_value = self.energy_demand
                elif dimension == "work_intensity":
                    student_dim_value = self.work_intensity
                elif dimension == "occupational_category":
                    student_dim_value = self.occupational_category
                
                # 5. Get specific suggestions
                if student_dim_value in suggestions:
                    suggestion = suggestions[student_dim_value]
                    location_sug = suggestion.get('location', '')
                    appliance_sug = suggestion.get('appliance', '')
                    intention_sug = suggestion.get('intention', '') # Default to moderate if not provided

                    feedback_section = f"""
        GLOBAL FEEDBACK PLAN:
        Current energy consumption deviates from reality.
        Based on your profile ({dimension}={student_dim_value}), you are STRONGLY advised to:
        - Go to Location: {location_sug}
        - Use Appliance: {appliance_sug}
        - Set Intention: {intention_sug}
        - If it's sleep time now and the above suggestions conflict with sleep, you can ignore them.
        - When sleeping, the agent must choose the dormitory and not use appliances.
        - The sleep time of most agents is 0.00 to 7.00, but it should be adjusted according to the characteristics of the agents. Agents with high work intensity tend to have shorter sleep times, while those with low work intensity tend to have longer sleep times.
        """
            # Fallback for old format
            elif isinstance(feedback_data, str):
                feedback_section = f"""
        GLOBAL FEEDBACK PLAN (MUST FOLLOW):
        {feedback_data}
        """
            # Fallback for old dictionary format (with target_criteria)
            elif isinstance(feedback_data, dict) and 'plan' in feedback_data:
                plan_text = feedback_data.get('plan', '')
                target_criteria = feedback_data.get('target_criteria')
                
                # Check if meets target audience criteria
                is_target = True
                if target_criteria:
                    if target_criteria.get('occupation') and self.occupation not in target_criteria['occupation']:
                        is_target = False
                    elif target_criteria.get('occupational_category') and self.occupational_category not in target_criteria['occupational_category']:
                        is_target = False
                    elif target_criteria.get('work_intensity') and self.work_intensity not in target_criteria['work_intensity']:
                        is_target = False
                    elif target_criteria.get('energy_demand') and self.energy_demand not in target_criteria['energy_demand']:
                        is_target = False
                    elif target_criteria.get('thermal_preference') and self.thermal_preference not in target_criteria['thermal_preference']:
                        is_target = False
                
                if is_target and plan_text:
                    feedback_section = f"""
        GLOBAL FEEDBACK PLAN (MUST FOLLOW, MOST IMPORTANT):
        {plan_text}
        """
        
        # Build final prompt
        final_prompt = f"""
        Please act as an assistant to make current behavior decisions for the agent based on the agent's settings.
        Please give priority to the content of the GLOBAL FEEDBACK PLAN (if it exists) and make decisions in combination with the suggestions therein.
        Current time: {env_context['time']} (24-hour format)
        Environmental condition: {env_context['condition']}
        {feedback_section}
        
        Agent characteristics:
        - Name: {self.name}
        - Occupation: {self.occupation}
        - Occupational Category: {self.occupational_category}
        - Age: {self.age}
        - Work intensity: {self.work_intensity}
        - Energy demand: {self.energy_demand}
        - Thermal preference: {self.thermal_preference}
        
        Current Status (from last step):
        - Current Location: {self.previous_location}
        - Previous Appliance Usage: {self.previous_appliance}
        
        Make a three-step decision:
        1. Choose location (dormitory/classroom/library/laboratory/canteen/other).
        2. Determine energy intention (strong/moderate/weak/none). It reflects the degree of demand for electrical energy and appliance usage.
        3. Select appliance (choose from laptop/light/phone_charger/air_conditioner/experimental_equipment or 'none').

        Make decisions based on the following content and analyze them in sequence, consider whether need to sleep and have meals first:
        1.  [Most Important] If the agent should be sleeping at the current time, please set the agent's location to the dormitory and do not use appliance either. The duration of sleep is related to the intensity of work.
        2.  [Most Important] If it is mealtime (breakfast, lunch, dinner), the agent must go to the canteen and does not need to use the appliances.
        3.  If it is a graduate student, in the work and learning time, may need to use the experimental_equipment or laptop in the laboratory.
        4.  If it is a undergraduate student, in the work and learning time, they are usually in the classroom or the library. If they need to study in the evening, they may tend to choose the library.
        5.  When studying or working at night or in low-light conditions, consider using a light. Agents with high energy demand may use it more frequently.
        6.  In the leisure time or evening, it might be need to use a phone_charger. Especially for those whose work intensity is low.
        7.  Agent with high work intensity tend to study more and use appliances like laptop and  experimental_equipment more frequently, while those with low work intensity use them less.
        8.  Agent with both a cool thermal preference and a high energy demand may use air conditioners more frequently when it is hot. 
        9.  If the previous appliance used has met the agent's requirements, other appliances can be used.
        10.  Different occupational categories have different behavior patterns:
            - Engineering: High probability of using experimental equipment in laboratory.
            - Science: High probability of using experimental equipment and laptops in laboratory or library.
            - Arts: High probability of using laptops or lights in studios (other) or classrooms.
            - Business: High probability of using laptops in library or classrooms for case studies.
            - Medicine: High probability of long hours in laboratory or library, using lights and laptops.
        11.  The selection of appliance is not mandatory. If there is no need to use it, none can be chosen.
        12.  If it is clear that the appliance is not currently in use or cannot be used, set the intention to none.

        Provide agent decision in JSON format within an "action" field:
        {{
            "action": {{
                "location": "agent_location_choice",
                "intention": "agent_intensity_choice", 
                "appliance": "agent_appliance_choice"
            }}
        }}
        """
        
        return final_prompt
    
    def _parse_llm_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Extract decision information from LLM response
            if isinstance(llm_response, dict):
                # Directly extract action part from response
                if 'action' in llm_response:
                    action_data = llm_response.get('action', {})
                else:
                    # If response itself is action data
                    action_data = llm_response
                
                # Extract location, intention and appliance information
                location = action_data.get('location', 'other').lower()
                intention = action_data.get('intention', 'weak').lower()
                appliance = action_data.get('appliance', 'none').lower()
            else:
                # If not dict, use default values
                location = 'other'
                intention = 'weak'
                appliance = 'none'
            
            # Ensure values are within valid range
            valid_locations = ['dormitory', 'classroom', 'library', 'laboratory', 'canteen', 'other']
            valid_intentions = ['strong', 'moderate', 'weak', 'none']
            valid_appliances = ['laptop', 'light', 'phone_charger', 'air_conditioner', 'experimental_equipment', 'none']
            
            if location not in valid_locations:
                location = 'other'
            if intention not in valid_intentions:
                intention = 'weak'
            if appliance not in valid_appliances:
                appliance = 'none'
            
            return {
                'location': location,
                'intention': intention,
                'appliance': appliance
            }
            
        except Exception as e:
            logger.warning(f"LLM response parsing failed, using default decision: {e}")
            return {
                'location': 'other',
                'intention': 'weak',
                'appliance': 'none'
            }
    
    async def initialize(self):
        """Initialize student agent"""
        logger.info(f"Student agent {self.agent_id} initializing...")
        # Specific initialization logic can be added here
        logger.info(f"Student agent {self.agent_id} initialization completed")
    
    def _calculate_energy_consumption(self, appliance: str, usage_hours: float, intention: str = "weak") -> float:
        """Calculate energy consumption (only calculates appliance consumption, excludes base energy consumption)"""
        # Appliance power consumption (kWh)
        power_ratings = {
            "laptop": 0.1,      # 100W
            "phone_charger": 0.05,       # 50W
            "light": 0.015,       # 15W
            "air_conditioner": 1.0,  # 1000W - Air conditioner power
            "experimental_equipment": 2.5,  # 2500W - Experimental equipment has higher power
            "none": 0.0
        }
        
        base_power = power_ratings.get(appliance, 0.05)
        
        # Adjust power based on intention intensity
        intention_multipliers = {
            "weak": 0.9,      # weak equivalent to *0.9 power
            "moderate": 1.0,  # moderate equivalent to *1.0 power
            "strong": 1.1,    # strong equivalent to *1.1 power
            "none": 1.0       # When selecting none, generally no appliance is used, but if used, calculate normally
        }
        
        intention_multiplier = intention_multipliers.get(intention, 1.0)
        
        # Calculate appliance consumption = base power × usage time × intention intensity
        consumption = base_power * usage_hours * intention_multiplier
        
        return round(consumption, 4)
    
    async def handle_time_event(self, event: TimeEvent) -> EnergyConsumptionEvent:
        """Handle time event"""
        current_time = event.current_time
        environmental_context = event.environmental_context
        
        logger.info(f"Student {self.name} processing time event: {current_time}")
        
        # Select decision method based on architecture type
        decision = None
        model_type = "llm"  # Default to LLM
        confidence = 0.0
        
        if self.architecture_type == "pure_llm":
            # Pure LLM architecture
            logger.info(f"Using pure LLM architecture for decision making")
            decision = await self._make_llm_decision(event)
            model_type = "llm"
            
        elif self.architecture_type == "pure_dl":
            # Pure deep learning architecture (cascade model)
            logger.info(f"Using pure deep learning architecture (cascade model) for decision making")
            decision = self._make_dl_decision(current_time, environmental_context)
            model_type = "dl"
            # Extract confidence from pure DL decision
            if decision and 'confidence' in decision:
                if isinstance(decision['confidence'], dict):
                    confidence = decision['confidence'].get('overall', 0.0)
                else:
                    confidence = decision['confidence']
            
        elif self.architecture_type == "hybrid":
            # Hybrid architecture - using cascade model router
            logger.info(f"Using hybrid architecture (cascade model) for decision making")
            decision, model_type = await self._make_hybrid_decision(current_time, environmental_context, event)
            # Extract confidence from hybrid decision
            if decision and 'confidence' in decision:
                if isinstance(decision['confidence'], dict):
                    confidence = decision['confidence'].get('overall', 0.0)
                else:
                    confidence = decision['confidence']
        
        elif self.architecture_type == "hybrid_direct":
            # Hybrid architecture - using direct prediction model router
            logger.info(f"Using hybrid architecture (direct prediction model) for decision making")
            decision, model_type = await self._make_hybrid_decision(current_time, environmental_context, event)
            # Extract confidence from hybrid decision
            if decision and 'confidence' in decision:
                if isinstance(decision['confidence'], dict):
                    confidence = decision['confidence'].get('overall', 0.0)
                else:
                    confidence = decision['confidence']
            
        elif self.architecture_type == "pure_direct_dl":
            # Pure direct prediction deep learning architecture
            logger.info(f"Using pure direct prediction deep learning architecture for decision making")
            decision, model_type = await self._make_hybrid_decision(current_time, environmental_context, event)
            # Extract confidence from direct prediction model decision
            if decision and 'confidence' in decision:
                if isinstance(decision['confidence'], dict):
                    confidence = decision['confidence'].get('overall', 0.0)
                else:
                    confidence = decision['confidence']
            
        else:
            # Default to LLM
            logger.warning(f"Unknown architecture type {self.architecture_type}, using LLM")
            decision = await self._make_llm_decision(event)
            model_type = "llm"
        
        # Extract decision information
        if decision:
            location = decision.get("location", "other")
            intention = decision.get("intention", "weak")
            appliance = decision.get("appliance", "none")
        else:
            # Fallback handling
            logger.error("Decision failed, using default decision")
            location = "other"
            intention = "weak"
            appliance = "none"
        
        # Usage duration equals time step
        usage_hours = event.time_step if appliance != "none" else 0
        
        # Calculate energy consumption
        consumption = self._calculate_energy_consumption(appliance, usage_hours, intention)
        
        # Update state - first save previous state, then update current state
        self.previous_location = self.current_location
        self.previous_appliance = self.current_appliance
        self.current_location = location
        self.current_appliance = appliance
        self.total_consumption += consumption
        
        logger.info(f"Student {self.name} decision: {location} uses {appliance} {usage_hours:.1f}hours, "
                   f"consumes {consumption:.3f} kWh, model type: {model_type}")
        
        # Determine student type
        student_type = "graduate" if "graduate" in self.occupation.lower() and "undergraduate" not in self.occupation.lower() else "undergraduate"
        
        # Return energy consumption event
        return EnergyConsumptionEvent(
            from_agent_id=self.agent_id,
            to_agent_id="environment",
            student_id=self.agent_id,
            consumption=consumption,
            location=location,
            intention=intention,
            appliance=appliance,
            scenario="normal",
            timestamp=current_time,
            model_type=model_type,
            confidence=confidence if model_type in ["dl", "direct_dl"] else 0.0,
            thermal_preference=self.thermal_preference,
            occupational_category=self.occupational_category,
            previous_location=self.previous_location,
            previous_appliance=self.previous_appliance,
            student_type=student_type
        )
    
    async def _make_llm_decision(self, event: TimeEvent) -> Dict[str, Any]:
        """Make decision using LLM"""
        try:
            # Build LLM input
            llm_input = self._build_llm_input(event)
            
            # Get LLM response
            llm_response = await self.generate_response(llm_input)
            logger.info(f"LLM response: {llm_response}")
            
            # Parse LLM response
            decision = self._parse_llm_response(llm_response)
            return decision
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return {
                'location': 'other',
                'intention': 'weak',
                'appliance': 'none'
            }
    
    def _make_dl_decision(self, current_time: str, environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using deep learning model"""
        if not self.model_router:
            logger.error("Model router not initialized")
            return {
                'location': 'other',
                'intention': 'weak',
                'appliance': 'none'
            }
        
        try:
            # Build student profile including environmental conditions and student state
            profile_with_env = self.profile_data.copy()
            profile_with_env['environmental_condition'] = environmental_context.get('condition', 'normal')
            profile_with_env['environmental_conditions'] = environmental_context.get('condition', 'warm')  # For DL model
            profile_with_env['previous_location'] = self.previous_location
            profile_with_env['previous_appliance'] = self.previous_appliance
            
            # Use deep learning model directly
            features = self.model_router.encode_input(current_time, profile_with_env)
            if features is not None:
                dl_result = self.model_router.predict_with_confidence(features)
                if dl_result:
                    decision, confidence = dl_result
                    decision['confidence'] = confidence
                    return decision
                    
            logger.error("Deep learning model prediction failed")
            return {
                'location': 'other',
                'intention': 'weak',
                'appliance': 'none'
            }
            
        except Exception as e:
            logger.error(f"Deep learning decision failed: {e}")
            return {
                'location': 'other',
                'intention': 'weak',
                'appliance': 'none'
            }
    
    async def _make_hybrid_decision(self, current_time: str, environmental_context: Dict[str, Any], event: TimeEvent) -> tuple:
        """Make decision using hybrid architecture"""
        if not self.model_router:
            logger.error("Model router not initialized")
            # Fallback to LLM
            decision = await self._make_llm_decision(event)
            return decision, "llm"
        
        try:
            # Build student profile including environmental conditions and student state
            profile_with_env = self.profile_data.copy()
            profile_with_env['environmental_condition'] = environmental_context.get('condition', 'normal')
            profile_with_env['environmental_conditions'] = environmental_context.get('condition', 'warm')  # For DL model
            profile_with_env['previous_location'] = self.previous_location
            profile_with_env['previous_appliance'] = self.previous_appliance
            
            # Define LLM function - supports anomaly detection prompts
            async def llm_function(prompt=None):
                if prompt and "is_anomaly" in prompt and "Environmental condition" in prompt:
                    # Anomaly detection mode - directly analyze environmental conditions
                    logger.info(f"LLM function entering anomaly detection mode - analyzing prompt")
                    try:
                        # Build LLM input specifically for anomaly detection
                        anomaly_input = f"""
                        {prompt}
                        
                        Please respond ONLY with the JSON format requested above.
                        """
                        
                        # Get LLM response
                        llm_response = await self.generate_response(anomaly_input)
                        logger.info(f"Anomaly detection LLM response: {llm_response}")
                        
                        # Try to directly return LLM response (should be JSON format)
                        return llm_response
                        
                    except Exception as e:
                        logger.error(f"Anomaly detection LLM call failed: {e}")
                        # Fallback to simple keyword-based detection
                        env_condition = environmental_context.get('condition', '')
                        is_anomaly = any(keyword in env_condition.lower() for keyword in 
                                         ['typhoon', 'power outage', 'extreme', 'failure', 'broken', 'storm'])
                        return {
                            "is_anomaly": is_anomaly,
                            "description": "Anomaly detected by keywords" if is_anomaly else "Normal"
                        }
                else:
                    # Normal decision mode
                    logger.info(f"LLM function entering normal decision mode")
                    return await self._make_llm_decision(event)
            
            # Use model router for intelligent routing
            decision, model_type = await self.model_router.route(current_time, profile_with_env, llm_function)
            return decision, model_type
            
        except Exception as e:
            logger.error(f"Hybrid decision failed: {e}")
            # Fallback to LLM
            decision = await self._make_llm_decision(event)
            return decision, "llm"
