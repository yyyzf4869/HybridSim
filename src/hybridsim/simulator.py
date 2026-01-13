import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

from .events import TimeEvent, Event
from .student_agent import StudentAgent
from .env_agent import EnvAgent
from .base_agent import LLMAgent
from .token_usage import export_token_usage_stats, log_token_usage

logger = logging.getLogger(__name__)

class EnergyConsumptionSimulator:
    """Main campus energy consumption simulator class"""
    
    def __init__(self, config_path: str = None, llm_workers: int = 30, dl_workers: int = 100,
                 architecture_type: str = "hybrid", dl_model_path: str = None, 
                 confidence_threshold: float = 0.8, direct_confidence_threshold: float = 0.6,
                 time_step: float = 1.0, enable_feedback: bool = False, 
                 environmental_condition: str = "normal"):
        self.model_config = self._load_model_config()
        self.environmental_condition = environmental_condition
        
        # Architecture configuration
        self.architecture_type = architecture_type  # 'pure_llm', 'hybrid', 'pure_dl', 'pure_rule'
        self.dl_model_path = dl_model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dl_model_no_intention', 'cascade_energy_model_no_intention.pth')
        self.confidence_threshold = confidence_threshold
        self.direct_confidence_threshold = direct_confidence_threshold
        
        # Feedback configuration
        self.enable_feedback = enable_feedback
        self.feedback_data = self._load_feedback_data() if enable_feedback else {}
        self.real_behavior_data = self._load_real_behavior_data() if enable_feedback else []
        self.simulated_behavior_data = self._load_simulated_behavior_data() if enable_feedback else []
        self.current_feedback_plan = ""
        self.feedback_plan_history = [] 
        
        # Agents
        self.environment_agent = None
        self.student_agents = []
        
        # Simulation state
        self.current_time = None
        self.time_step = time_step  # hours
        self.simulation_running = False
        self.simulation_step = 0
        self.max_steps = 24  # Default 24 hours
        
        # Parallel processing configuration
        # Select thread pool size based on architecture type
        if self.architecture_type in ["pure_dl", "pure_direct_dl"]:
            # Pure deep learning architecture (cascade or direct prediction) uses larger thread pool
            self.max_workers = dl_workers
            logger.info(f"Architecture is {self.architecture_type}, using DL dedicated thread pool size: {self.max_workers}")
        elif self.architecture_type == "pure_rule":
            self.max_workers = dl_workers  # Rule-driven uses larger thread pool
            logger.info(f"Architecture is pure_rule, using rule engine dedicated thread pool size: {self.max_workers}")
        else:
            # In pure_llm, hybrid, hybrid_direct modes, use smaller thread pool to prevent LLM API overload
            # Even though hybrid/hybrid_direct modes mostly use DL, falling back to LLM with high concurrency may cause API limits
            self.max_workers = llm_workers
            logger.info(f"Architecture is {self.architecture_type}, using LLM safe thread pool size: {self.max_workers}")
            
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Campus energy consumption simulator initialization completed - parallel worker threads: {self.max_workers}, architecture type: {architecture_type}")
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config', 'model_config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Model configuration file not found, using default configuration")
            return self._get_default_model_config()
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Default model configuration"""
        return {
            "chat": [{
                "provider": "openai",
                "config_name": "Qwen2.5-14B-Instruct",
                "model_name": "Qwen2.5-14B-Instruct",
                "api_key": "sk-xxx",
                "client_args": {"base_url": "http://localhost:9891/v1/", "timeout": 30, "max_retries": 3},
                "generate_args": {"temperature": 0.15}
            }]
        }

    def _load_feedback_data(self) -> Dict[str, Dict[str, float]]:
        """Load feedback data"""
        feedback_data = {}
        try:
            import csv
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback', 'feedback_data.csv'), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    time_key = row['Time']
                    feedback_data[time_key] = {
                        'real': float(row['Normalized_Real']),
                        'llm': float(row['Normalized_LLM'])
                    }
            logger.info(f"Loaded {len(feedback_data)} feedback data entries")
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
        return feedback_data

    def _load_real_behavior_data(self) -> List[Dict[str, Any]]:
        """Load real behavior data"""
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback', 'real_appliance.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load real behavior data: {e}")
            return []

    def _load_simulated_behavior_data(self) -> List[Dict[str, Any]]:
        """Load simulated behavior data"""
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback', 'simulated_behavior_15min.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load simulated behavior data: {e}")
            return []

    def _is_time_in_range(self, current_time: str, start_time: str, end_time: str) -> bool:
        """Check if time is within range (HH:MM)"""
        try:
            cur_h, cur_m = map(int, current_time.split(':'))
            start_h, start_m = map(int, start_time.split(':'))
            end_h, end_m = map(int, end_time.split(':'))
            
            cur_minutes = cur_h * 60 + cur_m
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            
            # Handle cross-day scenarios (e.g., 23:00 - 01:00)
            if end_minutes < start_minutes:
                return cur_minutes >= start_minutes or cur_minutes < end_minutes
            
            return start_minutes <= cur_minutes < end_minutes
        except Exception:
            return False

    async def _generate_global_feedback_plan(self, current_time_str: str) -> Dict[str, Any]:
        """Generate global Feedback planning"""
        if not self.enable_feedback:
            return None
            
        # Extract time part (e.g., "Monday 09:00" -> "09:00")
        _, time_part = current_time_str.rsplit(' ', 1)
        
        # Standardize time format, ensure it's HH:MM
        try:
            h, m = map(int, time_part.split(':'))
            time_part = f"{h:02d}:{m:02d}"
        except ValueError:
            pass
            
        # Find closest time point data
        target_data = self.feedback_data.get(time_part)
        if not target_data:
            # Try fuzzy matching
            try:
                hour, minute = map(int, time_part.split(':'))
                # Round to nearest 15 minutes
                minute_rounded = round(minute / 15) * 15
                if minute_rounded == 60:
                    hour += 1
                    minute_rounded = 0
                    if hour == 24:
                        hour = 0
                rounded_time = f"{hour:02d}:{minute_rounded:02d}"
                target_data = self.feedback_data.get(rounded_time)
            except Exception:
                pass
        
        if not target_data:
            logger.warning(f"Feedback data for time point {time_part} not found")
            return None
            
        real_val = target_data['real']
        llm_val = target_data['llm']
        diff = real_val - llm_val
            
        # Get real behavior data and simulated behavior data for the current time slot
        current_behavior = None
        for slot in self.real_behavior_data:
            time_range = slot['time_slot']
            try:
                start_str, end_str = time_range.split(' - ')
                # Complete format to HH:MM
                if ':' in start_str:
                    sh, sm = map(int, start_str.split(':'))
                    start_str = f"{sh:02d}:{sm:02d}"
                if ':' in end_str:
                    eh, em = map(int, end_str.split(':'))
                    end_str = f"{eh:02d}:{em:02d}"
                    
                if self._is_time_in_range(time_part, start_str, end_str):
                    current_behavior = slot
                    break
            except Exception as e:
                logger.warning(f"Failed to parse real behavior time slot {time_range}: {e}")
                continue
        
        # Get simulated behavior data (exact match time_part, as we generate 15min granularity)
        current_sim_behavior = None
        for slot in self.simulated_behavior_data:
            # Compatible with both new and old data formats
            if 'time_slot' in slot:
                # New format: time_slot field (e.g., "00:00")
                if slot['time_slot'] == time_part:
                    current_sim_behavior = slot
                    break
            elif 'timestamp' in slot:
                # Old format: timestamp field (e.g., "Monday 00:00")
                if time_part in slot['timestamp']:
                    current_sim_behavior = slot
                    break
        
        if not current_behavior:
            logger.warning(f"Real behavior data for time point {time_part} not found, generating feedback based on energy consumption data and general rules")
            # Continue even if no behavior data, do not return None
        
        # Build behavior pattern context
        behavior_context = ""
        behavior_requirements = ""
        
        if current_behavior:
            behavior_context = f"""
        Real World Behavior Patterns (Current Time Slot: {current_behavior['time_slot']}):
        {json.dumps(current_behavior, indent=2)}
            """
            
            if current_sim_behavior:
                behavior_context += f"""
        Previous Simulation Behavior Patterns (Current Time Slot: {time_part}):
        {json.dumps(current_sim_behavior, indent=2)}
                """
            
            behavior_requirements = """
           - The real behavioral data is the complete data within one hour (for example, 7:00-8:00), spanning multiple time steps.
           - You need to set suggestions for different types of agents based on the real probability distribution. For instance, locations and appliances with the largest probability distribution should be recommended to more agents.
           - Content with a smaller probability distribution should be recommended to a small number of agents.
           - Make sure that the content of the "Real World Behavior Patterns" top5 categories is assigned to the corresponding agent (Undergraduate and graduate students are separated), unless the proportion of these categories is very small, such as less than 5%.
           - Compare "Real World Behavior Patterns" with "Previous Simulation Behavior Patterns" to identify specific behavioral discrepancies (e.g., Real World has 50% in Laboratory but Simulation only has 10%). Target your suggestions to fix these specific discrepancies.
            """
        else:
            behavior_context = """
        Real World Behavior Patterns:
        No specific behavioral data is available for this time slot (likely late night 00:00 - 07:00).
            """
            behavior_requirements = """
           - Currently, there is no behavioral data for this moment. Please provide feedback based on energy consumption data.
           - Consider that it is likely late at night (e.g., 00:00-07:00), so agents must be in the dormitory, you can't suggest going anywhere else.
           - The appliances available in the dormitory include light, air_conditioner phone_charger and none.
           - No other appliances can be used in the dormitory, and when the agent is in sleep mode, the appliance is not allowed to be used, that is, none.
            """

        # Build LLM prompt
        prompt = f"""
        Current Simulation Time: {current_time_str}
        
        Energy Consumption Data Analysis:
        - Real World Normalized Consumption: {real_val:.4f}
        - Current Simulation Baseline (LLM) Consumption: {llm_val:.4f}
        - Difference (Real - Simulation): {diff:.4f}
        
        {behavior_context}
        
        Task:
        Generate a "Global Feedback Plan" to adjust student agents' behavior based on the difference and real-world patterns.
        
        Requirements:
        1. Threshold Check: The difference is {diff:.4f}. If the absolute value of diff is greater than 0.1, more attention should be paid to the guidance at the energy consumption alignment level.
        2. Categorization: Provide suggestions separately for "undergraduate" and "graduate" students.
        3. Sub-categorization: For each category, choose ONE dimension (energy_demand, work_intensity, or occupational_category) to further differentiate suggestions.
           - Decide which dimension to use based on the energy consumption difference. For example, if we need much higher consumption, targeting 'high energy_demand' students might be effective.
        4. Reference Real Data: Use the provided "Real World Behavior Patterns" to guide your suggestions for Location and Appliance.
        {behavior_requirements}
        5. Output Content: For each sub-group, provide a recommended "location", "appliance" and "intention".
        6. Location include dormitory, canteen, laboratory, library, classroom, other
        7. Appliance include light, phone_charger, laptop, experimental_equipment, air_conditioner, none(If the equipment is not needed)
        8. Intention include strong, moderate, weak, none. "strong" means high energy consumption intensity (e.g. running heavy tasks), "weak" means low intensity.

        Guidance based on Difference ({diff:.4f}):
        - If Difference > 0 (Real > Sim): We need MORE energy consumption. Suggest behaviors/appliances that consume more energy (e.g., air_conditioner, experimental_equipment) and use "strong" or "moderate" intention.
        - If Difference < 0 (Real < Sim): We need LESS energy consumption. Suggest energy-saving behaviors and use "weak" or "none" intention.
        - When setting plans for agents with different characteristics, the distribution of real data needs to be taken into consideration.
        
        Example Plan:
        - "Based on the current time and the agent's Settings, it is recommended that the agent choose the dormitory and use low-power appliances, such as light and phone_charger. And and set intention to weak."
        - "Based on the current time and the agent's Settings, it is recommended that the agent choose the canteen and not use appliances. And and set intention to none."
        - "Based on the current time and the agent's Settings, it is recommended that the agent choose the laboratory and use high-power appliances, such as experimental_equipment or air_conditioner. And and set intention to strong."
        - "Based on the current time and the agent's Settings, it is recommended that the agent choose the library and use medium-power appliances, such as laptop. And and set intention to moderate."
        
        Important Rules:
        - If there is more than one content to choose from, please provide multiple options, use "or" instead of "->".
        - If you want to increase energy consumption, do not set the same suggestions for agents of multiple categories, as this will cause sudden changes in energy consumption. Please set it up for the type of agent that is most needed.
        
        Output Format:
        Reply ONLY with a JSON object in the following structure:
        {{
            "undergraduate": {{
                "dimension": "energy_demand",  // Choose from: energy_demand, work_intensity, occupational_category, not only can energy_demand be selected. You can choose other features.
                "suggestions": {{
                    "high": {{"..."}},
                    "medium": {{"..."}},
                    "low": {{"..."}}
                }}
            }},
            "graduate": {{
                "dimension": "occupational_category", // Choose from: energy_demand, work_intensity, occupational_category, not only can occupational_category be selected. You can choose other features.
                "suggestions": {{
                    "engineering": {{"..."}},
                    "science": {{"..."}},
                    "arts": {{"..."}},
                    "business": {{"..."}},
                    "medicine": {{"..."}}
                }}
            }}
        }}
        
        Note: The keys in "suggestions" must match the possible values of the chosen "dimension".
        - energy_demand: high, medium, low
        - work_intensity: high, medium, low
        - occupational_category: engineering, science, arts, business, medicine
        - You should use a variety of characteristics to make recommendations, not just in accordance with the above examples. Undergraduate and graduate students should have different characteristics.
        """
        try:
            # Use temporary LLM Agent to call model
            temp_agent = LLMAgent("planner", "planner", "", self.model_config)
            response = await temp_agent.generate_response(prompt)

            plan_data = {}
            if isinstance(response, dict):
                plan_data = response
            else:
                try:
                    # Try to extract JSON from text
                    content = str(response)
                    # Find first { and last }
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1:
                        json_str = content[start:end+1]
                        plan_data = json.loads(json_str)
                    else:
                        logger.error(f"Cannot parse LLM response as JSON: {content[:100]}...")
                        return None
                except:
                    logger.error("JSON parsing exception")
                    return None
            
            # Record to history
            self.feedback_plan_history.append({
                "time": current_time_str,
                "plan": plan_data
            })
            
            return plan_data
            
        except Exception as e:
            logger.error(f"Failed to generate Global Feedback Plan: {e}")
            return None

    def get_current_environmental_context(self, time_str: str) -> Dict[str, Any]:
        """Get current environmental context"""
        # Parse time string "Monday 9:00" format
        weekday_str, time_part = time_str.rsplit(' ', 1)
        current_hour = int(time_part.split(':')[0])
        
        # Convert weekday string to number (0=Monday, 6=Sunday)
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        try:
            day_of_week = weekdays.index(weekday_str)
        except ValueError:
            day_of_week = datetime.now().weekday()  # Default to current weekday
        
        # Use environmental condition parameter
        condition = self.environmental_condition
        
        return {
            'time': time_str,
            'condition': condition,
            'day_of_week': day_of_week
        }
    
    def load_student_profiles(self, profiles_path: str = None) -> List[Dict[str, Any]]:
        """Load student profiles"""
        if profiles_path is None:
            profiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'profile', 'data', 'agents_100.json')
        
        try:
            with open(profiles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Student profiles file {profiles_path} not found, using default profiles")
            return self._get_default_student_profiles()
    
    def _get_default_student_profiles(self) -> List[Dict[str, Any]]:
        """Default student profiles"""
        return [
            {
                "name": "Alice",
                "occupation": "undergraduate",
                "occupational_category": "arts",
                "age": 20,
                "work_intensity": "medium",
                "energy_demand": "normal",
                "id": "NO.00001"
            },
            {
                "name": "Bob",
                "occupation": "graduate student",
                "occupational_category": "engineering",
                "age": 25,
                "work_intensity": "high",
                "energy_demand": "low",
                "id": "NO.00002"
            }
        ]
    
    async def initialize_simulation(self, start_time: str = "08:00", max_steps: int = 24, agent_count: int = 100):
        """Initialize simulation"""
        logger.info("Starting simulation environment initialization...")
        
        # Set simulation parameters
        self.current_time = start_time
        self.max_steps = max_steps
        self.simulation_step = 0
        
        # Select appropriate configuration based on agent_count
        if agent_count <= 100:
            profiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'profile', 'data', 'agents_100.json')
        elif agent_count <= 10000:
            profiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'profile', 'data', 'agents_10000.json')
        else:
            logger.warning(f"Unsupported agent count: {agent_count}, maximum supported is 10000, using 10000 agents")
            profiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'profile', 'data', 'agents_10000.json')
            agent_count = 10000
        
        # Load student profiles and create student agents
        student_profiles = self.load_student_profiles(profiles_path)
        
        # Slice agent list based on requested count
        if len(student_profiles) > agent_count:
            student_profiles = student_profiles[:agent_count]
            logger.info(f"Taking first {agent_count} agent configurations")
        
        total_agents = len(student_profiles)
        
        # Create environment agent, passing agent count, time step, and environmental condition
        self.environment_agent = EnvAgent(total_agents=total_agents, time_step=self.time_step, environmental_condition=self.environmental_condition)
        
        for profile in student_profiles:
            agent_id = profile.get('id', profile['name'])
            
            if self.architecture_type == "pure_rule":
                # Rule-driven architecture uses specialized agent class
                from .rule_based_student_agent import RuleBasedStudentAgent
                student_agent = RuleBasedStudentAgent(agent_id, profile)
            else:
                # Other architectures use existing StudentAgent
                student_agent = StudentAgent(
                    agent_id, 
                    profile, 
                    self.model_config,
                    architecture_type=self.architecture_type,
                    dl_model_path=self.dl_model_path,
                    confidence_threshold=self.confidence_threshold,
                    direct_confidence_threshold=self.direct_confidence_threshold
                )
            
            self.student_agents.append(student_agent)
        
        logger.info(f"Simulation initialization completed - Student count: {total_agents}, Max steps: {max_steps}, Architecture type: {self.architecture_type}")
    
    async def run_simulation_step(self):
        """Run a simulation step (using parallel processing)"""
        if not self.simulation_running:
            return
        
        logger.info(f"=== Simulation Step {self.simulation_step + 1}/{self.max_steps} ===")
        logger.info(f"Current Time: {self.current_time}")
        
        # Get environmental context
        environmental_context = self.get_current_environmental_context(self.current_time)
        
        # Generate Global Feedback Plan
        if self.enable_feedback:
            self.current_feedback_plan = await self._generate_global_feedback_plan(self.current_time)
            environmental_context['feedback_plan'] = self.current_feedback_plan
        
        # Create time event for each student agent
        time_event = TimeEvent(
            from_agent_id="simulator",
            to_agent_id="all_students",
            current_time=self.current_time,
            time_step=self.time_step,
            environmental_context=environmental_context
        )
        
        # Use thread pool to process all student decisions in parallel
        logger.info(f"Using {self.max_workers} worker threads to process {len(self.student_agents)} student agents in parallel")
        
        # Define function to process single student
        def process_student_agent(student_agent):
            """Function to process a single student agent"""
            try:
                # Run asynchronous handle_time_event method in thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(student_agent.handle_time_event(time_event))
                loop.close()
                return result
            except Exception as e:
                logger.error(f"Error processing student agent {student_agent.agent_id}: {e}")
                return None
        
        # Use thread pool for parallel processing
        import time
        start_time = time.time()
        
        # Submit all tasks to thread pool
        futures = []
        for student_agent in self.student_agents:
            future = self.executor.submit(process_student_agent, student_agent)
            futures.append(future)
        
        # Collect all results
        consumption_events = []
        for future in futures:
            try:
                event = future.result(timeout=30)  # 30 seconds timeout
                if event:
                    consumption_events.append(event)
            except Exception as e:
                logger.error(f"Error getting student agent processing result: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"Parallel processing completed - Processing time: {processing_time:.3f}s, Successfully processed: {len(consumption_events)}/{len(self.student_agents)} agents")
        
        # Process energy consumption events (serial processing to avoid concurrent write issues)
        for event in consumption_events:
            if event and hasattr(event, 'consumption'):
                await self.environment_agent.record_consumption(event)
        
        # Update environment agent time
        self.environment_agent.current_time = self.current_time
        
        # Record statistics
        stats = self.environment_agent.get_consumption_stats()
        logger.info(f"Step Stats - Total Consumption: {stats['total_consumption']:.3f} kWh, "
                   f"Student Count: {stats['student_count']}, "
                   f"Average per Student: {stats['average_per_student']:.3f} kWh")
    
    async def run_simulation(self):
        """Run complete simulation"""
        logger.info("Starting campus energy consumption simulation...")
        self.simulation_running = True
        
        try:
            # Initialize environment agent
            await self.environment_agent.initialize()
            
            # Initialize all student agents
            for student_agent in self.student_agents:
                await student_agent.initialize()
            
            # Run simulation steps
            for step in range(self.max_steps):
                self.simulation_step = step
                await self.run_simulation_step()
                
                # Update time (including cross-day handling)
                weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_str, time_part = self.current_time.rsplit(' ', 1)
                current_h, current_m = map(int, time_part.split(':'))
                
                # Calculate added time (minutes)
                step_minutes = int(self.time_step * 60)
                total_minutes_today = current_h * 60 + current_m + step_minutes
                
                # Calculate new hour and minute
                new_h = (total_minutes_today // 60) % 24
                new_m = total_minutes_today % 60
                
                # Handle cross-day
                days_to_add = (total_minutes_today // 60) // 24
                if days_to_add > 0:
                    try:
                        current_day_index = weekdays.index(weekday_str)
                        next_day_index = (current_day_index + days_to_add) % 7
                        weekday_str = weekdays[next_day_index]
                    except ValueError:
                        # Keep as is if weekday parsing fails
                        pass
                
                self.current_time = f"{weekday_str} {new_h:02d}:{new_m:02d}"
                
                # Small delay to prevent running too fast
                await asyncio.sleep(0.1)
            
            logger.info("Simulation run completed")
            
            # Export data to timestamp directory
            import os
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            runs_root = os.path.join('runs', ts)
            os.makedirs(runs_root, exist_ok=True)
            
            # Export Feedback plan history as JSON
            if self.feedback_plan_history:
                feedback_export_file = os.path.join(runs_root, f'feedback_plan_{ts}.json')
                try:
                    with open(feedback_export_file, 'w', encoding='utf-8') as f:
                        json.dump(self.feedback_plan_history, f, indent=4, ensure_ascii=False)
                    logger.info(f"Global Feedback Plan history exported to: {feedback_export_file}")
                except Exception as e:
                    logger.error(f"Failed to export Feedback Plan history: {e}")
            
            export_file = self.environment_agent.export_data(output_dir=runs_root)
            logger.info(f"Simulation data exported to: {export_file}")
            
            # Export token usage statistics
            try:
                token_export_file = export_token_usage_stats(
                    filepath=os.path.join(runs_root, f'token_usage_{ts}.json')
                )
                logger.info(f"Token usage stats exported to: {token_export_file}")
                
                # Log token usage
                log_token_usage()
            except Exception as e:
                logger.warning(f"Failed to export token usage stats: {e}")

            # Plot charts: Total consumption per time step and scenario distribution
            try:
                import matplotlib.pyplot as plt
                # Prepare data
                step_keys = sorted(self.environment_agent.step_consumptions.keys())
                
                # Get base consumption data
                consumption_stats = self.environment_agent.get_consumption_stats()
                step_base_consumptions = consumption_stats.get('step_base_consumptions', {})
                
                # Calculate total consumption including base consumption
                appliance_totals = [sum(self.environment_agent.step_consumptions[k]) for k in step_keys]
                base_consumption = [step_base_consumptions.get(k, 0.0) for k in step_keys]
                totals = [a + b for a, b in zip(appliance_totals, base_consumption)]
                
                # Scenario distribution (including base consumption)
                loc_break = self.environment_agent.step_location_summary
                dorm = [loc_break.get(k, {}).get('dormitory', 0.0) for k in step_keys]
                classroom = [loc_break.get(k, {}).get('classroom', 0.0) for k in step_keys]
                library = [loc_break.get(k, {}).get('library', 0.0) for k in step_keys]
                laboratory = [loc_break.get(k, {}).get('laboratory', 0.0) for k in step_keys]
                canteen = [loc_break.get(k, {}).get('canteen', 0.0) for k in step_keys]
                other = [loc_break.get(k, {}).get('other', 0.0) for k in step_keys]
                
                # Chart 1: Total consumption per time step
                plt.figure(figsize=(12, 4))
                plt.plot(range(len(step_keys)), totals, marker='o')
                plt.title('Total Consumption per Time Step')
                plt.xlabel('Time')
                plt.ylabel('kWh')
                plt.xticks(range(len(step_keys)), step_keys, rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig1_path = os.path.join(runs_root, 'total_consumption_per_step.png')
                plt.savefig(fig1_path)
                plt.close()

                # Chart 2: Consumption by location per time step (including base consumption)
                plt.figure(figsize=(12, 6))
                x_pos = range(len(step_keys))
                
                # Calculate base consumption distribution for each location
                step_location_base = {}
                for location in ['dormitory', 'classroom', 'library', 'laboratory', 'canteen', 'other']:
                    step_location_base[location] = []
                    for timestamp in step_keys:
                        # Get base consumption for that location at that time step
                        base_consumption = 0.0
                        # Recalculate base consumption for that location at that time step
                        for record in self.environment_agent.consumption_history:
                            if record['timestamp'] == timestamp and record['location'] == location:
                                # Calculate base consumption for each location only once
                                base_consumption = self.environment_agent._calculate_base_consumption(location, timestamp)
                                break
                        step_location_base[location].append(base_consumption)
                
                # Calculate total consumption including base consumption
                dorm_total = [d + b for d, b in zip(dorm, step_location_base['dormitory'])]
                classroom_total = [c + b for c, b in zip(classroom, step_location_base['classroom'])]
                library_total = [l + b for l, b in zip(library, step_location_base['library'])]
                laboratory_total = [lab + b for lab, b in zip(laboratory, step_location_base['laboratory'])]
                canteen_total = [cafe + b for cafe, b in zip(canteen, step_location_base['canteen'])]
                other_total = [o + b for o, b in zip(other, step_location_base['other'])]
                
                # Draw line chart
                plt.plot(x_pos, dorm_total, 'o-', label='dormitory', linewidth=2, markersize=6)
                plt.plot(x_pos, classroom_total, 's-', label='classroom', linewidth=2, markersize=6)
                plt.plot(x_pos, library_total, '^-', label='library', linewidth=2, markersize=6)
                plt.plot(x_pos, laboratory_total, 'd-', label='laboratory', linewidth=2, markersize=6)
                plt.plot(x_pos, canteen_total, 'p-', label='canteen', linewidth=2, markersize=6)
                plt.plot(x_pos, other_total, '*-', label='other', linewidth=2, markersize=6)
                
                plt.title('Consumption per Time Step by Location')
                plt.xlabel('Time')
                plt.ylabel('kWh')
                plt.xticks(x_pos, step_keys, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig2_path = os.path.join(runs_root, 'consumption_per_step_by_location.png')
                plt.savefig(fig2_path)
                plt.close()
                
                # Export chart data as JSON
                chart_data = {
                    "time_steps": step_keys,
                    "total_consumption": totals,
                    "consumption_by_location": {
                        "dormitory": dorm_total,
                        "classroom": classroom_total,
                        "library": library_total,
                        "laboratory": laboratory_total,
                        "canteen": canteen_total,
                        "other": other_total
                    }
                }
                json_path = os.path.join(runs_root, 'chart_data.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(chart_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Charts exported to: {fig1_path}, {fig2_path}")
                logger.info(f"Chart data exported to: {json_path}")
            except Exception as e:
                logger.warning(f"Plotting failed: {e}")
            
            # Show final statistics
            final_stats = self.environment_agent.get_consumption_stats()
            logger.info(f"=== Final Statistics ===")
            logger.info(f"Total Energy Consumption: {final_stats['total_consumption']:.3f} kWh")
            logger.info(f"Participating Students: {final_stats['student_count']}")
            logger.info(f"Per Capita Consumption: {final_stats['average_per_student']:.3f} kWh")
            
            if final_stats['location_summary']:
                logger.info(f"Consumption by Location:")
                for location, consumption in final_stats['location_summary'].items():
                    logger.info(f"  {location}: {consumption:.3f} kWh")
            
            if final_stats['appliance_summary']:
                logger.info(f"Consumption by Appliance:")
                for appliance, consumption in final_stats['appliance_summary'].items():
                    logger.info(f"  {appliance}: {consumption:.3f} kWh")
            
        except Exception as e:
            logger.error(f"Simulation run error: {e}")
            raise
        finally:
            self.simulation_running = False
            # Cleanup thread pool
            self._cleanup_executor()
    
    def _cleanup_executor(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor') and self.executor:
            logger.info("Shutting down thread pool executor...")
            self.executor.shutdown(wait=True)
            logger.info("Thread pool executor shut down")
    
    def __del__(self):
        """Destructor, ensures resources are cleaned up"""
        self._cleanup_executor()
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        if self.environment_agent:
            return self.environment_agent.get_consumption_stats()
        return {}
