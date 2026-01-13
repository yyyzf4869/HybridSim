"""
Direct prediction model routing module - implements single-step prediction deep learning model
"""
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os
import sys

logger = logging.getLogger(__name__)

class DirectModelRouter:
    """Direct prediction model router - uses single-step prediction architecture"""
    
    # Class variable: shared model instances
    _shared_models = {}
    
    def __init__(self, dl_model_path: str = None, confidence_threshold: float = 0.6):
        """
        Initialize direct prediction model router
        
        Args:
            dl_model_path: Direct prediction model path
            confidence_threshold: Confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.dl_model = None
        self.scaler = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Anomaly mapping dictionary - stores Environmental condition to (preset environment, intention, is_anomaly) mapping
        # Structure: {environmental_condition: (preset environment, intention, is_anomaly)}
        self.anomaly_intention_map = {}
        
        # Load direct prediction model
        if dl_model_path and os.path.exists(dl_model_path):
            self._load_direct_model_shared(dl_model_path)
        else:
            logger.warning(f"Direct prediction model path does not exist: {dl_model_path}")
        
        # Encoding mappings
        self._init_encodings()
        
        logger.info(f"Direct prediction model router initialized - confidence threshold: {confidence_threshold}")
    
    def _load_direct_model(self, model_path: str):
        """Load direct prediction model (using cascade model without intention prediction)"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Import new model class - uses cascade model without intention prediction
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dl_model'))
            from cascade_energy_model_no_intention import CascadeEnergyModelNoIntention
            
            # Create model - adapt to new model configuration (remove intention part)
            config = checkpoint.get('model_config', {'input_dim': 11, 'num_locations': 6, 'num_appliances': 6})
            self.model_config = config
            
            self.dl_model = CascadeEnergyModelNoIntention(
                input_dim=config['input_dim'],
                num_locations=config['num_locations'],
                num_appliances=config['num_appliances']
            ).to(self.device)
            
            self.dl_model.load_state_dict(checkpoint['model_state_dict'])
            self.dl_model.eval()
            
            # Load standardizer
            self.scaler = checkpoint.get('scaler', None)
            
            logger.info(f"Direct prediction model loaded successfully (using intention-free cascade model): {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load direct prediction model: {e}")
            self.dl_model = None
    
    def _load_direct_model_shared(self, model_path: str):
        """Load direct prediction model using shared instance"""
        if model_path in DirectModelRouter._shared_models:
            shared_data = DirectModelRouter._shared_models[model_path]
            self.dl_model = shared_data['model']
            self.scaler = shared_data['scaler']
            self.model_config = shared_data['config']
            logger.info(f"Using shared direct prediction model instance: {model_path}")
            return
        
        self._load_direct_model(model_path)
        
        if self.dl_model is not None:
            DirectModelRouter._shared_models[model_path] = {
                'model': self.dl_model,
                'scaler': self.scaler,
                'config': self.model_config
            }
            logger.info(f"Direct prediction model cached as shared instance: {model_path}")
    
    def _init_encodings(self):
        """Initialize encoding mappings - consistent with cascade model"""
        self.input_encoding = {
            'occupation': {
                'graduate student': 0,
                'undergraduate': 1
            },
            'work_intensity': {
                'high': 0,
                'medium': 1,
                'low': 2
            },
            'energy_demand': {
                'high': 0,
                'medium': 1,
                'low': 2
            },
            'thermal_preference': {
                'warm': 0,
                'normal': 1,
                'cool': 2
            },
            'occupational_category': {
                'engineering': 0,
                'science': 1,
                'arts': 2,
                'business': 3,
                'medicine': 4
            },
            'previous_location': {
                'dormitory': 0,
                'classroom': 1,
                'library': 2,
                'laboratory': 3,
                'canteen': 4,
                'other': 5
            },
            'previous_appliance': {
                'laptop': 0,
                'light': 1,
                'phone_charger': 2,
                'air_conditioner': 3,
                'experimental_equipment': 4,
                'none': 5
            },
            'environmental_conditions': {
                'hot': 0,
                'warm': 1,
                'cool': 2,
                'cold': 3,
                'bad weather': 4
            }
        }
        
        self.output_decoding = {
            'location': {
                0: 'dormitory',
                1: 'classroom',
                2: 'library',
                3: 'laboratory',
                4: 'canteen',
                5: 'other'
            },
            'intention': {
                0: 'strong',
                1: 'moderate',
                2: 'weak',
                3: 'none'
            },
            'appliance': {
                0: 'laptop',
                1: 'light',
                2: 'phone_charger',
                3: 'air_conditioner',
                4: 'experimental_equipment',
                5: 'none'
            }
        }
    
    def encode_input(self, timestamp: str, student_profile: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode input data - consistent with cascade model
        
        Args:
            timestamp: Timestamp string (e.g., "Monday 14:30")
            student_profile: Student profile information
            
        Returns:
            Encoded feature vector
        """
        try:
            # Parse timestamp
            parts = timestamp.split()
            day = parts[0]
            time = parts[1]
            
            # Day of week encoding
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            day_of_week = day_map.get(day, 0)
            
            # Convert time to hours
            hour, minute = map(int, time.split(':'))
            hour_decimal = hour + minute / 60.0
            
            # Build feature vector - consistent with training configuration (11 dimensions)
            features = [
                day_of_week,  # timestamp_day
                hour_decimal,  # timestamp_hour
                self.input_encoding['occupation'].get(student_profile.get('occupation', 'undergraduate'), 1),
                student_profile.get('age', 20),  # Use original age value, no normalization
                self.input_encoding['work_intensity'].get(student_profile.get('work_intensity', 'medium'), 1),
                self.input_encoding['energy_demand'].get(student_profile.get('energy_demand', 'medium'), 1),
                self.input_encoding['previous_location'].get(student_profile.get('previous_location', 'other'), 4),
                self.input_encoding['previous_appliance'].get(student_profile.get('previous_appliance', 'none'), 5),
                self.input_encoding['environmental_conditions'].get(student_profile.get('environmental_conditions', 'warm'), 1),
                self.input_encoding['occupational_category'].get(student_profile.get('occupational_category', 'engineering'), 0),
                self.input_encoding['thermal_preference'].get(student_profile.get('thermal_preference', 'normal'), 1)
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Input encoding failed: {e}")
            return None
    
    def check_input_anomaly(self, input_data: Dict[str, Any], llm_function) -> Tuple[bool, str]:
        """
        Check if input is anomalous - consistent with cascade model
        
        Args:
            input_data: Input data
            llm_function: LLM calling function (async)
            
        Returns:
            (is_anomaly, description)
        """
        try:
            # Extract environmental condition
            student_profile = input_data.get('student_profile', {})
            timestamp = input_data.get('timestamp', '')
            environmental_condition = student_profile.get('environmental_condition', 'Normal')
            
            logger.info(f"Starting anomaly detection - environmental condition: '{environmental_condition}', timestamp: '{timestamp}'")
            logger.info(f"Anomaly mapping dictionary current size: {len(self.anomaly_intention_map)}, existing keys: {list(self.anomaly_intention_map.keys())}")
            
            # Preset normal environment list
            preset_normal_environments = {'hot', 'warm', 'cool', 'cold', 'bad weather'}
            
            # Step 0: Check if it's a preset normal environment
            if environmental_condition.lower() in preset_normal_environments:
                logger.info(f"Environmental condition '{environmental_condition}' is a preset normal environment, no anomaly detection needed")
                # Record to dictionary (normal state, map to itself)
                self.anomaly_intention_map[environmental_condition] = (environmental_condition, 'moderate', False)
                return False, f"Normal preset environment: {environmental_condition}"
            
            # Step 1: Check if there's record in anomaly mapping dictionary (contains anomaly to normal environment mapping)
            if environmental_condition in self.anomaly_intention_map:
                preset_environment, stored_intention, stored_is_anomaly = self.anomaly_intention_map[environmental_condition]
                logger.info(f"Using record from anomaly mapping dictionary: preset environment={preset_environment}, anomaly={stored_is_anomaly}, intention={stored_intention}")
                return stored_is_anomaly, f"Cached: {'Anomaly' if stored_is_anomaly else 'Normal'} -> {preset_environment}"
            else:
                logger.info(f"Environmental condition '{environmental_condition}' not found in mapping dictionary, new detection needed")
            
            # Step 2: If no record and no LLM function, return default result
            if not llm_function:
                logger.info(f"Environmental condition '{environmental_condition}' has no record and no LLM function, default to normal")
                # Record to dictionary (normal state)
                self.anomaly_intention_map[environmental_condition] = ('warm', 'weak', False)
                return False, "Normal (default)"
            
            logger.info(f"Environmental condition '{environmental_condition}' needs async LLM detection, building prompt...")
            
            # Step 3: Build anomaly detection prompt (direct model version, does not include intention)
            anomaly_check_prompt = f"""
            Please analyze whether the following environmental condition is normal, particularly checking for abnormal situations (such as typhoon, power outage, extreme weather, etc.):
            
            Time: {timestamp}
            Environmental condition: {environmental_condition}
            
            Available preset normal environments: hot, warm, cool, cold, bad weather
            
            Please respond in the following JSON format:
            {{
                "is_anomaly": true/false,
                "description": "anomaly description (if abnormal) or \"normal\" (if normal)",
                "preset_environment": "hot/warm/cool/cold/bad weather (map to closest preset environment)"
            }}
            
            Judgment criteria:
            - Please select the preset environment that is closest to the current environment.
            - For abnormal conditions, map to the closest preset environment:
              * Very hot, extremely hot → hot
              * Very cold, extremely cold → cold  
              * Heavy rain, storm → bad weather
              * Normal comfortable temperature → warm
              * Slightly cool → cool
              * Other abnormal conditions → choose the most similar preset environment
            
            Note: This is for direct deep learning model which does not support intention injection.
            
            """
            
            # Return None to let caller handle async logic, while passing the prompt
            return None, anomaly_check_prompt
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, ""
    
    def predict_with_confidence(self, features: np.ndarray) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Predict using direct prediction model and calculate confidence (adapted for intention-free cascade model)
        
        Args:
            features: Input features
            
        Returns:
            (prediction result, overall confidence)
        """
        if self.dl_model is None or self.scaler is None:
            logger.error("Direct prediction model not loaded")
            return None
        
        try:
            # Standardize features
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Model prediction (adapt to new return format)
            with torch.no_grad():
                location_logits, appliance_logits, location_pred = self.dl_model(features_tensor)
                
                # Calculate probabilities
                location_probs = torch.softmax(location_logits, dim=1)
                appliance_probs = torch.softmax(appliance_logits, dim=1)
                
                # Get prediction results (new model only returns location and appliance predictions)
                location_pred_idx = location_pred.item()
                _, appliance_pred_idx = torch.max(appliance_probs, dim=1)
                appliance_pred_idx = appliance_pred_idx.item()
                
                # Get confidence
                location_confidence = torch.max(location_probs).item()
                appliance_confidence = torch.max(appliance_probs).item()
                
                # Calculate overall confidence (only consider location and appliance)
                overall_confidence = (location_confidence + appliance_confidence) / 2.0
                
                # Decode prediction results (remove intention part)
                result = {
                    'location': self.output_decoding['location'].get(location_pred_idx, 'other'),
                    'intention': 'weak',  # Fixed intention value, because model doesn't predict intention
                    'appliance': self.output_decoding['appliance'].get(appliance_pred_idx, 'none'),
                    'confidence': {
                        'location': location_confidence,
                        'intention': 0.5,  # Fixed confidence
                        'appliance': appliance_confidence,
                        'overall': overall_confidence
                    }
                }
                
                return result, overall_confidence
                
        except Exception as e:
            logger.error(f"Direct prediction model prediction failed: {e}")
            return None
    
    async def route(self, timestamp: str, student_profile: Dict[str, Any], llm_function) -> Tuple[Dict[str, Any], str]:
        """
        Direct prediction model routing - supports anomaly detection, consistent with hybrid architecture
        
        Args:
            timestamp: Timestamp
            student_profile: Student profile
            llm_function: LLM calling function (async)
            
        Returns:
            (decision result, model type used: 'direct_dl')
        """
        # Get environmental condition
        environmental_condition = student_profile.get('environmental_condition', 'Normal')
        
        # Step 1: Check input anomaly (prefer cache to avoid repeated LLM calls)
        has_anomaly, anomaly_desc = self.check_input_anomaly({
            'timestamp': timestamp,
            'student_profile': student_profile
        }, llm_function)
        
        # If check_input_anomaly returns None (needs async processing), handle async logic
        if has_anomaly is None:
            # anomaly_desc contains the anomaly detection prompt
            logger.info(f"Entering async anomaly detection flow - prompt length: {len(anomaly_desc) if anomaly_desc else 0} characters")
            try:
                # Call LLM for anomaly detection
                logger.info(f"=== Starting LLM call for anomaly detection ===")
                llm_response = await llm_function(anomaly_desc)
                logger.info(f"=== LLM Anomaly Detection Response ===")
                logger.info(f"LLM Response Type: {type(llm_response)}")
                logger.info(f"LLM Response Content: {llm_response}")
                
                # Parse LLM response (assuming JSON format returned)
                logger.info(f"=== Starting to parse LLM anomaly detection response ===")
                
                if isinstance(llm_response, dict) and 'is_anomaly' in llm_response:
                    has_anomaly = llm_response['is_anomaly']
                    anomaly_desc = llm_response.get('description', 'Unknown')
                    preset_environment = llm_response.get('preset_environment', 'warm')  # Get mapped preset environment
                    
                    # Validate preset environment validity
                    valid_preset_environments = {'hot', 'warm', 'cool', 'cold', 'bad weather'}
                    if preset_environment not in valid_preset_environments:
                        logger.warning(f"LLM returned preset_environment '{preset_environment}' is invalid, using default value 'warm'")
                        preset_environment = 'warm'
                    
                    logger.info(f"LLM response parsed successfully - anomaly: {has_anomaly}, description: {anomaly_desc}, preset environment: {preset_environment}")
                    
                    # Update anomaly mapping dictionary, direct model doesn't use intention, uses fixed value
                    self.anomaly_intention_map[environmental_condition] = (preset_environment, 'weak', has_anomaly)
                    logger.info(f"Updated anomaly mapping dictionary: {environmental_condition} -> (preset environment: {preset_environment}, intention: weak (direct model fixed), is_anomaly: {has_anomaly})")
                elif isinstance(llm_response, dict) and 'action' in llm_response:
                    # If LLM returns normal decision format, it means it entered error mode
                    logger.warning(f"LLM returned normal decision format instead of anomaly detection format, might have failed to identify anomaly detection task")
                    logger.warning(f"Response content: {llm_response}")
                    # Try simple judgment based on environmental condition keywords
                    env_condition = environmental_condition.lower()
                    has_anomaly = any(keyword in env_condition for keyword in ['typhoon', 'power outage', 'extreme', 'failure', 'broken', 'storm'])
                    anomaly_desc = f"Fallback detection based on keywords: {'Anomaly' if has_anomaly else 'Normal'}"
                    
                    # Direct model doesn't use intention, only handles environment mapping
                    if has_anomaly:
                        # Choose appropriate preset environment based on keywords
                        if 'hot' in env_condition or 'warm' in env_condition:
                            fallback_preset = 'warm'
                        elif 'cold' in env_condition:
                            fallback_preset = 'cold'
                        elif any(weather_keyword in env_condition for weather_keyword in ['rain', 'storm', 'typhoon']):
                            fallback_preset = 'bad weather'
                        else:
                            fallback_preset = 'warm'  # Default to warm
                    else:
                        fallback_preset = 'warm'  # Default preset environment
                    
                    # Update anomaly mapping dictionary (direct model version, doesn't use intention)
                    self.anomaly_intention_map[environmental_condition] = (fallback_preset, 'weak', has_anomaly)
                    logger.info(f"Using keyword fallback detection - anomaly: {has_anomaly}, description: {anomaly_desc}, preset environment: {fallback_preset} (direct model doesn't use intention)")
                else:
                    # If response is not in expected format, try to parse as normal
                    logger.warning(f"LLM response format does not meet expectations, using default values - response type: {type(llm_response)}, content: {llm_response}")
                    # Also try keyword detection as fallback
                    env_condition = environmental_condition.lower()
                    has_anomaly = any(keyword in env_condition for keyword in ['typhoon', 'power outage', 'extreme', 'failure', 'broken', 'storm'])
                    anomaly_desc = f"Fallback detection based on keywords: {'Anomaly' if has_anomaly else 'Normal'}"
                    
                    # Direct model doesn't use intention, only handles environment mapping
                    if has_anomaly:
                        # Choose appropriate preset environment based on keywords
                        if 'hot' in env_condition or 'warm' in env_condition:
                            fallback_preset = 'warm'
                        elif 'cold' in env_condition:
                            fallback_preset = 'cold'
                        elif any(weather_keyword in env_condition for weather_keyword in ['rain', 'storm', 'typhoon']):
                            fallback_preset = 'bad weather'
                        else:
                            fallback_preset = 'warm'  # Default to warm
                    else:
                        fallback_preset = 'warm'  # Default preset environment
                    
                    # Update anomaly mapping dictionary (direct model version, doesn't use intention)
                    self.anomaly_intention_map[environmental_condition] = (fallback_preset, 'weak', has_anomaly)
                    logger.info(f"Using keyword fallback detection - anomaly: {has_anomaly}, description: {anomaly_desc}, preset environment: {fallback_preset} (direct model doesn't use intention)")
                
                # Anomaly mapping dictionary has been updated above, here we just need to log completion
                logger.info(f"LLM anomaly detection completed: anomaly={has_anomaly}, description={anomaly_desc}")
                
            except Exception as e:
                logger.error(f"LLM anomaly detection failed: {e}, default to normal processing")
                has_anomaly = False
                anomaly_desc = 'Normal (LLM failed)'
                # Record failure result, use default preset environment (direct model doesn't use intention)
                self.anomaly_intention_map[environmental_condition] = ('warm', 'weak', False)
                logger.info(f"LLM anomaly detection failed, using default preset environment: warm (direct model doesn't use intention, anomaly status: False)")
        
        # Step 2: Use direct prediction model (regardless of anomaly)
        features = self.encode_input(timestamp, student_profile)
        if features is not None:
            
            # Direct model only handles environment mapping, doesn't support intention injection
            mapped_environment = environmental_condition  # Default map to itself
            
            if environmental_condition in self.anomaly_intention_map:
                preset_environment, stored_intention, stored_is_anomaly = self.anomaly_intention_map[environmental_condition]
                
                # Regardless of anomaly, only use mapped environment, don't use intention (direct model characteristic)
                mapped_environment = preset_environment
                if stored_is_anomaly:
                    logger.info(f"Detected anomaly, using preset environment from anomaly mapping dictionary: {preset_environment} (direct model doesn't support intention injection)")
                else:
                    logger.info(f"Normal environment, using mapped preset environment: {preset_environment} (direct model doesn't support intention injection)")
            else:
                # Use default value if there's no record for some reason
                mapped_environment = 'warm'
                if has_anomaly:
                    self.anomaly_intention_map[environmental_condition] = (mapped_environment, 'weak', has_anomaly)
                    logger.info(f"Detected anomaly but no record in mapping, using default preset environment: {mapped_environment} (direct model doesn't support intention injection)")
                else:
                    logger.info(f"Normal environment but no record in mapping, using default preset environment: {mapped_environment} (direct model doesn't support intention injection)")
            
            # Update environmental condition in student profile to mapped preset environment
            student_profile_copy = student_profile.copy()
            student_profile_copy['environmental_conditions'] = mapped_environment
            
            # Re-encode input (using mapped environment)
            features_mapped = self.encode_input(timestamp, student_profile_copy)
            if features_mapped is not None:
                # Use direct prediction model for prediction (only use mapped environment, don't preset intention)
                dl_result = self.predict_with_confidence(features_mapped)
                
                if dl_result is not None:
                    decision, confidence = dl_result
                    
                    # Decide based on confidence (consistent with cascade model)
                    if confidence >= self.confidence_threshold:
                        logger.info(f"Direct prediction model confidence {confidence:.3f} >= threshold {self.confidence_threshold}, using direct_dl model")
                        return decision, 'direct_dl'
                    else:
                        logger.info(f"Direct prediction model confidence {confidence:.3f} < threshold {self.confidence_threshold}, preparing to use LLM")
                        # When confidence is insufficient, continue to LLM branch
                else:
                    logger.warning("Direct prediction model prediction failed, preparing to use LLM")
            else:
                logger.warning("Mapped input encoding failed")
        else:
            logger.warning("Original input encoding failed")
        
        # Step 3: If anomaly is detected but direct prediction model confidence is insufficient, use LLM
        if has_anomaly:
            logger.info(f"Detected anomaly event, direct prediction model confidence insufficient, using LLM to handle anomaly")
        
        # Step 4: Use LLM (when confidence is insufficient or model fails)
        if llm_function:
            try:
                logger.info("Calling LLM for decision (confidence insufficient or model failed)")
                result = await llm_function()
                return result, 'llm'
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        # Step 5: If all methods fail, return default decision
        logger.error("Direct prediction model routing failed, returning default decision")
        return {
            'location': 'dormitory',
            'intention': 'weak',
            'appliance': 'none',
            'confidence': {
                'location': 0.0,
                'intention': 0.0,
                'appliance': 0.0,
                'overall': 0.0
            }
        }, 'direct_dl'
