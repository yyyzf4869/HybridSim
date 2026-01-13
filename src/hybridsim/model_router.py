"""
Model routing module - implements intelligent routing between deep learning and LLM
"""
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ModelRouter:
    """Intelligent model router - decides whether to use deep learning model or LLM"""
    
    # Class variable: shared model instances (avoid duplicate loading)
    _shared_models = {}
    
    def __init__(self, dl_model_path: str = None, confidence_threshold: float = 0.8):
        """
        Initialize model router
        
        Args:
            dl_model_path: Deep learning model path
            confidence_threshold: Confidence threshold, use DL model if above this value
        """
        self.confidence_threshold = confidence_threshold
        self.dl_model = None
        self.scaler = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Anomaly mapping dictionary - stores Environmental condition to (intention, is_anomaly) mapping
        # Structure: {environmental_condition: (intention, is_anomaly)}
        self.anomaly_intention_map = {}
        
        # Load deep learning model (use shared instance)
        if dl_model_path and os.path.exists(dl_model_path):
            self._load_dl_model_shared(dl_model_path)
        else:
            logger.warning(f"Deep learning model path doesn't exist: {dl_model_path}")
        
        # Encoding mapping
        self._init_encodings()
        
        logger.info(f"Model router initialization completed - confidence threshold: {confidence_threshold}")
    
    def _load_dl_model(self, model_path: str):
        """Load deep learning model"""
        try:
            # Use weights_only=False to maintain backward compatibility (trusted file source)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Import model class
            import sys
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dl_model'))
            from cascade_energy_model import CascadeEnergyModel
            
            # Create model
            config = checkpoint.get('model_config', {'input_dim': 11, 'num_locations': 6, 'num_intentions': 4, 'num_appliances': 6})
            self.model_config = config
            
            self.dl_model = CascadeEnergyModel(
                input_dim=config['input_dim'],
                num_locations=config['num_locations'],
                num_intentions=config['num_intentions'],
                num_appliances=config['num_appliances']
            ).to(self.device)
            
            self.dl_model.load_state_dict(checkpoint['model_state_dict'])
            self.dl_model.eval()
            
            # Load normalizer
            self.scaler = checkpoint.get('scaler', None)
            
            logger.info(f"Deep learning model loaded successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load deep learning model: {e}")
            self.dl_model = None
    
    def _load_dl_model_shared(self, model_path: str):
        """Load deep learning model using shared instance (avoid duplicate loading)"""
        # Check if shared instance already exists
        if model_path in ModelRouter._shared_models:
            # Use existing shared model
            shared_data = ModelRouter._shared_models[model_path]
            self.dl_model = shared_data['model']
            self.scaler = shared_data['scaler']
            self.model_config = shared_data['config']
            logger.info(f"Using shared deep learning model instance: {model_path}")
            return
        
        # First time loading model
        self._load_dl_model(model_path)
        
        # If loading successful, save to shared cache
        if self.dl_model is not None:
            ModelRouter._shared_models[model_path] = {
                'model': self.dl_model,
                'scaler': self.scaler,
                'config': self.model_config
            }
            logger.info(f"Deep learning model cached as shared instance: {model_path}")
    
    def _init_encodings(self):
        """Initialize encoding mapping"""
        # input feature encoding - consistent with training configuration
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
            # Add missing feature encodings
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
        
        # Output feature encoding (reverse mapping)
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
    
    def check_input_anomaly(self, input_data: Dict[str, Any], llm_function=None) -> Tuple[bool, str]:
        """
        Check input data for anomalies
        
        Args:
            input_data: Input data
            llm_function: LLM calling function (async) - used for anomaly detection
            
        Returns:
            (is_anomaly, anomaly_description)
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
                # Record in dictionary (normal state, map to itself)
                self.anomaly_intention_map[environmental_condition] = (environmental_condition, 'moderate', False)
                return False, f"Normal preset environment: {environmental_condition}"
            
            # Step 1: Check if there's a record in anomaly mapping dictionary (contains mapping from anomaly to normal environment)
            if environmental_condition in self.anomaly_intention_map:
                preset_environment, stored_intention, stored_is_anomaly = self.anomaly_intention_map[environmental_condition]
                logger.info(f"Using record from anomaly mapping dictionary: preset environment={preset_environment}, anomaly={stored_is_anomaly}, intention={stored_intention}")
                return stored_is_anomaly, f"Cached: {'Anomaly' if stored_is_anomaly else 'Normal'} -> {preset_environment}"
            else:
                logger.info(f"Environmental condition '{environmental_condition}' not found in mapping dictionary, new detection needed")
            
            # Step 2: If no record and no LLM function, return default result
            if not llm_function:
                logger.info(f"Environmental condition '{environmental_condition}' has no record and no LLM function, default to normal")
                # Record in dictionary (normal state)
                self.anomaly_intention_map[environmental_condition] = ('weak', False)
                return False, "Normal (default)"
            
            logger.info(f"Environmental condition '{environmental_condition}' needs async LLM detection, building prompt...")
            
            # Step 3: Build anomaly detection prompt (English) - add mapping from anomaly to normal environment
            anomaly_check_prompt = f"""
            Please analyze whether the following environmental condition is normal, particularly checking for abnormal situations (such as typhoon, power outage, extreme weather, etc.):
            
            Time: {timestamp}
            Environmental condition: {environmental_condition}
            
            Available preset normal environments: hot, warm, cool, cold, bad weather
            
            Please respond in the following JSON format:
            {{
                "is_anomaly": true/false,
                "description": "anomaly description (if abnormal) or \"normal\" (if normal)",
                "intention": "strong/moderate/weak/none",
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
            
            Intention selection guide:
            - "strong": High energy usage intention (Due to the needs of study/work/life, electrical equipment must be used.)
            - "moderate": Medium energy usage intention (Due to the needs of study/work/life, it may be necessary to use electrical equipment.)
            - "weak": Low energy usage intention (Less need to use electrical equipment due to study/work/life.)
            - "none": No significant energy usage intention (No need or inability to use electrical equipment.)
            
            """
            
            
            # Step 4: Call LLM for anomaly detection
            # Create temporary LLM call function
            async def check_llm():
                return await llm_function(anomaly_check_prompt)
            
            # Simplified processing here, actual use requires async call
            # Return None to let caller handle async logic, while passing prompt
            return None, anomaly_check_prompt
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, ""
    
    def encode_input(self, timestamp: str, student_profile: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode input data
        
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
            
            # Weekday encoding
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
    
    def predict_with_confidence(self, features: np.ndarray, preset_intention: str = None) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Predict using deep learning model and calculate confidence
        
        Args:
            features: Input features
            preset_intention: Preset intention (used in abnormal situations, if None, let model decide freely)
            
        Returns:
            (prediction result, overall confidence)
        """
        if self.dl_model is None or self.scaler is None:
            logger.error("Deep learning model not loaded")
            return None
        
        try:
            # Standardize features
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                location_logits, intention_logits, appliance_logits, _, _ = self.dl_model(features_tensor)
                
                # Calculate probabilities
                location_probs = torch.softmax(location_logits, dim=1)
                intention_probs = torch.softmax(intention_logits, dim=1)
                appliance_probs = torch.softmax(appliance_logits, dim=1)
                
                # Get prediction results
                location_pred = torch.argmax(location_probs, dim=1).item()
                
                # If preset intention exists, use preset value; otherwise use model prediction
                if preset_intention:
                    # Create reverse mapping: from intention string to index
                    intention_to_idx = {v: k for k, v in self.output_decoding['intention'].items()}
                    intention_pred = intention_to_idx.get(preset_intention, 3)  # Default none
                    intention_confidence = 0.95  # High confidence for preset intention
                else:
                    intention_pred = torch.argmax(intention_probs, dim=1).item()
                    intention_confidence = torch.max(intention_probs).item()
                
                appliance_pred = torch.argmax(appliance_probs, dim=1).item()
                
                # Get confidence (maximum probability)
                location_confidence = torch.max(location_probs).item()
                appliance_confidence = torch.max(appliance_probs).item()
                # intention_confidence is already set above
                
                # Calculate overall confidence (average)
                overall_confidence = (location_confidence + intention_confidence + appliance_confidence) / 3.0
                
                # Decode prediction results
                result = {
                    'location': self.output_decoding['location'].get(location_pred, 'other'),
                    'intention': self.output_decoding['intention'].get(intention_pred, 'weak'),
                    'appliance': self.output_decoding['appliance'].get(appliance_pred, 'none'),
                    'confidence': {
                        'location': location_confidence,
                        'intention': intention_confidence,
                        'appliance': appliance_confidence,
                        'overall': overall_confidence
                    }
                }
                
                return result, overall_confidence
                
        except Exception as e:
            logger.error(f"Deep learning model prediction failed: {e}")
            return None
    
    async def route(self, timestamp: str, student_profile: Dict[str, Any], llm_function) -> Tuple[Dict[str, Any], str]:
        """
        Intelligent routing - Decide whether to use deep learning model or LLM
        
        Args:
            timestamp: Timestamp
            student_profile: Student profile
            llm_function: LLM calling function (async)
            
        Returns:
            (decision result, model type used: 'dl' or 'llm')
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
                logger.info(f"=== LLM anomaly detection response ===")
                logger.info(f"LLM response type: {type(llm_response)}")
                logger.info(f"LLM response content: {llm_response}")
                
                # Parse LLM response (assuming JSON format)
                logger.info(f"=== Starting to parse LLM anomaly detection response ===")
                
                if isinstance(llm_response, dict) and 'is_anomaly' in llm_response:
                    has_anomaly = llm_response['is_anomaly']
                    anomaly_desc = llm_response.get('description', 'Unknown')
                    llm_intention = llm_response.get('intention', 'weak')  # Get intention returned by LLM
                    preset_environment = llm_response.get('preset_environment', 'warm')  # Get mapped preset environment
                    
                    # Validate intention validity
                    valid_intentions = ['strong', 'moderate', 'weak', 'none']
                    if llm_intention not in valid_intentions:
                        logger.warning(f"LLM returned intention '{llm_intention}' is invalid, using default value 'weak'")
                        llm_intention = 'weak'
                    
                    # Validate preset environment validity
                    valid_preset_environments = {'hot', 'warm', 'cool', 'cold', 'bad weather'}
                    if preset_environment not in valid_preset_environments:
                        logger.warning(f"LLM returned preset_environment '{preset_environment}' is invalid, using default value 'warm'")
                        preset_environment = 'warm'
                    
                    logger.info(f"LLM response parsed successfully - anomaly: {has_anomaly}, description: {anomaly_desc}, intention: {llm_intention}, preset environment: {preset_environment}")
                    
                    # Update anomaly mapping dictionary using new triple structure (preset environment, intention, is_anomaly)
                    self.anomaly_intention_map[environmental_condition] = (preset_environment, llm_intention, has_anomaly)
                    logger.info(f"Updated anomaly mapping dictionary: {environmental_condition} -> (preset environment: {preset_environment}, intention: {llm_intention}, is_anomaly: {has_anomaly})")
                elif isinstance(llm_response, dict) and 'action' in llm_response:
                    # If LLM returns normal decision format, it means it entered error mode
                    logger.warning(f"LLM returned normal decision format instead of anomaly detection format, may not have correctly identified anomaly detection task")
                    logger.warning(f"Response content: {llm_response}")
                    # Try simple judgment based on environmental condition keywords
                    env_condition = environmental_condition.lower()
                    has_anomaly = any(keyword in env_condition for keyword in ['typhoon', 'power outage', 'extreme', 'failure', 'broken', 'storm'])
                    anomaly_desc = f"Fallback detection based on keywords: {'Anomaly' if has_anomaly else 'Normal'}"
                    
                    # Choose appropriate intention and preset environment based on anomaly type
                    if has_anomaly:
                        # For severe anomalies, use none intention; for general anomalies, use weak intention
                        fallback_intention = 'none' if any(severe_keyword in env_condition for severe_keyword in ['typhoon', 'power outage', 'extreme']) else 'weak'
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
                        fallback_intention = 'moderate'  # Use moderate intention for normal conditions
                        fallback_preset = 'warm'  # Default preset environment
                    
                    # Update anomaly mapping dictionary (new triple structure)
                    self.anomaly_intention_map[environmental_condition] = (fallback_preset, fallback_intention, has_anomaly)
                    logger.info(f"Using keyword fallback detection - anomaly: {has_anomaly}, description: {anomaly_desc}, preset environment: {fallback_preset}, intention: {fallback_intention}")
                else:
                    # If response is not in expected format, try to parse as normal
                    logger.warning(f"LLM response format does not meet expectations, using default values - response type: {type(llm_response)}, content: {llm_response}")
                    # Also try keyword detection as fallback
                    env_condition = environmental_condition.lower()
                    has_anomaly = any(keyword in env_condition for keyword in ['typhoon', 'power outage', 'extreme', 'failure', 'broken', 'storm'])
                    anomaly_desc = f"Fallback detection based on keywords: {'Anomaly' if has_anomaly else 'Normal'}"
                    
                    # Choose appropriate intention and preset environment based on anomaly type
                    if has_anomaly:
                        # For severe anomalies, use none intention; for general anomalies, use weak intention
                        fallback_intention = 'none' if any(severe_keyword in env_condition for severe_keyword in ['typhoon', 'power outage', 'extreme']) else 'weak'
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
                        fallback_intention = 'moderate'  # Use moderate intention for normal conditions
                        fallback_preset = 'warm'  # Default preset environment
                    
                    # Update anomaly mapping dictionary (new triple structure)
                    self.anomaly_intention_map[environmental_condition] = (fallback_preset, fallback_intention, has_anomaly)
                    logger.info(f"Using keyword fallback detection - anomaly: {has_anomaly}, description: {anomaly_desc}, preset environment: {fallback_preset}, intention: {fallback_intention}")
                
                # Anomaly mapping dictionary has been updated above, here we just need to log completion
                logger.info(f"LLM anomaly detection completed: anomaly={has_anomaly}, description={anomaly_desc}")
                
            except Exception as e:
                logger.error(f"LLM anomaly detection failed: {e}, default to normal processing")
                has_anomaly = False
                anomaly_desc = 'Normal (LLM failed)'
                # Record failure result, use default preset environment and intention
                self.anomaly_intention_map[environmental_condition] = ('warm', 'weak', False)
                logger.info(f"LLM anomaly detection failed, using default preset environment: warm, intention: weak (anomaly status: False)")
        
        # Step 2: Try using deep learning model (regardless of anomaly)
        if self.dl_model is not None:
            features = self.encode_input(timestamp, student_profile)
            if features is not None:
                
                # Only preset intention when there's anomaly, let deep learning model decide freely when no anomaly
                preset_intention = None
                mapped_environment = environmental_condition  # Default map to itself
                
                if environmental_condition in self.anomaly_intention_map:
                    preset_environment, stored_intention, stored_is_anomaly = self.anomaly_intention_map[environmental_condition]
                    
                    if stored_is_anomaly:
                        # Abnormal case: use mapped preset environment and intention
                        preset_intention = stored_intention
                        mapped_environment = preset_environment
                        logger.info(f"Detected anomaly, using preset environment from anomaly mapping dictionary: {preset_environment}, preset intention: {preset_intention}")
                    else:
                        # Normal case: don't preset intention, let model decide freely, but record mapped environment
                        mapped_environment = preset_environment
                        logger.info(f"Normal environment, using mapped preset environment: {preset_environment}, not presetting intention, letting model decide freely")
                else:
                    # If no record for some reason, use default values
                    if has_anomaly:
                        preset_intention = 'weak'
                        mapped_environment = 'warm'
                        self.anomaly_intention_map[environmental_condition] = (mapped_environment, preset_intention, has_anomaly)
                        logger.info(f"Detected anomaly but no record in mapping, using default preset environment: {mapped_environment}, default intention: {preset_intention}")
                    else:
                        mapped_environment = 'warm'
                        logger.info(f"Normal environment but no record in mapping, using default preset environment: {mapped_environment}, not presetting intention, letting model decide freely")
                
                # Use deep learning model for prediction (using mapped environment and preset intention)
                dl_result = self.predict_with_confidence(features, preset_intention)
                
                if dl_result is not None:
                    decision, confidence = dl_result
                    
                    # Step 3: Decide based on confidence
                    if confidence >= self.confidence_threshold:
                        logger.info(f"Deep learning model confidence {confidence:.3f} >= threshold {self.confidence_threshold}, using DL model")
                        return decision, 'dl'
                    else:
                        logger.info(f"Deep learning model confidence {confidence:.3f} < threshold {self.confidence_threshold}, using LLM")
                else:
                    logger.warning("Deep learning model prediction failed, using LLM")
            else:
                logger.warning("Input encoding failed, using LLM")
        else:
            logger.warning("Deep learning model not loaded, using LLM")
        
        # Step 4: If anomaly detected but deep learning model confidence insufficient, use LLM
        if has_anomaly:
            logger.info(f"Detected anomaly event, deep learning model confidence insufficient, using LLM to handle anomaly: {anomaly_desc}")
        
        # Use LLM
        result = await llm_function()
        return result, 'llm'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'dl_model_loaded': self.dl_model is not None,
            'confidence_threshold': self.confidence_threshold,
            'device': str(self.device),
            'model_config': self.model_config
        }
