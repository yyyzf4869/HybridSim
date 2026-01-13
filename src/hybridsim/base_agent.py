from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import httpx
from .token_usage import get_token_tracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base agent class"""
    
    def __init__(self, agent_id: str, name: str = ""):
        self.agent_id = agent_id
        self.name = name or agent_id
        self.event_queue = asyncio.Queue()
        self.running = False
        self.handlers = {}
        
    def register_handler(self, event_type: str, handler_func):
        """Register event handler"""
        self.handlers[event_type] = handler_func
        logger.info(f"Agent {self.agent_id} registered handler for {event_type}")
    
    async def send_event(self, event: 'Event'):
        """Send event to queue"""
        await self.event_queue.put(event)
        logger.debug(f"Agent {self.agent_id} received event: {event.event_type}")
    
    async def process_events(self):
        """Process events in the event queue"""
        while self.running:
            try:
                # Set timeout to avoid infinite blocking
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error processing event: {e}")
    
    async def handle_event(self, event: 'Event'):
        """Handle a single event"""
        handler = self.handlers.get(event.event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error handling {event.event_type}: {e}")
        else:
            logger.warning(f"Agent {self.agent_id} no handler for {event.event_type}")
    
    async def start(self):
        """Start the agent"""
        self.running = True
        logger.info(f"Agent {self.agent_id} started")
        await self.process_events()
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent"""
        pass

class LLMAgent(BaseAgent):
    """Base agent class supporting LLM"""
    
    def __init__(self, agent_id: str, name: str, sys_prompt: str, model_config: Dict[str, Any]):
        super().__init__(agent_id, name)
        self.sys_prompt = sys_prompt
        self.model_config = model_config
        self.memory = []
        
    async def initialize(self):
        """Initialize the agent"""
        # LLMAgent initialization doesn't require special operations, can be left empty or add logs
        pass
        
    async def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate LLM response"""
        self.memory.append({"role": "user", "content": user_input})
        
        try:
            # Get first model configuration
            model_config = self.model_config["chat"][0]
            
            # Build request data
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Build request parameters
            request_data = {
                "model": model_config["model_name"],
                "messages": messages,
                "temperature": model_config["generate_args"]["temperature"],
                "max_tokens": 500
            }
            
            # Set request headers
            headers = {
                "Authorization": f"Bearer {model_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            # Send request to LLM service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{model_config['client_args']['base_url']}chat/completions",
                    headers=headers,
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    llm_response = result["choices"][0]["message"]["content"]
                    
                    # Get token usage information
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    # Track token usage
                    tracker = get_token_tracker()
                    tracker.track(
                        model_name=model_config["model_name"],
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                    
                    logger.info(f"LLM Agent {self.agent_id} received response from {model_config['model_name']} "
                               f"(tokens: {total_tokens}, prompt: {prompt_tokens}, completion: {completion_tokens})")
                    
                    # Save to memory
                    self.memory.append({"role": "assistant", "content": llm_response})
                    
                    # Try to parse as JSON format
                    try:
                        parsed_response = json.loads(llm_response)
                        return parsed_response
                    except json.JSONDecodeError:
                        # If not JSON, return original text
                        return {"content": llm_response}
                else:
                    logger.error(f"LLM service returned error: {response.status_code} - {response.text}")
                    # Return simulated response as fallback
                    return self._generate_fallback_response(user_input)
                    
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Return simulated response as fallback
            return self._generate_fallback_response(user_input)
    
    def _generate_fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Generate fallback response"""
        # Estimate token usage (Chinese characters approximately 1.5-2 tokens, plus some overhead)
        prompt_tokens = len(user_input) * 2 + 50  # System prompt, etc.
        completion_tokens = 50  # Simulated response token count
        total_tokens = prompt_tokens + completion_tokens
        
        # Track simulated token usage
        tracker = get_token_tracker()
        tracker.track(
            model_name="fallback_simulation",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        logger.info(f"LLM Agent {self.agent_id} using fallback response "
                   f"(estimated tokens: {total_tokens})")
        
        fallback_response = f"Simulated LLM response: Analysis result based on '{user_input}'"
        self.memory.append({"role": "assistant", "content": fallback_response})
        return {"content": fallback_response}
    
    def get_memory_context(self) -> str:
        """Get memory context"""
        if not self.memory:
            return ""
        
        context = "Recent conversation history:\n"
        for msg in self.memory[-5:]:  # Recent 5 messages
            context += f"{msg['role']}: {msg['content']}\n"
        return context
