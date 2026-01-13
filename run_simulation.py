#!/usr/bin/env python3
"""
Campus Energy Consumption Simulation Standalone Version
Simulates student energy consumption behavior on campus
"""

import asyncio
import logging
import sys
import argparse
from datetime import datetime

from code.simulator import EnergyConsumptionSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function"""
    logger.info("=== Campus Energy Consumption Simulation System Started ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Campus Energy Consumption Simulation System')
    parser.add_argument('--architecture', type=str, default='hybrid', 
                       choices=['pure_llm', 'hybrid', 'pure_dl', 'pure_rule', 'hybrid_direct', 'pure_direct_dl'],
                       help='Architecture type: pure_llm (pure LLM), hybrid (hybrid), pure_dl (pure deep learning), pure_rule (pure rule-based), hybrid_direct (hybrid + direct prediction model), pure_direct_dl (pure direct prediction model)')
    parser.add_argument('--dl-model-path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'dl_model', 'cascade_energy_model.pth'),
                       help='Deep learning model path') 
    parser.add_argument('--confidence-threshold', type=float, default=0.75,
                       help='Deep learning model confidence threshold (cascade prediction model)')
    parser.add_argument('--direct-confidence-threshold', type=float, default=0.75,
                       help='Direct prediction deep learning model confidence threshold')
    parser.add_argument('--llm-workers', type=int, default=30,
                       help='LLM model parallel worker threads (default: 30)')
    parser.add_argument('--dl-workers', type=int, default=5000,
                       help='Deep learning model parallel worker threads (default: 5000)')
    parser.add_argument('--start-time', type=str, default='Monday 00:00',
                       help='Simulation start time')
    parser.add_argument('--max-steps', type=int, default=24*4,
                       help='Maximum simulation steps')
    parser.add_argument('--simulation-step', type=float, default=0.25,
                       help='Simulation step size, unit: hours')
    parser.add_argument('--feedback', type=str, default='false',
                       choices=['true', 'false'],
                       help='Enable Feedback alignment with real world (default: true)')
    parser.add_argument('--environmental-condition', type=str, default='hot',
                       choices=['hot', 'warm', 'cool', 'cold', 'bad weather'],
                       help='Environmental condition parameters: hot (hot), warm (warm), cool (cool), cold (cold), bad weather (bad weather)')
    parser.add_argument('--agent-count', type=int, default=100,
                       help='Agent count (default: 100, maximum support 10000)')
    parser.add_argument('--feedback-iterations', type=int, default=1,
                       help='Feedback iteration rounds (default: 1)')
    # At present, due to the typhoon, most areas have experienced power outages, making it impossible to use electrical equipment.
    # The heavy rain made it inconvenient to go out.
    args = parser.parse_args()
    
    # Convert feedback parameter to boolean
    enable_feedback = args.feedback.lower() == 'true'
    
    # Check feedback iteration configuration consistency
    if not enable_feedback and args.feedback_iterations > 1:
        logger.warning(f"Feedback function not enabled (--feedback=false), but iteration count set ({args.feedback_iterations}), iteration settings will be ignored")
    
    logger.info(f"Configuration parameters - Architecture: {args.architecture}, Model path: {args.dl_model_path}, "
                f"Confidence threshold: {args.confidence_threshold}, LLM threads: {args.llm_workers}, DL threads: {args.dl_workers}, "
                f"Simulation step: {args.simulation_step} hours, Feedback: {enable_feedback}, "
                f"Environmental condition: {args.environmental_condition}, Agent count: {args.agent_count}, Iteration rounds: {args.feedback_iterations}")
    
    if enable_feedback and args.feedback_iterations > 1:
        logger.info(f"Feedback iteration pipeline enabled, will run {args.feedback_iterations} rounds...")
        from code.feedback_pipeline import FeedbackPipeline
        pipeline = FeedbackPipeline(args)
        await pipeline.run()
        logger.info("=== Simulation system closed (Pipeline completed) ===")
        return

    # Automatically select appropriate model path based on architecture type
    dl_model_path = args.dl_model_path
    if args.architecture in ['hybrid_direct', 'pure_direct_dl']:
        # Direct prediction model architecture uses new no-intention cascade model
        dl_model_path = os.path.join(os.path.dirname(__file__), 'dl_model_no_intention', 'cascade_energy_model_no_intention.pth')
        logger.info(f"Architecture {args.architecture} uses no-intention cascade model: {dl_model_path}")
    elif args.dl_model_path == os.path.join(os.path.dirname(__file__), 'dl_model', 'cascade_energy_model.pth'):
        # Default path, maintain original logic
        logger.info(f"Architecture {args.architecture} uses cascade model: {dl_model_path}")

    try:
        # Create simulator
        simulator = EnergyConsumptionSimulator(
            llm_workers=args.llm_workers,
            dl_workers=args.dl_workers,
            architecture_type=args.architecture,
            dl_model_path=dl_model_path,
            confidence_threshold=args.confidence_threshold,
            direct_confidence_threshold=args.direct_confidence_threshold,
            time_step=args.simulation_step,
            enable_feedback=enable_feedback,
            environmental_condition=args.environmental_condition
        )
        
        # Initialize simulation
        await simulator.initialize_simulation(
            start_time=args.start_time,
            max_steps=args.max_steps,
            agent_count=args.agent_count
        )
        
        # Run simulation
        await simulator.run_simulation()
        
        # Get simulation summary
        summary = simulator.get_simulation_results()
        
        logger.info("=== Simulation completion summary ===")
        logger.info(f"Simulation steps: {summary.get('simulation_steps', 0)}")
        logger.info(f"Student count: {summary.get('student_count', 0)}")
        logger.info(f"Total consumption: {summary.get('total_consumption', 0.0):.3f} kWh")
        logger.info(f"Average per student: {summary.get('average_per_student', 0.0):.3f} kWh")
        
        # Location distribution
        location_summary = summary.get('location_summary', {})
        if location_summary:
            logger.info("=== Location energy consumption distribution ===")
            for location, consumption in location_summary.items():
                logger.info(f"{location}: {consumption:.3f} kWh")
        
        # Appliance distribution
        appliance_summary = summary.get('appliance_summary', {})
        if appliance_summary:
            logger.info("=== Appliance energy consumption distribution ===")
            for appliance, consumption in appliance_summary.items():
                logger.info(f"{appliance}: {consumption:.3f} kWh")
        
    except Exception as e:
        logger.error(f"Simulation run failed: {e}")
        raise
    
    logger.info("=== Simulation system closed ===")

if __name__ == "__main__":
    # Run asynchronous main function
    asyncio.run(main())