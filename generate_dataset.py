import argparse
from data_generator.orchestrator.simulation_runner import SimulationRunner

def main():
    parser = argparse.ArgumentParser(description="CFD Synthetic Dataset Generator for Transient Heat Conduction")
    parser.add_argument("--config", type=str, default="config/generator_config.yaml",
                        help="Path to the generator YAML configuration file")
    
    args = parser.parse_args()
    
    print("================================================================================")
    print("CFD DATASET GENERATION FRAMEWORK")
    print("================================================================================")
    print(f"Loading configuration from {args.config}...")
    
    runner = SimulationRunner(args.config)
    runner.run_all()
    
    print("\n" + "="*80)
    print("                [SUCCESS] DATASET GENERATION COMPLETED                  ")
    print("="*80 + "\n")
if __name__ == "__main__":
    main()
