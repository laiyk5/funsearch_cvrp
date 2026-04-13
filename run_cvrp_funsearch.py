#!/usr/bin/env python3
"""Run FunSearch on CVRP problem."""

import os
from funsearch_cvrp.cvrp.core import CVRPInstance, generate_synthetic_benchmarks
from funsearch_cvrp.cvrp.io import load_cvrplib_instance
from funsearch_cvrp.funsearch import funsearch, config
from funsearch_cvrp.funsearch.sampler import OpenAILLM
from pathlib import Path


def main():
    # Set up OpenAI API key
    # Option 1: Set environment variable
    # export OPENAI_API_KEY="your-key"
    
    # Option 2: Pass directly (not recommended for production)
    # openai_api_key = "your-key"
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Generate test instances
    print("Generating CVRP test instances...")
    instances = generate_synthetic_benchmarks(seed=2026, sizes=[20, 50])
    print(f"Created {len(instances)} test instances")
    
    # Or load from CVRPLib files
    # instances = [load_cvrplib_instance(Path("data/A/A-n32-k5.vrp"))]
    
    # Load specification
    print("Loading specification...")
    with open("cvrp_spec.py") as f:
        specification = f.read()
    
    # Configure FunSearch
    cfg = config.Config(
        programs_database=config.ProgramsDatabaseConfig(
            num_islands=10,
            reset_period=600,  # Reset islands every 10 minutes
            functions_per_prompt=3,  # Include 3 previous versions in prompt
        ),
        num_evaluators=1,
        num_samplers=1,
        samples_per_prompt=5,  # Generate 5 candidates per prompt
    )
    
    # Create LLM
    print("Setting up OpenAI LLM...")
    llm = OpenAILLM(
        samples_per_prompt=5,
        model="gpt-4o-mini",  # or "gpt-4" for better quality
        temperature=0.7,
        max_tokens=1000,
        api_key=openai_api_key,
    )
    
    # Run FunSearch
    print("\nStarting FunSearch...")
    print("This will iteratively improve the priority function.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        funsearch.main(
            specification=specification,
            inputs=instances,
            config=cfg,
            llm=llm,
        )
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


if __name__ == "__main__":
    main()
