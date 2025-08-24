#!/usr/bin/env python3
"""Debug script to trace directory creation during robust optimization"""

import json
from pathlib import Path

# Check what results are being generated
results_file = Path("/Users/b80985/Projects/homodyne/homodyne_results/homodyne_analysis_results.json")

if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=== RESULTS STRUCTURE ===")
    print("Top-level keys:")
    for key in data.get("results", {}).keys():
        print(f"  - {key}")
    
    print("\nMethods used:")
    methods_used = data.get("results", {}).get("methods_used", [])
    print(f"  {methods_used}")
    
    print("\nChecking classical optimization presence:")
    has_classical = "classical_optimization" in data.get("results", {})
    print(f"  has 'classical_optimization': {has_classical}")
    
    if has_classical:
        print("  ❌ PROBLEM FOUND: classical_optimization key exists in robust results!")
    else:
        print("  ✓ No classical_optimization key found")
        
    print("\nChecking MCMC presence:")
    has_mcmc = "mcmc_optimization" in data.get("results", {})
    print(f"  has 'mcmc_optimization': {has_mcmc}")
    
    print("\nSave logic evaluation:")
    results = data.get("results", {})
    condition = (
        "classical_optimization" in results and 
        "mcmc_optimization" not in results and 
        "mcmc_summary" not in results and
        results.get("methods_used", []) == ["Classical"]
    )
    print(f"  Would create classical directory: {condition}")
    
    # Show the actual structure 
    print("\n=== ACTUAL RESULTS DATA ===")
    if "results" in data:
        print(json.dumps(data["results"], indent=2)[:1000] + "..." if len(json.dumps(data["results"], indent=2)) > 1000 else json.dumps(data["results"], indent=2))
else:
    print("Results file not found!")