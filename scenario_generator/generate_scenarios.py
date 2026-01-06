#!/usr/bin/env python3
"""
Scenario Generation Script

Generates N validated driving scenarios with:
- Natural language description generation
- Full pipeline execution
- Deep scene validation
- Retry on failure
- Diversity tracking
- Clean progress output

Usage:
    python generate_scenarios.py --count 10
    python generate_scenarios.py --count 5 --categories "Unprotected Left Turn" "Lane Change Negotiation"
    python generate_scenarios.py --count 20 --output-dir my_scenarios --verbose
"""

import sys
from pathlib import Path

# Add package path
sys.path.insert(0, str(Path(__file__).parent))

from scenario_generator import (
    GenerationLoop,
    GenerationLoopConfig,
    get_feasible_categories,
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate validated driving scenarios with retry logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 scenarios across all feasible categories
  python generate_scenarios.py --count 10

  # Generate 5 scenarios for specific categories
  python generate_scenarios.py --count 5 --categories "Unprotected Left Turn" "Lane Change Negotiation"

  # Generate with verbose output (shows pipeline details)
  python generate_scenarios.py --count 10 --verbose

  # Generate with custom output directory
  python generate_scenarios.py --count 10 --output-dir my_scenarios

  # Show pipeline output (not muted)
  python generate_scenarios.py --count 5 --show-pipeline

  # List available categories
  python generate_scenarios.py --list-categories

Feasible Categories:
""" + "\n".join(f"  - {c}" for c in get_feasible_categories())
    )
    
    parser.add_argument(
        "--count", "-n", type=int, default=10,
        help="Number of valid scenarios to generate (default: 10)"
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Specific categories to generate from (default: all feasible)"
    )
    parser.add_argument(
        "--difficulties", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Difficulty levels to use (default: 1-5)"
    )
    parser.add_argument(
        "--output-dir", default="log",
        help="Output directory for results (default: log)"
    )
    parser.add_argument(
        "--viz-objects", dest="viz_objects", action="store_true", default=True,
        help="Enable scene_objects.png visualization output (default: enabled)"
    )
    parser.add_argument(
        "--no-viz-objects", dest="viz_objects", action="store_false",
        help="Disable scene_objects.png visualization output"
    )
    parser.add_argument(
        "--routes-out-dir", default="routes",
        help="Directory to write routes output (default: routes)"
    )
    parser.add_argument(
        "--routes-ego-num", type=int, default=None,
        help="Override ego vehicle count for route conversion (default: auto-detect)"
    )
    parser.add_argument(
        "--town", default="Town05",
        help="CARLA town to use (default: Town05)"
    )
    parser.add_argument(
        "--carla-host", type=str, default="127.0.0.1",
        help="CARLA server host for route alignment (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--carla-port", type=int, default=2000,
        help="CARLA server port for route alignment (default: 2000)"
    )
    parser.add_argument(
        "--no-align-routes", action="store_true",
        help="Disable route alignment (skip CARLA GlobalRoutePlanner step)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Maximum retries per scenario before moving on (default: 5)"
    )
    parser.add_argument(
        "--ir-repair-passes", type=int, default=1,
        help="LLM repair attempts for IR JSON parsing (default: 1)"
    )
    parser.add_argument(
        "--no-stage-repairs", action="store_true",
        help="Skip pipeline stage repairs (just regenerate text on failure)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum validation score to accept a scenario (default: 0.6)"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.7,
        help="Maximum similarity allowed between scenarios (default: 0.7)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed debug output"
    )
    parser.add_argument(
        "--show-pipeline", action="store_true",
        help="Show internal pipeline output (not muted)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-32B-Instruct-AWQ)"
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="List all feasible categories and exit"
    )
    parser.add_argument(
        "--debug-ir", action="store_true",
        help="Print IR extraction details (raw LLM response, parsed JSON, extracted data)"
    )
    
    args = parser.parse_args()
    
    # List categories mode
    if args.list_categories:
        print("\nFeasible Categories:")
        print("=" * 40)
        for cat in get_feasible_categories():
            print(f"  • {cat}")
        print("\nUse --categories to select specific categories")
        return
    
    # Validate categories
    feasible = get_feasible_categories()
    if args.categories:
        invalid = [c for c in args.categories if c not in feasible]
        if invalid:
            print(f"❌ Invalid categories: {invalid}")
            print(f"\nValid categories are:")
            for cat in feasible:
                print(f"  • {cat}")
            sys.exit(1)
    
    # Create configuration
    config = GenerationLoopConfig(
        target_count=args.count,
        categories=args.categories,
        difficulties=args.difficulties,
        output_dir=args.output_dir,
        town=args.town,
        max_retries_per_scenario=args.max_retries,
        min_validation_score=args.min_score,
        similarity_threshold=args.similarity_threshold,
        mute_pipeline=not args.show_pipeline,
        model_id=args.model,
        verbose=args.verbose,
        viz_objects=args.viz_objects,
        routes_out_dir=args.routes_out_dir,
        routes_ego_num=args.routes_ego_num,
        debug_ir=args.debug_ir,
        ir_repair_passes=args.ir_repair_passes,
        stage_repairs=not args.no_stage_repairs,
        align_routes=not args.no_align_routes,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
    )
    
    print()
    print("=" * 60)
    print("Scenario Generation")
    print("=" * 60)
    print(f"Model:            {config.model_id}")
    print(f"Target count:     {config.target_count}")
    print(f"Categories:       {len(args.categories or feasible)} selected")
    print(f"Difficulties:     {config.difficulties}")
    print(f"Max retries:      {config.max_retries_per_scenario}")
    print(f"IR repairs:       {config.ir_repair_passes}")
    print(f"Min score:        {config.min_validation_score}")
    print(f"Output dir:       {config.output_dir}")
    print(f"Viz objects:      {config.viz_objects}")
    print(f"Routes out dir:   {config.routes_out_dir}")
    print(f"Align routes:     {config.align_routes} (CARLA {config.carla_host}:{config.carla_port})")
    print("=" * 60)
    print()
    
    # Run generation loop
    loop = GenerationLoop(config)
    results = loop.run()
    
    # Final summary
    print()
    print("=" * 60)
    print("📋 Generation Summary")
    print("=" * 60)
    print(f"✅ Successful:        {len(results)}")
    print(f"❌ Failed:            {len(loop.failed_scenarios)}")
    print(f"🔄 Total attempts:    {loop.total_attempts}")
    print(f"🔧 Pipeline runs:     {loop.total_pipeline_runs}")
    print(f"⚠️  Validation fails:  {loop.total_validation_failures}")
    print()
    print(f"📁 Results saved to: {args.output_dir}/")
    print(f"   - generated_scenarios.json  (scenario texts by category)")
    print(f"   - generation_results.json   (detailed results)")
    print(f"   - generation_log.txt        (generation log)")
    print()
    
    if results:
        print("Sample generated scenarios:")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. [{r.category}] {r.text[:60]}...")
    print()


if __name__ == "__main__":
    main()
