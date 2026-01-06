#!/usr/bin/env python3
"""
Example: Generate Multi-Agent Scenarios

Demonstrates using the scenario generator with:
- Research context for domain knowledge
- Chain-of-thought reasoning
- Optimized sampling for diversity

Usage:
    python example_generate.py
    python example_generate.py --no-research-context
    python example_generate.py --no-chain-of-thought
"""

import argparse
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))

from scenario_generator import (
    ScenarioGenerator,
    GenerationConfig,
    CATEGORY_FEASIBILITY,
    get_feasible_categories,
)
from scenario_generator.generator import (
    build_generation_prompt,
    build_system_prompt,
    select_variation_values,
)


def demo_prompt_construction():
    """Show what prompts look like with different settings."""
    print("=" * 70)
    print("PROMPT CONSTRUCTION DEMO")
    print("=" * 70)
    
    category = "Courtesy & Deadlock Negotiation"
    cat_info = CATEGORY_FEASIBILITY[category]
    
    # Show system prompt without research context
    print("\n--- SYSTEM PROMPT (without research context) ---")
    short_prompt = build_system_prompt(include_research_context=False)
    print(f"Length: {len(short_prompt)} chars")
    print(short_prompt[:500] + "..." if len(short_prompt) > 500 else short_prompt)
    
    # Show system prompt with research context
    print("\n--- SYSTEM PROMPT (with research context) ---")
    full_prompt = build_system_prompt(include_research_context=True)
    print(f"Length: {len(full_prompt)} chars")
    print(f"(Research context adds {len(full_prompt) - len(short_prompt)} chars)")
    
    # Show generation prompt with chain-of-thought
    print("\n--- GENERATION PROMPT (with chain-of-thought) ---")
    forced = select_variation_values(cat_info)
    gen_prompt = build_generation_prompt(
        category=category,
        difficulty=4,
        cat_info=cat_info,
        existing_scenarios=["Example scenario 1...", "Example scenario 2..."],
        forced_variations=forced,
    )
    print(gen_prompt)
    
    print("\n" + "=" * 70)


def generate_examples(
    num_examples: int = 3,
    include_research: bool = True,
    use_cot: bool = True,
):
    """Generate example scenarios with specified settings."""
    
    print("=" * 70)
    print("SCENARIO GENERATION")
    print("=" * 70)
    print(f"Settings:")
    print(f"  Research context: {include_research}")
    print(f"  Chain-of-thought: {use_cot}")
    print(f"  Examples to generate: {num_examples}")
    print()
    
    # Create config with specified settings
    config = GenerationConfig(
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.15,
        max_new_tokens=1024 if use_cot else 512,
        include_research_context=include_research,
        use_chain_of_thought=use_cot,
    )
    
    # Create generator
    generator = ScenarioGenerator(config)
    
    # Categories that are especially good for multi-agent scenarios
    priority_categories = [
        "Multi-Way Standoff",
        "Courtesy & Deadlock Negotiation",
        "Platoon Merge Conflict",
        "Narrow Passage Negotiation",
        "Unprotected Left Turn",
    ]
    
    # Filter to only feasible categories
    feasible = set(get_feasible_categories())
    categories = [c for c in priority_categories if c in feasible]
    
    print(f"Using categories: {categories[:3]}")
    print()
    
    for i in range(num_examples):
        category = categories[i % len(categories)]
        difficulty = 4 + (i % 2)  # Alternate between 4 and 5
        
        print(f"\n--- Example {i+1}: {category} (difficulty {difficulty}) ---")
        
        try:
            scenario_text, validation = generator.generate_scenario(category, difficulty)
            
            if scenario_text:
                print("✅ Generated successfully")
                print(f"Text: {scenario_text[:200]}...")
            else:
                errors = (
                    validation.structural_errors +
                    validation.semantic_errors +
                    validation.parse_errors
                )
                print(f"❌ Failed: {errors[:2] if errors else 'Unknown'}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Demo scenario generation with research context and chain-of-thought"
    )
    parser.add_argument(
        "--demo-prompts", action="store_true",
        help="Show prompt construction demo (no generation)"
    )
    parser.add_argument(
        "--no-research-context", action="store_true",
        help="Disable research context in system prompt"
    )
    parser.add_argument(
        "--no-chain-of-thought", action="store_true",
        help="Disable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=3,
        help="Number of examples to generate"
    )
    
    args = parser.parse_args()
    
    if args.demo_prompts:
        demo_prompt_construction()
    else:
        generate_examples(
            num_examples=args.num,
            include_research=not args.no_research_context,
            use_cot=not args.no_chain_of_thought,
        )


if __name__ == "__main__":
    main()
