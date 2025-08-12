"""
Configuration Creator for Homodyne Analysis
==========================================

Helper script to create and customize configuration files from templates.
"""

import json
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def create_config_from_template(template_type="minimal", output_file="my_config.json", 
                               sample_name=None, experiment_name=None, author=None):
    """
    Create a configuration file from a template.
    
    Parameters
    ----------
    template_type : str
        Template to use: "minimal" or "complete"
    output_file : str
        Output configuration file name
    sample_name : str, optional
        Sample name to use in paths
    experiment_name : str, optional
        Experiment description
    author : str, optional
        Author name
    """
    
    # Get template path
    template_dir = Path(__file__).parent
    if template_type == "minimal":
        template_file = template_dir / "config_minimal_template.json"
    elif template_type == "complete":
        template_file = template_dir / "config_template.json"
    else:
        raise ValueError("template_type must be 'minimal' or 'complete'")
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    
    # Load template
    with open(template_file, 'r') as f:
        config = json.load(f)
    
    # Remove template-specific fields from final config
    if "_template_info" in config:
        del config["_template_info"]
    
    # Apply customizations
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if "metadata" in config:
        config["metadata"]["created_date"] = current_date
        config["metadata"]["updated_date"] = current_date
        
        if experiment_name:
            config["metadata"]["description"] = experiment_name
        
        if author:
            config["metadata"]["authors"] = [author]
    
    if sample_name and "experimental_data" in config:
        config["experimental_data"]["data_folder_path"] = f"./data/{sample_name}/"
        if "cache_file_path" in config["experimental_data"]:
            config["experimental_data"]["cache_file_path"] = f"./data/{sample_name}/"
    
    # Save configuration
    output_path = Path(output_file)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuration created: {output_path.absolute()}")
    
    # Provide next steps
    print("\nNext steps:")
    print(f"1. Edit {output_path} and customize the parameters for your experiment")
    print("2. Replace placeholder values (YOUR_*) with actual values")
    print("3. Adjust initial_parameters.values based on your system")
    print("4. Run analysis with: python run_homodyne.py --config", output_path)
    
    if template_type == "minimal":
        print(f"\nFor advanced options, see: {template_dir / 'config_template.json'}")
        print(f"Documentation: {template_dir / 'CONFIG_TEMPLATES_README.md'}")


def main():
    """Command-line interface for config creation."""
    parser = argparse.ArgumentParser(
        description="Create homodyne analysis configuration from templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_config.py --output my_experiment.json
  python create_config.py --template complete --sample protein_01 --author "Your Name"
  python create_config.py --template minimal --experiment "Protein dynamics under shear"
        """
    )
    
    parser.add_argument(
        "--template", "-t",
        choices=["minimal", "complete"],
        default="minimal",
        help="Template type to use (default: minimal)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="my_config.json",
        help="Output configuration file name (default: my_config.json)"
    )
    
    parser.add_argument(
        "--sample", "-s",
        help="Sample name (used in data paths)"
    )
    
    parser.add_argument(
        "--experiment", "-e", 
        help="Experiment description"
    )
    
    parser.add_argument(
        "--author", "-a",
        help="Author name"
    )
    
    args = parser.parse_args()
    
    try:
        create_config_from_template(
            template_type=args.template,
            output_file=args.output,
            sample_name=args.sample,
            experiment_name=args.experiment,
            author=args.author
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())