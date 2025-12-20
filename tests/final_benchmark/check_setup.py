"""
Test if evaluation pipeline is ready to run.
Quick diagnostic script to verify setup.
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

print("="*80)
print("FINAL BENCHMARK EVALUATION - SETUP CHECK")
print("="*80)
print()

errors = []
warnings = []

# Check 1: Python version
print("1. Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor >= 8:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    errors.append(f"Python 3.8+ required, got {version.major}.{version.minor}")

# Check 2: Required packages
print("\n2. Checking required packages...")
required_packages = {
    "matplotlib": "Plotting",
    "seaborn": "Statistical visualization",
    "numpy": "Numerical computing",
}

for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {package} - {description}")
    except ImportError:
        errors.append(f"Missing package: {package} ({description})")
        print(f"   ✗ {package} - MISSING")

# Check 3: Project imports
print("\n3. Checking project imports...")
project_imports = {
    "services.query.service": "QueryService",
    "shared.models.schemas": "Schema definitions",
    "shared.config.settings": "Settings",
}

for module, description in project_imports.items():
    try:
        __import__(module)
        print(f"   ✓ {module} - {description}")
    except ImportError as e:
        warnings.append(f"Cannot import {module}: {e}")
        print(f"   ⚠ {module} - Warning (may work at runtime)")

# Check 4: Configuration files
print("\n4. Checking configuration files...")
config_files = [
    "ablation_configs.py",
    "run_evaluations.py",
    "calculate_metrics.py",
    "generate_plots.py",
]

script_dir = Path(__file__).parent
for filename in config_files:
    filepath = script_dir / filename
    if filepath.exists():
        print(f"   ✓ {filename}")
    else:
        errors.append(f"Missing file: {filename}")

# Check 5: Output directories
print("\n5. Checking output directories...")
output_dirs = ["results", "plots"]
for dirname in output_dirs:
    dirpath = script_dir / dirname
    dirpath.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {dirname}/ created")

# Check 6: Environment
print("\n6. Checking environment variables...")
import os
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"   ✓ GROQ_API_KEY set ({groq_key[:15]}...{groq_key[-8:]})")
    print(f"   ✓ Key length: {len(groq_key)} characters")
else:
    errors.append("GROQ_API_KEY not set - check .env file")
    print("   ✗ GROQ_API_KEY not found")

# Summary
print("\n" + "="*80)
if errors:
    print("❌ SETUP INCOMPLETE - Fix these errors:")
    for error in errors:
        print(f"   • {error}")
    print("\nInstall missing packages:")
    print("   pip3 install matplotlib seaborn numpy")
elif warnings:
    print("⚠ SETUP MOSTLY COMPLETE - Check these warnings:")
    for warning in warnings:
        print(f"   • {warning}")
    print("\nYou can probably proceed, but verify:")
    print("   python run_evaluations.py --test fever_full_system")
else:
    print("✅ SETUP COMPLETE - Ready to run!")
    print("\nNext steps:")
    print("   1. python run_evaluations.py --test fever_full_system")
    print("   2. Check results/fever_full_system.json")
    print("   3. Continue with remaining tests")

print("="*80)
