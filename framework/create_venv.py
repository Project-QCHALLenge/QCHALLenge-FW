import os
import re
import sys
import subprocess
from pathlib import Path


def create_virtualenv(venv_dir):
    """Erstellt eine virtuelle Umgebung."""
    if not venv_dir.exists():
        subprocess.check_call([sys.executable, '-m', 'venv', str(venv_dir)])
    print(f"Virtuelle Umgebung unter {venv_dir} erstellt.")


def merge_requirements(data, output_file="merged_requirements.txt"):
    """
    Merges all requirements files into a single file, ensuring:
    - The newest version for each dependency is used for '>=' constraints.
    - '==' is preserved if it occurs, and takes precedence over '>='.
    """
    # Unterstützt sowohl '==' als auch '>='
    dependency_pattern = re.compile(r"^([a-zA-Z0-9_\-]+)\s*(==|>=)\s*(\d+\.\d+[^\s]*)$")

    req_paths = [f"{use_case['folder']}/{use_case['requirements']}" for use_case in data["use_cases"]]
    latest_versions = {}

    for req_path in req_paths:
        req_path = Path(req_path)
        if req_path.exists():
            with req_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue  # Ignoriere leere Zeilen und Kommentare
                    match = dependency_pattern.match(line)
                    if match:
                        package, operator, version = match.groups()
                        existing = latest_versions.get(package)

                        if existing:
                            existing_operator, existing_version = existing
                            if operator == "==":
                                # '==' hat Vorrang
                                latest_versions[package] = (operator, version)
                            elif existing_operator == ">=":
                                # Vergleiche nur, wenn beides '>=' sind
                                if version > existing_version:
                                    latest_versions[package] = (operator, version)
                            # sonst: bestehendes '==' bleibt erhalten
                        else:
                            latest_versions[package] = (operator, version)
        else:
            print(f"Warnung: {req_path} existiert nicht.")

    # Schreibe die zusammengeführten Anforderungen
    with open(output_file, "w") as f:
        for package, (operator, version) in sorted(latest_versions.items()):
            f.write(f"{package} {operator} {version}\n")

    return output_file


def install_requirements(venv_dir, data):
    """Installs all requirements from the merged requirements file in the virtual environment."""

    pip_path = venv_dir / 'bin' / 'pip'
    python_path = venv_dir / 'bin' / 'python'

    # Fallback for Windows (pip.exe and python.exe)
    if not pip_path.exists():
        pip_path = venv_dir / 'Scripts' / 'pip.exe'

    if not python_path.exists():
        python_path = venv_dir / 'Scripts' / 'python.exe'

    # Upgrade pip
    subprocess.check_call([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Merge all requirements into one file
    merged_requirements_file = merge_requirements(data, output_file="merged_requirements.txt")

    # Install from the merged requirements file
    subprocess.check_call([str(pip_path), 'install', '-r', merged_requirements_file])


def print_activation_instructions(venv_dir):
    """Print the activation instructions for the virtual environment."""
    if os.name == 'nt':  # Windows
        activate_command = f"{venv_dir / 'Scripts' / 'activate'}"
    else:  # Unix or MacOS
        activate_command = f"source {venv_dir / 'bin' / 'activate'}"

    print("\n=== Install completed successfully ===")
    print(f"To activate the virtual environment, run the following command in your terminal:")
    print(f"\n{activate_command}\n")


def create_and_activate_venv(data):
    venv_path = Path('venv')
    create_virtualenv(venv_path)
    install_requirements(venv_path, data)
    print_activation_instructions(venv_path)
    return venv_path