#!/usr/bin/env python3
"""
Script to fix merge conflicts in Jupyter notebook
Keeps the "Stashed changes" version which contains the actual code
"""

import json
import sys

def fix_notebook_conflicts(input_file, output_file):
    """
    Remove git merge conflict markers and keep stashed changes version
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    in_conflict = False
    keep_section = True
    conflict_count = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        if '<<<<<<< Updated upstream' in line:
            # Start of conflict - skip upstream version
            in_conflict = True
            keep_section = False
            conflict_count += 1
            print(f"Found conflict #{conflict_count} at line {i+1}")
        elif '=======' in line and in_conflict:
            # Middle of conflict - start keeping stashed version
            keep_section = True
        elif '>>>>>>> Stashed changes' in line:
            # End of conflict
            in_conflict = False
            keep_section = True
        elif keep_section:
            # Keep this line
            cleaned_lines.append(lines[i].rstrip('\n'))

        i += 1

    # Join cleaned lines
    cleaned_content = '\n'.join(cleaned_lines)

    # Try to parse as JSON to validate
    try:
        notebook = json.loads(cleaned_content)

        # Save the properly formatted notebook
        with open(output_file, 'w') as f:
            json.dump(notebook, f, indent=1)

        print(f"\nSuccess! Resolved {conflict_count} conflicts")
        print(f"Cleaned notebook saved to: {output_file}")

        # Verify notebook structure
        print(f"\nNotebook structure:")
        print(f"  - Number of cells: {len(notebook.get('cells', []))}")
        print(f"  - Kernel: {notebook.get('metadata', {}).get('kernelspec', {}).get('display_name', 'Unknown')}")

        return True

    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON after conflict resolution")
        print(f"JSON Error: {e}")

        # Save the raw cleaned content for debugging
        debug_file = output_file.replace('.ipynb', '_debug.txt')
        with open(debug_file, 'w') as f:
            f.write(cleaned_content)
        print(f"Raw cleaned content saved to: {debug_file}")

        return False

if __name__ == "__main__":
    input_file = "/Users/harsha/Desktop/deepst/DeepST/run_deepst.ipynb"
    output_file = "/Users/harsha/Desktop/deepst/DeepST/run_deepst_fixed.ipynb"

    if fix_notebook_conflicts(input_file, output_file):
        print("\n✓ Notebook successfully repaired!")
        print("You can now open run_deepst_fixed.ipynb in Jupyter")
    else:
        print("\n✗ Failed to repair notebook")
        print("Please check the debug file for issues")