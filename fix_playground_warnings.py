#!/usr/bin/env python3
"""
Fix warnings in playground.ipynb notebook.

This script fixes:
1. Pandas FutureWarning about groupby.apply() on grouping columns
2. Matplotlib UserWarning about missing emoji glyphs in font

Usage:
    python fix_playground_warnings.py
"""

import json

def fix_playground_warnings():
    """Fix the warnings in playground.ipynb"""
    
    notebook_path = "/Users/yangjiao/Documents/Projects/geo_test/playground.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("🔧 Fixing playground.ipynb warnings...")
    
    # Find and fix the pandas groupby warning
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']
            
            # Convert source to string if it's a list
            if isinstance(source_lines, list):
                source_text = ''.join(source_lines)
            else:
                source_text = source_lines
            
            # Fix 1: Pandas groupby warning
            if 'coverage_by_method = detailed_results.groupby' in source_text:
                print("  ✅ Found pandas groupby warning - fixing...")
                
                # Replace the problematic line
                old_pattern = "coverage_by_method = detailed_results.groupby(['assignment_method', 'reporting_method']).apply("
                new_pattern = "coverage_by_method = detailed_results.groupby(['assignment_method', 'reporting_method'], include_groups=False).apply("
                
                # Fix the line
                if isinstance(source_lines, list):
                    for i, line in enumerate(source_lines):
                        if old_pattern in line:
                            source_lines[i] = line.replace(old_pattern, new_pattern)
                            print(f"    Fixed line {i}: Added include_groups=False")
                else:
                    if old_pattern in source_text:
                        cell['source'] = source_text.replace(old_pattern, new_pattern)
                        print("    Fixed: Added include_groups=False to groupby")
            
            # Fix 2: Font warnings for emojis - replace emoji with text alternatives
            emoji_replacements = {
                '📊': '[BAR CHART]',
                '🏆': '[TROPHY]', 
                '💡': '[BULB]',
                '🔬': '[MICROSCOPE]',
                '⚡': '[LIGHTNING]',
                '🔄': '[ARROWS]',
                '✅': '[CHECK]',
                '❌': '[X]',
                '⚠️': '[WARNING]',
                '🎯': '[TARGET]',
                '🎉': '[PARTY]',
                '🚀': '[ROCKET]',
                '🧠': '[BRAIN]',
                '🧪': '[TEST TUBE]',
                '📈': '[CHART UP]',
                '📋': '[CLIPBOARD]',
                '💻': '[LAPTOP]',
                '🔧': '[WRENCH]',
                '📁': '[FOLDER]'
            }
            
            # Check for emojis in titles and labels that cause font warnings
            for emoji, replacement in emoji_replacements.items():
                if emoji in source_text and ('plt.title' in source_text or 'set_title' in source_text):
                    if isinstance(source_lines, list):
                        for i, line in enumerate(source_lines):
                            if emoji in line and ('plt.title' in line or 'set_title' in line):
                                source_lines[i] = line.replace(emoji, replacement)
                                print(f"    Fixed emoji in plot title: {emoji} -> {replacement}")
                    else:
                        if emoji in source_text and ('plt.title' in source_text or 'set_title' in source_text):
                            cell['source'] = source_text.replace(emoji, replacement)
                            print(f"    Fixed emoji in plot: {emoji} -> {replacement}")
    
    # Write the fixed notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Fixed playground.ipynb warnings!")
    print("  • Added include_groups=False to pandas groupby operations")
    print("  • Replaced emojis in plot titles with text alternatives")
    print("  • The warnings should no longer appear when running the notebook")

if __name__ == "__main__":
    fix_playground_warnings()