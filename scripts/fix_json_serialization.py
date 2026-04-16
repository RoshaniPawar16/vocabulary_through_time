#!/usr/bin/env python3
"""
FIX JSON SERIALIZATION - Quick fix for the bool_ serialization issue
"""

import json
import numpy as np
from pathlib import Path

def fix_json_serialization():
    """Fix the JSON serialization issue in the main pipeline"""
    
    # Read the main pipeline file
    pipeline_file = Path("semantic_shift_analysis/main_pipeline.py")
    
    if not pipeline_file.exists():
        print("❌ Main pipeline file not found")
        return False
    
    try:
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Add a custom JSON encoder class
        json_encoder_fix = '''
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
'''
        
        # Replace the json.dump call
        old_dump = 'json.dump(comprehensive_results, f, indent=2)'
        new_dump = 'json.dump(comprehensive_results, f, indent=2, cls=NumpyEncoder)'
        
        # Add the encoder class before the class definition
        if 'class NumpyEncoder' not in content:
            # Find the class definition
            class_start = content.find('class IntegratedTemporalPipeline:')
            if class_start != -1:
                # Insert the encoder class before the main class
                content = content[:class_start] + json_encoder_fix + '\n' + content[class_start:]
        
        # Replace the json.dump call
        content = content.replace(old_dump, new_dump)
        
        # Write the fixed content
        with open(pipeline_file, 'w') as f:
            f.write(content)
        
        print("✅ Fixed JSON serialization issue")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing JSON serialization: {e}")
        return False

if __name__ == "__main__":
    fix_json_serialization() 