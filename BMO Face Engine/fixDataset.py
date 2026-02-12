import re

def remove_variable(input_file, output_file, variable_index):
    # variable_index is 0-based (12th variable = index 11)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    
    for line in lines:
        # Regex to find the content inside the curly braces { ... }
        # Group 1: Prefix (faces["name"] = { )
        # Group 2: The numbers
        # Group 3: Suffix ( }; )
        match = re.search(r'(.*\{)(.*)(\}.*)', line)
        
        if match:
            prefix = match.group(1)
            content = match.group(2)
            suffix = match.group(3)
            
            # Split the numbers by comma
            # We strip whitespace to handle the formatting cleanly
            values = [v.strip() for v in content.split(',')]
            
            # Filter out any empty strings resulting from trailing commas
            values = [v for v in values if v]
            
            if len(values) > variable_index:
                # REMOVE THE VARIABLE
                removed_val = values.pop(variable_index)
                
                # Reconstruct the string
                # We add the 'f' back implicitly because we didn't strip it, 
                # we just split by comma.
                new_content = ", ".join(values)
                
                # Reassemble the line
                new_lines.append(f"{prefix} {new_content} {suffix}\n")
            else:
                # If a line is too short (weird error), keep it as is
                new_lines.append(line)
        else:
            # Keep comments or empty lines
            new_lines.append(line)
            
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Success! Removed variable at index {variable_index} (12th position).")
    print(f"Saved to {output_file}")

# Run the function: Remove index 11 (the 12th number)
remove_variable('face_database.txt', 'face_database_fixed.txt', 11)