import sys
import os
import re

def save_output_to_file(model_name, prompt_type, task_filename, iteration, result):
    # Define regular expression pattern to extract the behavior tree from the complete output
    pattern = r'<root .*?</root>'
    matches = re.findall(pattern, result, re.DOTALL)

    if matches:
        final_tree = matches[-1]
        print(final_tree)

        # Create the output directory if it doesn't exist
        output_dir = "../workspace/src/bt_client/bt_xml"
        os.makedirs(output_dir, exist_ok=True)

        # Construct the filename
        base_filename = os.path.splitext(os.path.basename(task_filename))[0]
        filename = f"{model_name}-{prompt_type}-{base_filename}-{iteration}.xml"
        file_path = os.path.join(output_dir, filename)

        # Write the result to the file
        with open(file_path, "w") as file:
            file.write(final_tree)
        
        print(f"Output saved to {file_path}")
    else:
        print("No valid behavior tree found in the output.")