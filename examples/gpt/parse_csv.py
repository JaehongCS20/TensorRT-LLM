import pandas as pd
from io import StringIO

def _parse_nvtx(model, i, o, to_file=False):
    """
    Parses NVTX profiling data, trims unnecessary sections, and calculates the average latency
    for specified layers. Optionally saves the results to a file.

    Args:
        model (str): The model name used in the file naming convention.
        i (int): Input size identifier in the file name.
        o (int): Output size identifier in the file name.
        to_file (bool): Whether to save the trimmed data and results to a CSV file.

    Returns:
        dict: A dictionary with the layer names (keywords) as keys and their average latency values as values.
    """

    # Step 1: Define the file path and read the file lines
    file = f"profiled_result/{model}_analysis_i{i}_o{o}.csv"

    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Step 2: Trim NVTX information
    # Locate the header row containing "Time (%)"
    header_key = "Time (%)"
    header_index = next((i for i, line in enumerate(lines) if header_key in line), None)
    data_lines = lines[header_index:]  # Extract lines from the header onward

    # Locate the "Processing" keyword to trim unnecessary sections
    keyword = "Processing"
    keyword_index = next((i for i, line in enumerate(data_lines) if keyword in line), None)
    trimmed_lines = data_lines[:keyword_index] if keyword_index else data_lines  # Trim at the keyword

    # Convert the trimmed lines into a DataFrame
    trimmed_data = pd.read_csv(StringIO("".join(trimmed_lines)))

    # Step 3: Optionally save the trimmed data
    if to_file:
        extracted_file_path = f"profiled_result/{model}_i{i}_o{o}_trimmed.csv"
        trimmed_data.to_csv(extracted_file_path, index=False)

    # Step 4: Define the keywords for layer identification
    keywords = [
        "mlp/gelu", "ln_f", "vocab_embedding", "post_layernorm",
        "input_layernorm", "attention/wrapper", "attention/dense",
        "attention/qkv", "lm_head", "mlp/proj", "mlp/fc"
    ]

    # Step 5: Calculate the average latency for each layer
    result_df = pd.DataFrame()

    for keyword in keywords:
        # Filter rows containing the keyword in the 'Range' column
        filtered_rows = trimmed_data[trimmed_data['Range'].str.contains(keyword, case=False, na=False)]

        if not filtered_rows.empty:
            # Calculate the average latency for the 'Avg (ns)' column
            average_value = filtered_rows['Avg (ns)'].mean()
            # Create a new row with the keyword and average latency
            new_row = {'Range': keyword, 'Avg (ns)': average_value}
            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    # Step 6: Optionally save the results
    if to_file:
        average_result_file_path = f"profiled_result/{model}_i{i}_o{o}_parsed.csv"
        result_df.to_csv(average_result_file_path, index=False)

    # Step 7: Convert the results to a dictionary
    result_dict = result_df.set_index('Range')['Avg (ns)'].to_dict()
    print(result_dict)  # Print the result dictionary for debugging
    return result_dict



def make_perf_model(model, max_i, max_o, to_file=True):
    """
    Calculates latencies for each iteration given input size (i) and output size (o).

    Args:
        model (str): The model name used in the file naming convention.
        max_i (int): Maximum value for input size (i).
        max_o (int): Maximum value for output size (o).
        to_file (bool): Whether to save the results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing calculated latencies for each iteration.
    """
    results = [] # list of dict

    for i in range(1, max_i + 1):
        previous_latency = None # dict

        for o in range(1, max_o + 1):
            # Parse NVTX profiling data for the given i and o
            nvtx_data = _parse_nvtx(model, i, o, to_file=False)
            
            # Calculate latency for the current iteration
            for k, v in nvtx_data.items():
                if previous_latency is not None:
                    iteration_latency = (v * o - previous_latency[k]) / o
                    results.append({
                        "layer_name": k,
                        "input": 1,
                        "kv_cache": i+o-2,
                        "latency(ns)": iteration_latency
                    })
                else:
                    results.append({
                        "layer_name": k,
                        "input": i,
                        "kv_cache": 0,
                        "latency(ns)": v
                    })

            # Update previous latency for the next iteration
            previous_latency = nvtx_data

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV if required
    if to_file:
        results_file_path = f"perf_model/{model}.csv"
        results_df.to_csv(results_file_path, index=False)

    return results_df

make_perf_model('gpt', 2, 2, to_file=True)