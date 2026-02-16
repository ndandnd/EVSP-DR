#%%
# 
import pandas as pd

def generate_practice_buses(input_file, x_buses):
    """
    Creates a practice CSV from vehicle details.
    - Selects the first x unique VehicleTasks.
    - Filters for 'Regular' trips only.
    - Adds a count_trip_id column starting from 0.
    """
    # Load the source data
    df = pd.read_csv(input_file)
    
    # 1. Identify the first x unique VehicleTasks
    unique_tasks = df['VehicleTask'].unique()
    selected_tasks = unique_tasks[:x_buses]
    
    # 2. Filter for selected tasks and only keep 'Regular' rows
    # This automatically removes prep-out, pull-out, deadhead, etc.
    filtered_df = df[(df['VehicleTask'].isin(selected_tasks)) & (df['Identifier'] == 'Regular')].copy()
    
    # 3. Create the count_trip_id starting from 0 for these rows
    filtered_df['count_trip_id'] = range(len(filtered_df))
    
    # 4. Select the specific columns requested
    columns_to_keep = [
        'Identifier', 
        'From1', 
        'Start1', 
        'End1', 
        'To1', 
        'Distance1', 
        'Usage kWh', 
        'count_trip_id', 
        'Ordered_Trip_ID'
    ]
    
    # Final selection (ensuring columns exist)
    result_df = filtered_df[[col for col in columns_to_keep if col in filtered_df.columns]]
    
    # Formatting: Convert IDs to integers (removes .0 from the display)
    if 'Ordered_Trip_ID' in result_df.columns:
        result_df.loc[:, 'Ordered_Trip_ID'] = result_df['Ordered_Trip_ID'].astype(int)
    
    # 5. Save to CSV
    output_filename = f'Practice_{x_buses}bus.csv'
    result_df.to_csv(output_filename, index=False)
    
    print(f"File '{output_filename}' generated successfully.")
    return output_filename

# --- EXECUTION ---
# Change x to the number of unique buses you want
x = 1
generate_practice_buses('Par_VehicleDetails_Updated.csv', x)
# %%
