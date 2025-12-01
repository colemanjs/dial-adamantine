import vtk
import xml.etree.ElementTree as ET
import numpy as np
import os
import concurrent.futures
from functools import partial

def get_timesteps_and_files(pvd_file):
    """
    Parses the .pvd file to extract timesteps and corresponding .pvtu file paths.
    """
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    collection = root.find('Collection')
    
    timesteps = []
    files = []
    
    pvd_directory = os.path.dirname(pvd_file)

    for dataset in collection.findall('DataSet'):
        timesteps.append(float(dataset.get('timestep')))
        file_path = os.path.join(pvd_directory, dataset.get('file'))
        files.append(file_path)
        
    return np.array(timesteps), files

def probe_data(file_path, probe_locations, mode='interpolate', variable_name='temperature'):
    """
    Reads a .pvtu file and probes data at specified locations.

    Args:
        file_path (str): Path to the .pvtu file.
        probe_locations (list of tuples): List of (x, y, z) coordinates to probe.
        mode (str): Probing method. Can be 'interpolate' or 'nearest_centroid'.
        variable_name (str): The name of the data array to probe (e.g., 'T', 'Pressure').
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return [np.nan] * len(probe_locations)

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()

    # --- Mode 1: Interpolate at the probe location ---
    if mode == 'interpolate':
        points_to_probe = vtk.vtkPoints()
        for loc in probe_locations:
            points_to_probe.InsertNextPoint(loc)

        poly_data_to_probe = vtk.vtkPolyData()
        poly_data_to_probe.SetPoints(points_to_probe)

        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(data)
        probe.SetInputData(poly_data_to_probe)
        probe.Update()

        probed_data = probe.GetOutput().GetPointData().GetArray(variable_name)
        
        if probed_data:
            return [probed_data.GetValue(i) for i in range(len(probe_locations))]
        else:
            print(f"Warning: '{variable_name}' data array not found in {file_path}")
            return [np.nan] * len(probe_locations)

    # --- Mode 2: Find the value of the nearest cell ---
    elif mode == 'nearest_centroid':
        # Convert point data to cell data (by averaging point values)
        pd_to_cd = vtk.vtkPointDataToCellData()
        pd_to_cd.SetInputData(data)
        pd_to_cd.PassPointDataOff()
        pd_to_cd.Update()
        
        data_with_cell_data = pd_to_cd.GetOutput()
        cell_data_array = data_with_cell_data.GetCellData().GetArray(variable_name)

        if not cell_data_array:
            print(f"Warning: '{variable_name}' data array not found in {file_path}")
            return [np.nan] * len(probe_locations)

        # Create a cell locator for efficient searching
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(data_with_cell_data)
        cell_locator.BuildLocator()
        
        probed_values = []
        # These are dummy variables required by the vtk function call
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        for loc in probe_locations:
            cell_locator.FindClosestPoint(loc, closest_point, cell_id, sub_id, dist2)
            value = cell_data_array.GetValue(cell_id)
            probed_values.append(value)
            
        return probed_values

    else:
        raise ValueError("Invalid probe mode. Choose 'interpolate' or 'nearest_centroid'.")


def main():
    """
    Main function to run the post-processing.
    """
    # --- USER INPUT ---
    pvd_file = 'workspace/output.pvd'
    
    # Set the number of parallel workers. None uses all available CPU cores.
    MAX_WORKERS = None
    
    # Set probe interpolation logic and vairable name
    PROBE_MODE = 'interpolate'
    VARIABLE_NAME = 'temperature'
        
    # Define probe locations (x, y, z)
    probe_locations = [
        (0.0262,  0.0184,  0.0254),
        (0.0762,  0.0184,  0.0254),
        (0.1262,  0.0184,  0.0254)
    ]

    print(f"Probe Mode: {PROBE_MODE}")
    print(f"Variable to Probe: {VARIABLE_NAME}")
    
    timesteps, files = get_timesteps_and_files(pvd_file)
    
    # Create a parallel task function to pass constant arguements to workers
    task_function = partial(probe_data, 
                            probe_locations=probe_locations, 
                            mode=PROBE_MODE, 
                            variable_name=VARIABLE_NAME)
    
    all_probed_values = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print(f"Distributing {len(files)} files to worker processes...")
        all_probed_values = list(executor.map(task_function, files))
        print("...All parallel tasks completed. Assembling results.")

    # Re-organize the results from a list of rows to a dictionary of columns
    probe_results = {f'probe_{i}': [] for i in range(len(probe_locations))}
    for single_timestep_results in all_probed_values:
        for i, value in enumerate(single_timestep_results):
            probe_results[f'probe_{i}'].append(value)

    # --- Print and Save Results ---
    print("\n--- Probe Results ---")
    print(f"{'Time':<10}", end="")
    for i in range(len(probe_locations)):
        print(f"Probe {i} @ {probe_locations[i]}".ljust(30), end="")
    print()

    for i, time in enumerate(timesteps):
        print(f"{time:<10.4f}", end="")
        for j in range(len(probe_locations)):
            print(f"{probe_results[f'probe_{j}'][i]:<30.6f}", end="")
        print()
        
    output_filename = 'probe_data.csv'
    with open(output_filename, 'w') as f:
        probe_headers = [
            f"Probe_{i}_({loc[0]}_{loc[1]}_{loc[2]})"
            for i, loc in enumerate(probe_locations)
        ]
        header = "Time," + ",".join(probe_headers)
        f.write(header + '\n')

        for i, time in enumerate(timesteps):
            results_for_timestep = [
                str(probe_results[f'probe_{j}'][i])
                for j in range(len(probe_locations))
            ]
            line = f"{time}," + ",".join(results_for_timestep)
            f.write(line + '\n')
            
    print(f"\nProbe data saved to {output_filename}")

if __name__ == '__main__':
    main()
