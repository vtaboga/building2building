import os
import pandas as pd
import subprocess
import logging
import json
from typing import List, Dict
from src.generator.utils import get_counties_from_coords_batch

logger = logging.getLogger('generator')

def process_metadata(state: str):
    """
    Process all metadata CSV files in the metadata directory to get the county name for each row.
    For each file:
    1. Loads the CSV
    2. If County column exists, skip the file as it's already processed
    3. Parses the Centroid column to extract latitude and longitude
    4. Creates a new County column with the county name for each coordinate pair
    Uses async requests to optimize county lookups
    """
    csv_file = os.path.join("data", "metadata", f"{state}.csv")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Skip if County column already exists
        if 'County' in df.columns:
            logger.info(f"Skipping {csv_file} - already processed")
            return
        
        logger.info(f"Processing metadata for state: {state}")
        
        # Split the Centroid column into latitude and longitude
        df[['Latitude', 'Longitude']] = df['Centroid'].str.split('/', expand=True)
        
        # Convert to float
        df['Latitude'] = df['Latitude'].astype(float)
        df['Longitude'] = df['Longitude'].astype(float)
        
        # Process all coordinates at once
        logger.debug("Getting county names for all coordinates...")
        coords_list = list(zip(df['Latitude'], df['Longitude']))
        counties = get_counties_from_coords_batch(coords_list)
        
        # Add counties to dataframe
        df['County'] = counties
        
        # Save the updated CSV file
        df.to_csv(csv_file, index=False)
        logger.info(f"Successfully processed and updated {csv_file}")
            
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")
        raise

def find_transitioned_file(original_file: str, to_ver: str) -> str:
    """Helper function to find the transitioned file which might have different naming patterns"""
    idf_dir = os.path.dirname(original_file)
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    
    # Different possible patterns for the output file
    possible_patterns = [
        # Pattern 1: original_name.idfnew (actual pattern we're seeing)
        os.path.join(idf_dir, f"{base_name}.idfnew"),
        # Pattern 2: original_name-V{version}.idf
        os.path.join(idf_dir, f"{base_name}-V{to_ver.replace('.', '-')}.idf"),
        # Pattern 3: original_name.V{version}.idf
        os.path.join(idf_dir, f"{base_name}.V{to_ver.replace('.', '-')}.idf"),
        # Pattern 4: original_name-{version}.idf
        os.path.join(idf_dir, f"{base_name}-{to_ver.replace('.', '-')}.idf"),
        # Pattern 5: original_name.{version}.idf
        os.path.join(idf_dir, f"{base_name}.{to_ver}.idf"),
        # Pattern 6: original_name.new
        os.path.join(idf_dir, f"{base_name}.new"),
        # Pattern 7: The original file might have been modified in place
        original_file,
    ]
    
    for pattern in possible_patterns:
        if os.path.exists(pattern):
            logger.debug(f"Found matching file: {pattern}")
            return pattern
            
    # If no pattern matches, log all files in directory for debugging
    logger.warning(f"No matching transitioned file found. Available files in {idf_dir}:")
    for file in os.listdir(idf_dir):
        logger.debug(f"  {file}")
            
    return None

def transition_idf(idf_path: str, state: str, county: str, target_version: str = "24.1") -> str:
    """
    Transition an IDF file to the target EnergyPlus version using the transition executables.
    Saves the processed file in data/processed_idf/state/county/ directory.

    Args:
        idf_path (str): Path to the input IDF file
        state (str): Two-letter state code
        county (str): County name
        target_version (str): Target EnergyPlus version (default: "24.1")

    Returns:
        str: Path to the transitioned file in the processed_idf directory
    """

    # Create the output directory structure
    processed_dir = os.path.join("data", "processed_idf", state, county)
    os.makedirs(processed_dir, exist_ok=True)

    idf_dir = os.path.dirname(idf_path)
    idf_name = os.path.basename(idf_path)
    
    # Define the final output path
    final_output_path = os.path.join(processed_dir, idf_name)
    
    # Create a temporary directory for transition files
    temp_dir = os.path.join(idf_dir, "temp_transition")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy the original file to temp directory to work with
    working_file = os.path.join(temp_dir, idf_name)
    with open(idf_path, 'r') as src, open(working_file, 'w') as dst:
        dst.write(src.read())
    
    try:
        # Define the transition sequence from 9.4 to 24.1
        transitions = [
            "9.4.0-to-9.5.0",
            "9.5.0-to-9.6.0",
            "9.6.0-to-22.1.0",
            "22.1.0-to-22.2.0",
            "22.2.0-to-23.1.0",
            "23.1.0-to-23.2.0",
            "23.2.0-to-24.1.0"
        ]
        
        # Path to the transition executables directory
        energyplus_dir = "/usr/local/EnergyPlus-24-1-0"
        transition_dir = os.path.join(energyplus_dir, "PreProcess", "IDFVersionUpdater")
        
        for transition in transitions:
            from_ver, to_ver = transition.split("-to-")
            
            # Get paths for the transition executable and IDD files
            transition_exe = os.path.join(transition_dir, f"Transition-V{from_ver.replace('.', '-')}-to-V{to_ver.replace('.', '-')}")
            from_idd = os.path.join(transition_dir, f"V{from_ver.replace('.', '-')}-Energy+.idd")
            to_idd = os.path.join(transition_dir, f"V{to_ver.replace('.', '-')}-Energy+.idd")
            
            # Check if required files exist
            if not os.path.exists(transition_exe):
                logger.error(f"Error: Transition executable not found at {transition_exe}")
                return None
            if not os.path.exists(from_idd):
                logger.error(f"Error: Source IDD file not found at {from_idd}")
                return None
            if not os.path.exists(to_idd):
                logger.error(f"Error: Target IDD file not found at {to_idd}")
                return None
                
            # Create symbolic links to IDD files in the working directory
            current_dir = os.getcwd()
            from_idd_link = os.path.join(current_dir, os.path.basename(from_idd))
            to_idd_link = os.path.join(current_dir, os.path.basename(to_idd))
            
            try:
                if os.path.exists(from_idd_link):
                    os.remove(from_idd_link)
                if os.path.exists(to_idd_link):
                    os.remove(to_idd_link)
                    
                os.symlink(from_idd, from_idd_link)
                os.symlink(to_idd, to_idd_link)
                
                logger.info(f"\nRunning transition from {from_ver} to {to_ver}")
                logger.debug(f"Using: {transition_exe}")
                logger.debug(f"Input: {working_file}")
                
                result = subprocess.run(
                    [transition_exe, working_file],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={"DISPLAY": ""}
                )
                
                if result.stdout:
                    logger.debug("Transition output:", result.stdout)
                if result.stderr:
                    logger.warning("Transition errors:", result.stderr)
                
                # Clean up any .idfold files that might have been created
                old_file = working_file + "old"
                if os.path.exists(old_file):
                    os.remove(old_file)
                
                # Look for the transitioned file
                transitioned_file = find_transitioned_file(working_file, to_ver)
                if transitioned_file:
                    logger.debug(f"Found transitioned file at: {transitioned_file}")
                    if transitioned_file != working_file:
                        # Replace working file with transitioned file
                        with open(transitioned_file, 'r') as src, open(working_file, 'w') as dst:
                            dst.write(src.read())
                        # Remove the transitioned file after copying
                        os.remove(transitioned_file)
                else:
                    logger.error(f"Error: Could not find transitioned file for {working_file}")
                    return None
                    
            finally:
                # Clean up symbolic links
                if os.path.exists(from_idd_link):
                    os.remove(from_idd_link)
                if os.path.exists(to_idd_link):
                    os.remove(to_idd_link)
        
        # Copy the final file to the processed_idf directory instead of original location
        with open(working_file, 'r') as src, open(final_output_path, 'w') as dst:
            dst.write(src.read())
        
        logger.info(f"\nFinal transitioned file saved to: {final_output_path}")
        
        # Convert the final IDF to epJSON
        epjson_path = convert_to_epjson(final_output_path)
        if epjson_path:
            return epjson_path
        else:
            logger.error("Failed to convert to epJSON format")
            return final_output_path  # Return IDF path as fallback
        
    finally:
        # Clean up temporary directory and additional files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {file}: {e}")
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")
        
        # Clean up Energy+.ini and Transition.audit files
        cleanup_files = ['Energy+.ini', 'Transition.audit']
        for file in cleanup_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Removed {file}")
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")
    
    return final_output_path

def process_idf(idf_files: List[int], state: str, county: str):
    """
    Process IDF files if they haven't been processed already.
    
    Args:
        idf_files (List[int]): List of IDF file IDs
        state (str): Two-letter state code
        county (str): County name
    
    Returns:
        List[str]: List of paths to processed IDF files
    """
    processed_dir = os.path.join("data", "processed_idf", state, county)
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_paths = []
    for idf_id in idf_files:
        # Check if processed file already exists
        processed_path = os.path.join(processed_dir, f"{idf_id}.idf")
        if os.path.exists(processed_path):
            logger.info(f"Skipping {idf_id}.idf - already processed")
            processed_paths.append(processed_path)
            continue
            
        # Get path to original file
        original_path = os.path.join("data", "idf", f"{state}_{county}_IDF", f"{idf_id}.idf")
        if not os.path.exists(original_path):
            logger.warning(f"Warning: Original file not found at {original_path}")
            continue
            
        # Process the file
        processed_path = transition_idf(original_path, state, county)
        if processed_path:
            processed_paths.append(processed_path)
            
    return processed_paths

def convert_to_epjson(idf_path: str) -> str:
    """
    Convert an IDF file to epJSON format using EnergyPlus converter.
    Deletes the original IDF file after successful conversion.

    Args:
        idf_path (str): Path to the input IDF file

    Returns:
        str: Path to the converted epJSON file, or None if conversion fails
    """
    try:
        # Define the output path
        epjson_path = os.path.splitext(idf_path)[0] + '.epJSON'
        
        # Path to the EnergyPlus executable
        energyplus_dir = "/usr/local/EnergyPlus-24-1-0"
        converter = os.path.join(energyplus_dir, "ConvertInputFormat")
        
        logger.info(f"Converting {idf_path} to epJSON format")
        
        # Run the conversion
        result = subprocess.run(
            [converter, idf_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stderr:
            logger.warning(f"Conversion warnings: {result.stderr}")
            
        # Check if the epJSON file was created
        if os.path.exists(epjson_path):
            logger.info(f"Successfully converted to: {epjson_path}")
            # Delete the original IDF file
            os.remove(idf_path)
            logger.debug(f"Deleted original IDF file: {idf_path}")
            return epjson_path
        else:
            logger.error(f"epJSON file not created at expected path: {epjson_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return None
    

def add_hvac_meters_to_epjson(epjson_path: str, output_path: str = None) -> None:
    """
    Check if HVAC energy consumption meters exist in an epJSON file.
    If not, add the meters and save the modified epJSON.
    
    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path
    
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
    
    # Check if we already have the necessary meters
    has_elec_hvac_meter = False
    has_gas_hvac_meter = False
    
    # First, check existing Output:Meter objects
    if "Output:Meter" in epjson:
        for meter_key, meter_data in epjson["Output:Meter"].items():
            if meter_data.get("key_name") == "Electricity:HVAC" and meter_data.get("reporting_frequency") == "Timestep":
                has_elec_hvac_meter = True
                print("Found existing Electricity:HVAC meter with Timestep reporting.")
            
            if meter_data.get("key_name") == "NaturalGas:HVAC" and meter_data.get("reporting_frequency") == "Timestep":
                has_gas_hvac_meter = True
                print("Found existing NaturalGas:HVAC meter with Timestep reporting.")
    
    # Check Output:Meter:MeterFileOnly objects as well
    if "Output:Meter:MeterFileOnly" in epjson:
        for meter_key, meter_data in epjson["Output:Meter:MeterFileOnly"].items():
            if meter_data.get("key_name") == "Electricity:HVAC" and meter_data.get("reporting_frequency") == "Timestep":
                has_elec_hvac_meter = True
                print("Found existing Electricity:HVAC meter file only with Timestep reporting.")
            
            if meter_data.get("key_name") == "NaturalGas:HVAC" and meter_data.get("reporting_frequency") == "Timestep":
                has_gas_hvac_meter = True
                print("Found existing NaturalGas:HVAC meter file only with Timestep reporting.")
    
    # Add meters if they don't exist
    modified = False
    
    # Make sure the Output:Meter category exists
    if "Output:Meter" not in epjson:
        epjson["Output:Meter"] = {}
    
    # Add electricity HVAC meter if needed
    if not has_elec_hvac_meter:
        new_meter_name = f"Output:Meter:ElectricityHVAC"
        epjson["Output:Meter"][new_meter_name] = {
            "key_name": "Electricity:HVAC",
            "reporting_frequency": "Timestep"
        }
        print(f"Added Electricity:HVAC meter with Timestep reporting.")
        modified = True
    
    # Add natural gas HVAC meter if needed
    if not has_gas_hvac_meter:
        new_meter_name = f"Output:Meter:NaturalGasHVAC"
        epjson["Output:Meter"][new_meter_name] = {
            "key_name": "NaturalGas:HVAC",
            "reporting_frequency": "Timestep"
        }
        print(f"Added NaturalGas:HVAC meter with Timestep reporting.")
        modified = True
    
    # Save the modified epJSON if changes were made
    if modified:
        with open(output_path, 'w') as f:
            json.dump(epjson, f, indent=2)
        print(f"Modified epJSON saved to {output_path}")
    else:
        print("No changes needed. All required meters already exist.")

def check_meter_availability(epjson_path: str) -> Dict[str, bool]:
    """
    Check the availability of all end-use meters in an epJSON file.
    Returns a dictionary of meter names and whether they exist.
    
    Args:
        epjson_path: Path to the epJSON file
        
    Returns:
        Dictionary mapping meter names to boolean (True if present)
    """
    # List of common end-use meters to check
    meter_list = [
        "Electricity:Facility",
        "Electricity:HVAC",
        "Electricity:Heating",
        "Electricity:Cooling", 
        "Electricity:Fans",
        "Electricity:InteriorLights",
        "Electricity:ExteriorLights",
        "Electricity:InteriorEquipment",
        "NaturalGas:Facility",
        "NaturalGas:HVAC",
        "NaturalGas:Heating",
        "Fans:Electricity",
        "Cooling:Electricity",
        "Heating:Electricity",
        "Heating:NaturalGas"
    ]
    
    # Initialize results dictionary
    meter_availability = {meter: False for meter in meter_list}
    
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
    
    # Check Output:Meter objects
    if "Output:Meter" in epjson:
        for meter_key, meter_data in epjson["Output:Meter"].items():
            meter_name = meter_data.get("key_name")
            if meter_name in meter_availability:
                meter_availability[meter_name] = True
    
    # Check Output:Meter:MeterFileOnly objects
    if "Output:Meter:MeterFileOnly" in epjson:
        for meter_key, meter_data in epjson["Output:Meter:MeterFileOnly"].items():
            meter_name = meter_data.get("key_name")
            if meter_name in meter_availability:
                meter_availability[meter_name] = True
    
    return meter_availability



def add_outdoor_air_meters_to_epjson(epjson_path: str, output_path: str = None) -> None:
    """
    Check if outdoor air temperature and humidity output variables exist in an epJSON file.
    If not, add them and save the modified epJSON.
    
    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path
    
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
    
    # Define outdoor air variables we want to check/add
    outdoor_vars = [
        # Variable Name, Key Value (usually "Environment")
        ("Site Outdoor Air Drybulb Temperature", "Environment"),
        ("Site Outdoor Air Humidity Ratio", "Environment"),
        ("Site Outdoor Air Relative Humidity", "Environment"),
        ("Site Outdoor Air Wetbulb Temperature", "Environment"),
        ("Site Outdoor Air Dewpoint Temperature", "Environment")
    ]
    
    # Check if we already have the necessary output variables
    has_outdoor_vars = {var[0]: False for var in outdoor_vars}
    
    # Check existing Output:Variable objects
    if "Output:Variable" in epjson:
        for var_key, var_data in epjson["Output:Variable"].items():
            var_name = var_data.get("variable_name")
            for outdoor_var, _ in outdoor_vars:
                if var_name == outdoor_var and var_data.get("reporting_frequency") == "Timestep":
                    has_outdoor_vars[outdoor_var] = True
                    print(f"Found existing output variable: {var_name} with Timestep reporting.")
    
    # Add variables if they don't exist
    modified = False
    
    # Make sure the Output:Variable category exists
    if "Output:Variable" not in epjson:
        epjson["Output:Variable"] = {}
    
    # Add missing variables
    for var_name, key_value in outdoor_vars:
        if not has_outdoor_vars[var_name]:
            new_var_key = f"Output:Variable {var_name}"
            
            # Ensure the key is unique by adding a suffix if needed
            suffix = 1
            while new_var_key in epjson["Output:Variable"]:
                new_var_key = f"Output:Variable {var_name} {suffix}"
                suffix += 1
            
            epjson["Output:Variable"][new_var_key] = {
                "key_value": key_value,
                "variable_name": var_name,
                "reporting_frequency": "Timestep"
            }
            print(f"Added output variable: {var_name} with Timestep reporting.")
            modified = True
    
    # Save the modified epJSON if changes were made
    if modified:
        with open(output_path, 'w') as f:
            json.dump(epjson, f, indent=2)
        print(f"Modified epJSON saved to {output_path}")
    else:
        print("No changes needed. All required outdoor air variables already exist.")

def check_output_variables_availability(epjson_path: str) -> Dict[str, bool]:
    """
    Check the availability of common output variables in an epJSON file.
    Returns a dictionary of variable names and whether they exist.
    
    Args:
        epjson_path: Path to the epJSON file
        
    Returns:
        Dictionary mapping variable names to boolean (True if present)
    """
    # List of common variables to check
    variable_list = [
        "Site Outdoor Air Drybulb Temperature",
        "Site Outdoor Air Humidity Ratio",
        "Site Outdoor Air Relative Humidity",
        "Site Outdoor Air Wetbulb Temperature",
        "Site Outdoor Air Dewpoint Temperature",
        "Zone Air Temperature",
        "Zone Air Humidity Ratio",
        "Zone Air Relative Humidity",
        "Zone Thermostat Heating Setpoint Temperature",
        "Zone Thermostat Cooling Setpoint Temperature",
        "Zone Thermal Comfort Mean Radiant Temperature",
        "Zone People Occupant Count",
        "Zone Air System Sensible Heating Energy",
        "Zone Air System Sensible Cooling Energy",
        "System Node Temperature",
        "System Node Relative Humidity",
        "System Node Mass Flow Rate",
        "Fan Electricity Rate"
    ]
    
    # Initialize results dictionary
    var_availability = {var: False for var in variable_list}
    
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
    
    # Check Output:Variable objects
    if "Output:Variable" in epjson:
        for var_key, var_data in epjson["Output:Variable"].items():
            var_name = var_data.get("variable_name")
            if var_name in var_availability:
                var_availability[var_name] = True
    
    return var_availability

def add_outdoor_air_nodes_if_missing(epjson_path: str, output_path: str = None) -> None:
    """
    Check if outdoor air node exists and add it if missing.
    This ensures that outdoor air can be properly monitored.
    
    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path
    
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
    
    # Check if we have an OutdoorAir:Node defined
    has_outdoor_air_node = "OutdoorAir:Node" in epjson and len(epjson["OutdoorAir:Node"]) > 0
    
    # Add OutdoorAir:Node if not present
    modified = False
    if not has_outdoor_air_node:
        # Create the OutdoorAir:Node category if it doesn't exist
        if "OutdoorAir:Node" not in epjson:
            epjson["OutdoorAir:Node"] = {}
        
        epjson["OutdoorAir:Node"]["Model Outdoor Air Node"] = {}
        print("Added OutdoorAir:Node for monitoring outdoor conditions.")
        modified = True
    else:
        print("Found existing OutdoorAir:Node.")
    
    # Check if we have an OutdoorAir:NodeList defined
    has_outdoor_air_nodelist = "OutdoorAir:NodeList" in epjson and len(epjson["OutdoorAir:NodeList"]) > 0
    
    # Add OutdoorAir:NodeList if not present
    if not has_outdoor_air_nodelist:
        # Create the OutdoorAir:NodeList category if it doesn't exist
        if "OutdoorAir:NodeList" not in epjson:
            epjson["OutdoorAir:NodeList"] = {}
        
        epjson["OutdoorAir:NodeList"]["OutdoorAir:NodeList"] = {
            "nodes": [
                {
                    "node_or_nodelist_name": "Model Outdoor Air Node"
                }
            ]
        }
        print("Added OutdoorAir:NodeList referencing outdoor air node.")
        modified = True
    else:
        print("Found existing OutdoorAir:NodeList.")
    
    # Save the modified epJSON if changes were made
    if modified:
        with open(output_path, 'w') as f:
            json.dump(epjson, f, indent=2)
        print(f"Modified epJSON saved to {output_path}")
    else:
        print("No changes needed to outdoor air nodes configuration.")
        
