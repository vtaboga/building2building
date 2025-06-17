from pathlib import Path
import os
import pandas as pd
import subprocess
import logging
import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from building2building.generator.utils import get_counties_from_coords_batch
from building2building.simulator.action_spaces import get_controllable_setpoints_rdf
from building2building.simulator import query_info
import building2building.env as env
from building2building.utils import cd

logger = logging.getLogger(__name__)

def process_metadata(state: str) -> Path:
    """
    Process all metadata CSV files in the metadata directory to get the county name for each row.
    For each file:
    1. Loads the CSV
    2. If County column exists, skip the file as it's already processed
    3. Parses the Centroid column to extract latitude and longitude
    4. Creates a new County column with the county name for each coordinate pair

    Returns the path of the processed file.
    """
    path_in = Path("data", "metadata", f"{state}.csv")
    path_out = Path("data", "metadata", f"{state}_processed.parquet")

    if path_out.exists():
        return path_out

    # Read the CSV file
    df = pd.read_csv(path_in)

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
    df.to_parquet(path_out)
    logger.info(f"Successfully processed and wrote to {path_out}")

    return path_out

def find_transitioned_file(original_file: Path, to_ver: str) -> Path:
    """Helper function to find the transitioned file which might have different naming patterns"""
    idf_dir = original_file.parent
    base_name = original_file.with_suffix("").name

    new_path = (idf_dir / base_name).with_suffix(".idfnew")

    # # Different possible patterns for the output file
    # possible_patterns = [
    #     # Pattern 1: original_name.idfnew (actual pattern we're seeing)
    #     os.path.join(idf_dir, f"{base_name}.idfnew"),
    #     # Pattern 2: original_name-V{version}.idf
    #     os.path.join(idf_dir, f"{base_name}-V{to_ver.replace('.', '-')}.idf"),
    #     # Pattern 3: original_name.V{version}.idf
    #     os.path.join(idf_dir, f"{base_name}.V{to_ver.replace('.', '-')}.idf"),
    #     # Pattern 4: original_name-{version}.idf
    #     os.path.join(idf_dir, f"{base_name}-{to_ver.replace('.', '-')}.idf"),
    #     # Pattern 5: original_name.{version}.idf
    #     os.path.join(idf_dir, f"{base_name}.{to_ver}.idf"),
    #     # Pattern 6: original_name.new
    #     os.path.join(idf_dir, f"{base_name}.new"),
    # ]

    if not new_path.exists():
        raise Exception(f"can't find transitioned file at {new_path}")

    return new_path

def transition_idf(idf_path: Path, state: str, county: str, target_version: str = "24.1") -> Path:
    """
    Transition an IDF file to the target EnergyPlus version using the transition executables.
    Saves the processed file in data/processed_buildings/state/county/ directory.

    Args:
        idf_path (str): Path to the input IDF file
        state (str): Two-letter state code
        county (str): County name
        target_version (str): Target EnergyPlus version (default: "24.1")

    Returns:
        str: Path to the transitioned file in the processed_buildigns directory
    """

    idf_path = idf_path.resolve()

    # Create the output directory structure
    processed_dir = Path("data", "processed_buildings", state, county).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    idf_dir = idf_path.parent
    idf_name = Path(idf_path.name)

    # Define the final output path
    final_output_path = processed_dir / idf_name

    # Create a temporary directory for transition files
    temp_dir = idf_dir / "temp_transition"
    temp_dir.mkdir(parents=True, exist_ok=True)

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

    # Copy the original file to temp directory to work with

    energyplus_dir = env.ENERGYPLUS_PATH.get()
    transition_dir = energyplus_dir / "PreProcess" / "IDFVersionUpdater"

    def transition_exe(from_ver, to_ver):
        p = transition_dir / f"Transition-V{from_ver.replace('.', '-')}-to-V{to_ver.replace('.', '-')}"
        if not p.exists():
            raise Exception(f"Error: Transition executable not found at {transition_exe}")
        return p

    with cd(temp_dir):
        working_file = idf_name
        shutil.copy(idf_path, working_file)

        for transition in transitions:
            from_ver, to_ver = transition.split("-to-")

            # Get paths for the transition executable and IDD files
            from_idd = os.path.join(transition_dir, f"V{from_ver.replace('.', '-')}-Energy+.idd")
            to_idd = os.path.join(transition_dir, f"V{to_ver.replace('.', '-')}-Energy+.idd")

            # Check if required files exist
            if not os.path.exists(from_idd):
                e = f"Error: Source IDD file not found at {from_idd}"
                logger.error(e)
                raise Exception(e)
            if not os.path.exists(to_idd):
                e = f"Error: Target IDD file not found at {to_idd}"
                logger.error(e)
                raise Exception(e)

            # Create symbolic links to IDD files in the temp directory (not in /opt/repository)
            from_idd_link = os.path.basename(from_idd)
            to_idd_link = os.path.basename(to_idd)
            if not os.path.exists(from_idd_link):
                os.symlink(from_idd, from_idd_link)
            if not os.path.exists(to_idd_link):
                os.symlink(to_idd, to_idd_link)

            logger.info(f"\nRunning transition from {from_ver} to {to_ver}")

            cwd = os.getcwd()
            # Run the transition executable from the temp_dir
            cmdline = [transition_exe(from_ver, to_ver), os.path.basename(working_file)]
            logging.debug(f"With CWD: {cwd}")
            logging.debug(f"Using cmd: {cmdline}")
            result = subprocess.run(
                cmdline,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"DISPLAY": ""},
                cwd=os.getcwd()
            )

            print(f"return code trnasition {from_ver} {to_ver} is {result.returncode}")

            if result.stdout:
                logger.debug(f"Transition output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Transition errors: {result.stderr}")

            # Clean up any .idfold files that might have been created
            old_file = Path(str(working_file) + "old")
            if old_file.exists():
                old_file.unlink()

            # Look for the transitioned file
            transitioned_file = find_transitioned_file(working_file, to_ver)
            if transitioned_file:
                logger.debug(f"Found transitioned file at: {transitioned_file}")
                if transitioned_file != working_file:
                    # Replace working file with transitioned file
                    os.rename(transitioned_file, working_file)
            else:
                e = f"Error: Could not find transitioned file for {working_file}"
                logger.error(e)
                raise Exception(e)
        # Copy the final file to the processed_buildigs directory instead of original location
        shutil.copy(working_file, final_output_path)

        logger.info(f"\nFinal transitioned file saved to: {final_output_path}")

        # Convert the final IDF to epJSON
        epjson_path = convert_to_epjson(final_output_path)
        logger.info(f"Converted to epJSON: {epjson_path}")
        add_setpoint_control_to_epjson(epjson_path)
        logger.info(f"Added setpoint control to {epjson_path}")
        if epjson_path:
            return epjson_path
        else:
            logger.error("Failed to convert to epJSON format")
            return final_output_path  # Return IDF path as fallback

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


    return final_output_path

def process_idf(building_files: List[tuple], state: str, county: str):
    """
    Process IDF files if they haven't been processed already.

    Args:
        idf_files (List[int]): List of IDF file IDs
        state (str): Two-letter state code
        county (str): County name

    Returns:
        List[str]: List of paths to processed IDF files
    """
    processed_dir = os.path.join("data", "processed_buildings", state, county)
    os.makedirs(processed_dir, exist_ok=True)

    processed_paths = []
    for idf_id, characteristics in building_files:
        # Check if processed file already exists
        processed_path = Path(processed_dir, f"{idf_id}.epJSON")
        if processed_path.exists():
            logger.info(f"Skipping {idf_id}.epJSON - already processed")
            processed_paths.append(processed_path)
            continue

        # Get path to original file
        original_path = Path("data", "idf", f"{state}_{county}_IDF", f"{idf_id}.idf")
        if not original_path.exists():
            logger.warning(f"Warning: Original file not found at {original_path}")
            continue

        # Process the file
        processed_path = transition_idf(original_path, state, county)
        if processed_path:
            processed_paths.append(processed_path)

            # Save the building characteristics as a JSON file
            characteristics_path = os.path.join(processed_dir, f"{idf_id}.json")
            # Add zone lists to characteristics
            zone_lists = get_zone_lists(processed_path)
            characteristics["zone_lists"] = zone_lists
            with open(characteristics_path, 'w') as json_file:
                json.dump(characteristics, json_file, indent=4)
            logger.info(f"Saved characteristics to {characteristics_path}")
            
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
        epjson_path = os.path.splitext(Path(idf_path).resolve())[0] + '.epJSON'
        
        # Path to the EnergyPlus executable
        energyplus_dir = env.ENERGYPLUS_PATH.get()
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
        if result.stdout:
            logger.warning(f"Conversion warnings: {result.stdout}")            

        # Check if the epJSON file was created
        if os.path.exists(epjson_path):
            logger.info(f"Successfully converted to: {epjson_path}")
            # Delete the original IDF file
            os.remove(idf_path)
            logger.debug(f"Deleted original IDF file: {idf_path}")
            return epjson_path
        else:
            logger.error(f"epJSON file not created at expected path: {epjson_path}")
            raise Exception(f"epJSON file not created at expected path: {epjson_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise
    

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
        

def modify_timestep(epjson_path: str, output_path: Optional[str] = None, timesteps_per_hour: int = 4) -> None:
    """
    Modifies the timestep in an epJSON file to a specific value.
    
    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
        timesteps_per_hour: Number of timesteps per hour (default: 4, which is 15-minute timesteps)
    
    Returns:
        None
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path
    
    # Load the epJSON file
    try:
        with open(epjson_path, 'r') as f:
            epjson = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{epjson_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{epjson_path}' is not a valid JSON file.")
        return
    
    # Check for existing Timestep object
    timestep_found = False
    timestep_modified = False
    
    if "Timestep" in epjson:
        timestep_found = True
        # Process all timestep objects (usually there's just one)
        for timestep_key, timestep_data in epjson["Timestep"].items():
            current_timestep = timestep_data.get("number_of_timesteps_per_hour", 1)
            if current_timestep != timesteps_per_hour:
                print(f"Changing timestep from {current_timestep} to {timesteps_per_hour} timesteps per hour")
                epjson["Timestep"][timestep_key]["number_of_timesteps_per_hour"] = timesteps_per_hour
                timestep_modified = True
            else:
                print(f"Timestep already set to {timesteps_per_hour} timesteps per hour")
    
    # If no Timestep object found, create one
    if not timestep_found:
        print(f"No Timestep object found. Creating one with {timesteps_per_hour} timesteps per hour")
        epjson["Timestep"] = {
            "Timestep 1": {
                "number_of_timesteps_per_hour": timesteps_per_hour
            }
        }
        timestep_modified = True
    
    # Check if we need to update RunPeriod objects to match timestep
    # Some EnergyPlus simulations may have issues if RunPeriod doesn't align with timestep
    if "RunPeriod" in epjson:
        for runperiod_key, runperiod_data in epjson["RunPeriod"].items():
            # Just noting this - we don't need to change anything in RunPeriod for timestep
            pass
    
    # Save the modified epJSON if changes were made
    if timestep_modified:
        try:
            with open(output_path, 'w') as f:
                json.dump(epjson, f, indent=2)
            print(f"Modified epJSON saved to {output_path}")
        except Exception as e:
            print(f"Error saving modified file: {e}")
    else:
        print("No changes were made to the epJSON file")
        
def add_setpoint_control_to_epjson(epjson_path: str, output_path: str = None) -> None:
    """
    Modifies an epJSON file to add controllable temperature setpoint schedules.
    
    Args:
        epjson_path (str): Path to the input epJSON file
        output_path (str, optional): Path to save the modified epJSON. If None, overwrites input file.
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path
        
    # Load the epJSON file
    with open(epjson_path, 'r') as f:
        epjson = json.load(f)
        
    # Get thermostat setpoints used in the building
    thermostat_setpoints = get_temperature_setpoints(epjson_path)
    
    # Make sure Schedule:Compact exists in epjson
    if "Schedule:Compact" not in epjson:
        epjson["Schedule:Compact"] = {}
        
    def create_schedule_compact(temperature: float):
        """Helper function to create a schedule compact object in the correct format"""
        return {
            "data": [
                {"field": "Through: 12/31"},
                {"field": "For: AllDays"},
                {"field": "Until: 24:00"},
                {"field": temperature}
            ],
            "schedule_type_limits_name": "Temperature"
        }
        
    # Process each thermostat
    for control_type, setpoint_name in thermostat_setpoints:
        if control_type == "ThermostatSetpoint:DualSetpoint":
            # Get the original schedule names
            dual_setpoint = epjson["ThermostatSetpoint:DualSetpoint"][setpoint_name]
            
            # Create new schedule names
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"
            
            # Update the thermostat to use new schedules
            dual_setpoint["cooling_setpoint_temperature_schedule_name"] = cooling_schedule_name
            dual_setpoint["heating_setpoint_temperature_schedule_name"] = heating_schedule_name
            
            # Create cooling setpoint schedule if it doesn't exist
            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = create_schedule_compact(25.0)
            
            # Create heating setpoint schedule if it doesn't exist
            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = create_schedule_compact(20.0)
            
        elif control_type == "ThermostatSetpoint:SingleHeating":
            # Create new schedule name
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"
            
            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleHeating"][setpoint_name]["setpoint_temperature_schedule_name"] = heating_schedule_name
            
            # Create heating setpoint schedule if it doesn't exist
            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = create_schedule_compact(20.0)
            
        elif control_type == "ThermostatSetpoint:SingleCooling":
            # Create new schedule name
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            
            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleCooling"][setpoint_name]["setpoint_temperature_schedule_name"] = cooling_schedule_name
            
            # Create cooling setpoint schedule if it doesn't exist
            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = create_schedule_compact(25.0)
            
        elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
            # Create new schedule name for the single setpoint
            setpoint_schedule_name = f"{setpoint_name} Setpoint"
            
            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleHeatingOrCooling"][setpoint_name]["setpoint_temperature_schedule_name"] = setpoint_schedule_name
            
            # Create setpoint schedule if it doesn't exist
            if setpoint_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][setpoint_schedule_name] = create_schedule_compact(22.5)
    
    # Save the modified epJSON
    with open(output_path, 'w') as f:
        json.dump(epjson, f, indent=2)

def get_temperature_setpoints(epjson_path: str) -> List[tuple]:
    """
    Analyzes an epJSON file to identify thermostat setpoints that are used to control zones.
    
    Args:
        epjson_path (str): Path to the epJSON file
        
    Returns:
        List[tuple]: List of tuples containing (thermostat_type, thermostat_name) for thermostats 
                     that are used to control at least one zone
    """
    # Convert epJSON to RDF for querying
    rdf_graph = query_info.rdf_from_json(epjson_path)
    
    # Query to find thermostats that are used in zone controls
    thermostat_query = """# -*- mode: sparql -*-
    SELECT DISTINCT ?control_type ?setpoint_name
    WHERE {
        # Find zone controls and their types
        ?control a ns:ZoneControl%3AThermostat .
        ?control ns:control_1_object_type ?control_type .
        ?control ns:control_1_name ?setpoint_name .

        # Make sure the control is used by at least one zone
        ?control ?zone_prop ?zone_name .
        FILTER(?zone_prop = ns:zone_or_zonelist_name) .
    }
    """
    
    # Execute query and process results
    results = []
    for row in rdf_graph.query(thermostat_query, initNs={"ns": query_info.ns}):
        control_type = str(row.control_type)
        setpoint_name = str(row.setpoint_name)
        results.append((control_type, setpoint_name))
    
    return results


def get_zone_lists(epjson_path: str) -> List:
    """
    Analyzes an epJSON file to identify zone lists
    
    Args:
        epjson_path (str): Path to the epJSON file
    """

    with open(epjson_path, 'r') as f:
        epjson = json.load(f)

    data = epjson["ZoneList"]
    zone_names = []
    # Iterate through each space type in the dictionary
    for space_type, space_info in data.items():
        # Check if 'zones' key exists in the current space type
        if 'zones' in space_info:
            # Extract zone names from each zone dictionary
            for zone in space_info['zones']:
                if 'zone_name' in zone:
                    zone_names.append(zone['zone_name'])
    
    return zone_names  

