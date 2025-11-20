from sentence_transformers import SentenceTransformer
from turtle import Turtle
import numpy as np 


BLOCK_SIZE = 2
STEP_SIZE = 40


# Nomic embeddings class
class NomicEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    
    def __call__(self, input):
        # Make sure input is always a list, even for a single string
        if isinstance(input, str):
            input = [input]
        
        # Generate embeddings
        embeddings = self.model.encode(input, prompt_name="passage")
        
        # Make sure the output has the right shape
        # If embeddings is a single vector, ensure it's returned properly
        if len(input) == 1 and len(embeddings.shape) == 1:
            # Convert 1D array to 2D array with one row
            return [embeddings]
        
        return embeddings.tolist()


# Function to generate donation data from donor data
def predict_platelet_yield(sex: str, 
                            weight_kg: float, 
                            height_cm: float, 
                            hematocrit: float, 
                            platelet_count_k_per_ul: int) -> dict:
    """
    Simulates a platelet donation to predict yield and time based on donor parameters.
    
    This model attempts to find the highest achievable target yield (from 8.2e11
    down to 3.0e11) that the donor can safely provide within machine and time limits.

    Args:
        sex: Donor's sex ("male" or "female").
        weight_kg: Donor's weight in kilograms.
        height_cm: Donor's height in centimeters.
        hematocrit: Donor's hematocrit as a percentage (e.g., 45 for 45%).
        platelet_count_k_per_ul: Donor's pre-donation platelet count in K/uL (e.g., 220 for 220,000/uL).

    Returns:
        A dictionary containing:
        - 'predicted_yield_e11': The predicted platelet yield (e.g., 3.0e11).
        - 'predicted_time_min': The predicted donation time in minutes.
        - 'tbv_ml': Calculated Total Blood Volume.
        - 'status': A message indicating success or any limitations hit.
    """
    
    # --- 1. Simulation Assumptions & Constants ---
    TARGET_YIELDS_E11 = [8.2, 7.8, 5.8, 5.4, 5.1, 3.0] # Yields to check, from highest to lowest
    
    BLOOD_PROCESS_RATE_ML_MIN = 65.0  # Avg. speed of blood draw/processing
    COLLECTION_EFFICIENCY = 0.60      # 60% efficiency in separating platelets
    MAX_PROC_TBV_MULTIPLIER = 1.5     # Max blood volume to process (e.g., 1.5 * TBV)
    MAX_DONATION_TIME_MIN = 120       # Max allowable donation time
    MIN_POST_PLATELET_COUNT = 100000  # Minimum safe post-donation count (per uL)

    # --- 2. Convert inputs and Calculate TBV ---
    hematocrit_decimal = hematocrit / 100.0  # Convert 45 -> 0.45
    platelet_count_per_ul = platelet_count_k_per_ul * 1000  # Convert 220 -> 220000
    platelet_count_per_ml = platelet_count_per_ul * 1000.0

    if sex.lower() == 'male' or sex.lower() == 'm':
        tbv_ml = weight_kg * 70  # Avg. 70 mL/kg for males
    elif sex.lower() == 'female' or sex.lower() == 'f':
        tbv_ml = weight_kg * 65  # Avg. 65 mL/kg for females
    else:
        raise ValueError("Sex must be 'male' or 'female'")
        
    total_pre_platelets = tbv_ml * platelet_count_per_ml
    
    platelets_collected_per_ml = platelet_count_per_ml * COLLECTION_EFFICIENCY
    
    if platelets_collected_per_ml == 0:
        return {
            'predicted_yield_e11': 0,
            'predicted_time_min': 0,
            'tbv_ml': tbv_ml,
            'status': 'Error: Platelet count is zero.'
        }

    # --- 3. Determine Absolute Max Possible Yield based on ALL constraints ---

    # Constraint 1: Max volume based on TBV
    max_volume_to_process_ml = tbv_ml * MAX_PROC_TBV_MULTIPLIER
    max_yield_from_max_vol = (max_volume_to_process_ml * platelets_collected_per_ml) / 1e11

    # Constraint 2: Max time
    max_volume_to_process_time = MAX_DONATION_TIME_MIN * BLOOD_PROCESS_RATE_ML_MIN
    max_yield_from_max_time = (max_volume_to_process_time * platelets_collected_per_ml) / 1e11

    # Constraint 3: Min post-donation platelet count
    # Max platelets we can safely take
    max_platelets_to_take = total_pre_platelets - (MIN_POST_PLATELET_COUNT * 1000 * tbv_ml)
    max_yield_from_post_count = max_platelets_to_take / 1e11
    
    # The true max yield is the *lowest* of these three constraints
    max_possible_yield_e11 = max(0, min(max_yield_from_max_vol, max_yield_from_max_time, max_yield_from_post_count))

    # --- 4. Find Highest Achievable Target Yield ---
    
    for target_yield in TARGET_YIELDS_E11:
        if max_possible_yield_e11 >= target_yield:
            # This is the highest target this donor can hit
            
            # Now, calculate the time required *for this specific target*
            target_yield_platelets = target_yield * 1e11
            volume_to_process_ml = target_yield_platelets / platelets_collected_per_ml
            predicted_time_min = volume_to_process_ml / BLOOD_PROCESS_RATE_ML_MIN
            
            return {
                'predicted_yield_e11': round(target_yield, 2),
                'predicted_time_min': round(predicted_time_min, 1),
                'tbv_ml': round(tbv_ml, 0),
                'status': f"Highest achievable target: {target_yield}e11"
            }

    # --- 5. If no target yield was met (max_possible_yield_e11 < 3.0) ---
    # Return the max possible yield, which will be below 3.0
    
    predicted_yield_e11 = max_possible_yield_e11
    target_yield_platelets = predicted_yield_e11 * 1e11
    
    if platelets_collected_per_ml > 0:
        volume_to_process_ml = target_yield_platelets / platelets_collected_per_ml
        predicted_time_min = volume_to_process_ml / BLOOD_PROCESS_RATE_ML_MIN
    else:
        predicted_time_min = 0 # Should be caught above, but for safety

    return {
        'predicted_yield_e11': round(predicted_yield_e11, 2),
        'predicted_time_min': round(predicted_time_min, 1),
        'tbv_ml': round(tbv_ml, 0),
        'status': "Donor unsuitable for standard 3.0e11 yield. Max possible yield shown."
    }


# Let's the user set up the simulation by inserting the donor's details
def machine_setup() -> list:
    print('//////////  Sub-Trima Machine Setup  //////////\n')
    # Required donor details
    data = [
        'Donor age: ',
        'Donor sex: ',
        'Height: ',
        'Weght: ',
        'Hematocrits: ',
        'Platelets count: ',
        'Donation number: ',
    ]
    # Initialise empty list to store the user input
    inserted_data = []
    # Populate the donor data by prompting the user
    for data_point in data:
        operator_input = input(f'{data_point}')
        inserted_data.append(operator_input)
    # Run function to calculate the donation data
    predicted_donation_values = predict_platelet_yield(
        inserted_data[1], int(inserted_data[3]), inserted_data[2], int(inserted_data[4]), int(inserted_data[5])
        )

    return inserted_data, predicted_donation_values


# Recursive function to create 'lines' by combining a main turtle unit and its clones
def create_line(original: Turtle, loop_count: int, distance: int = STEP_SIZE) -> None:
    doppleganger = original.clone()
    doppleganger.fd(distance)
    loop_count += 1
    if loop_count != 4:
        return create_line(doppleganger, loop_count, STEP_SIZE)
    else:
        return None


# Writes what phase the donation is currently on
def status_writer(status: int, writer: Turtle, title_font: tuple) -> None:
    if status == 1:
        writer.write("Return", align="center", font=title_font)
    elif status == 0:
        writer.write("Draw", align="center", font=title_font)
    elif status == 2:
        writer.write("Rinseback", align="center", font=title_font)
    elif status == 3:
        writer.write("Terminated", align="center", font=title_font)


# Function to write the donor and donation data on the display
def donor_info_writer(donor_info: list, donation_data: dict) -> None:
    writer = Turtle()
    writer.hideturtle()
    writer.penup()
    writer.goto(-400, 235)
    title_font = ("Arial", 30, "bold")
    donation_data_extract = [donation_data['predicted_time_min'], donation_data['predicted_yield_e11']]
    info = [
        'Age: ', 'Sex: ', 'Height: ', 'Weight: ', 'Hematocrits: ', 'PLT count: ', 'Donation No: ', 'Yield: ', 'Time: '
        ]
    info_to_write = [(info[idx] + data) for idx, data in enumerate(donor_info)]
    for i in range(len(donor_info)):
        writer.write(info_to_write[i], font=title_font)
        writer.goto(-400, (235 - (STEP_SIZE*(i+1))))
    landmark = 235 - (STEP_SIZE * len(donor_info))
    writer.goto(-400, landmark)
    for idx, data_point in enumerate(donation_data_extract):
        writer.write((info[::-1][idx] + str(data_point)), font=title_font)
        writer.goto(-400, (landmark - STEP_SIZE))


# Function to visualise the donation progress
def update_time(time_tracker: Turtle) -> Turtle:
    # 22 total blocks to fill
    new_block = time_tracker.clone()
    new_block.fd(STEP_SIZE//2)
    return new_block


# Set the chance of the draw and return to be ubnormal depending on what adjustments have been made during the donation
def scenario_orchestrator(scenario: list, adjustments: list) -> list:
    # Scenario 1: low draw presssure
    results = ['standard_draw', 'standard_return']
    # both pressure readings to remain within 10-20
    if 'draw' in scenario:
        # Give a small chance to have high return, even on a low draw scenario
        if 'Lower return speed from terminal' not in adjustments:
            return_pressure = [np.random.randint(11, 26) for _ in range(22)] # Low chance of being high return
        else:
            return_pressure = [np.random.randint(10, 23) for _ in range(22)] # Lowest chance of being high return
        high_pressure_ticks = [tick for tick in return_pressure if tick > 20]
        if len(high_pressure_ticks) > len(return_pressure) // 2: # Over 50% high ticks
            results[1] = ('high_return')
        # List of pressure ticks in a cycle, if more than half of them are below 10, return as a low draw cycle
        if 'First needle adjustment' not in adjustments:
            draw_pressure = [np.random.randint(0, 15) for _ in range(22)] # High chance of being low draw
        else:
            needle_adjustments = len([adj for adj in adjustments if adj == 'First needle adjustment' or adj == 'Second needle adjustment'])
            # 1 needle adjustment performed
            if needle_adjustments == 1:
                draw_pressure = [np.random.randint(0, 18) for _ in range(22)] # Low chance of being low draw
            # 2 needle adjustments performed
            elif needle_adjustments == 2:
                draw_pressure = [np.random.randint(5, 20) for _ in range(22)] # Lowest chance of being low draw
        low_pressure_ticks = [tick for tick in draw_pressure if tick < 10]
        if len(low_pressure_ticks) > len(draw_pressure) // 2: # Over 50% low ticks
            results[0] = ('low_draw')
    # Scenario 2: high return pressure
    if 'return' in scenario:
        # Give a small chance to have low draw, even on a high return scenario
        if 'First needle adjustment' not in adjustments:
            draw_pressure = [np.random.randint(5, 20) for _ in range(22)] # Low chance of being low draw
        else:
            draw_pressure = [np.random.randint(8, 20) for _ in range(22)] # Lowest chance of being low draw
        low_pressure_ticks = [tick for tick in draw_pressure if tick < 10]
        if len(low_pressure_ticks) > len(draw_pressure) // 2: # Over 50% low ticks
            results[0] = ('low_draw')
        if 'Lower return speed from terminal' not in adjustments:
            return_pressure = [np.random.randint(15, 31) for _ in range(22)] # High chance of having high return
        else:
            return_adjustments = len([adj for adj in adjustments if adj == 'Lower return speed from terminal'])
            if return_adjustments == 1:
                return_pressure = [np.random.randint(12, 31) for _ in range(22)] # Low chance of being high return
            elif return_adjustments == 2:
                return_pressure = [np.random.randint(10, 26) for _ in range(22)] # Lowest chance of being high return
        high_pressure_ticks = [tick for tick in return_pressure if tick > 20]
        if len(high_pressure_ticks) > len(return_pressure) // 2: # Over 50% high ticks
            results[1] = ('high_return')
    return results


# Get the variance (difference between one red square and another) of low draw pressure
def red_squares_variance(red_squares: list) -> list:
    variance = 0
    running_logic = False
    # If first revised feedback is green, count how many green cycles there were before it
    if red_squares[-1] == 0:
        # Loops through an inverted list of feedbacks
        for val in red_squares[::-1]:
            if val == 0:
                variance += 1
            # If the value of the feedback is 1 (1 = low draw) end the count
            else: break
        return variance
    else:
        for idx, val in enumerate(red_squares[::-1]):
            if not running_logic:
                pass
            else:
                if val == 0:
                    variance += 1
                else:
                    break
            # If the value checked is not the last one in the list, check if the value is 0 (0 = regular draw)
            if idx != (len(red_squares) - 1):
                if val == 1 and red_squares[::-1][idx+1] == 0:
                    running_logic = True
            else:
                pass
        return variance


# Cycle feedback motion display
def feedback_motion(feedback_list: list) -> None:
    # Move all feedback turtles
    for feedback in feedback_list:
        feedback.fd(80)
    # Hide non-active turtles
    if len(feedback_list) >= 3:
        for feedback in feedback_list[:-2]:
            feedback.hideturtle()


# Keep track of and display the draw pressure in each cycle
def cycle_feedback(feedback_list: list, state: str) -> list:
    # Create new feedback square
    feedback = Turtle()
    feedback.color('ForestGreen')
    # If the draw pressure is low, add a red square to the list
    if 'low_draw' in state:
        feedback.color('red1')
    feedback.shape('square')
    feedback.penup()
    feedback.left(180)
    feedback.shapesize(stretch_wid=BLOCK_SIZE*1.5, stretch_len=BLOCK_SIZE*1.5, outline=1)
    feedback.goto(80, -280)
    feedback_list.append(feedback)
    return feedback_list


# Update donors parameters with the model output
def params_updater_from_model(output: str, params: dict) -> dict:
    if output != 'No action':
        # Get the number that the machine was stopped from the parameters
        machine_stopped_n = int(params['instancesMachineWasStopped'])
        if output == 'Pause machine for 1 minute and inform nurse':
            params['instancesMachineWasStopped'] = (machine_stopped_n + 1)
        # Update needle adjustments count
        elif output == 'First needle adjustment' or output == 'Second needle adjustment':
            original_adjustments = int(params['needleAdjusted'])
            params['instancesMachineWasStopped'] = (machine_stopped_n + 1)
            params['needleAdjusted'] = (original_adjustments + 1)
        elif output == 'Pause machine for 1 minute and inform nurse' or output == 'Inform nurse and lower AC concentration':
            # Give 70% chance that the adjustemtn fixes the AC reaction
            chance = [num for num in [np.random.randint(0, 11) for _ in range(10)] if num < 8]
            if len(chance) >= 7: # Adjustment successfull
                params['donorHasACReactionSymptoms'] = 'no'
                params['reactionSeverity'] = 'none'
                print('\n--- AC adjusted successfully ---\n')
            else: # Symptoms persist
                print('\n--- AC symptoms persist ---\n')
                current_reactions = params['donorCurrentDonationReaction']
                params['donorCurrentDonationReaction'] = (current_reactions + 1)
        return params
    else: 
        return params
