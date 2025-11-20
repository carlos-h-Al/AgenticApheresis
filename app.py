from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from turtle import Screen, Turtle
from utils import *
import pandas as pd
import numpy as np 
import chromadb
import joblib
import time


STEP_SIZE = 40

PRESSURE_COLORS = [
    'darkred',   # lightred
    'goldenrod', # khaki
    'seagreen'   # cyan
]

BLOCK_SIZE = 2

# Order of priority for the model
PRIORITY_ORDER = [
    "End donation, AC toxicity reached",            # Safety Critical (Highest Priority)
    "End donation, AC toxicity buildup",            # Safety Critical
    "End donation, give rinseback",                 # Safety Critical
    "No Action",                                    # Preferred state (Stability)
    "Lower return speed from terminal",             # Minor intervention
    "Inform nurse and lower AC concentration",      # Nurse intervention
    "Pause machine for 1 minute and inform nurse", 
    "First needle adjustment",                      # Physical intervention (Last resort)
    "Second needle adjustment"
]

# Initialise a dictionary with all the parameters needed to run the model
PARAMETERS = {
    'donorAge': 0,                                     # Integer                            :   yes
    'donorHasACReactionSymptoms': 'no',                # Yes or no                          :   ~
    'firstDonation': 'no',                             # Yes or no                          :   no
    'machineRedSquares': 0,                            # integer                            :   ~
    'reactionSeverity': 'none',                        # 'none', 'low', 'medium', 'high'    :   ~
    'donorCurrentDonationReaction': 0,                 # Integer                            :   ~
    'timePressureBarGoingToYellowInSeconds': 0,        # Integer                            :   ~
    'donorHasReactionHistory': 'no',                   # Yes or no                          :   no
    'instancesMachineWasStopped': 0,                   # Integer                            :   ~
    'cuffPressureON': 'yes',                           # Yes or no                          :   no
    'lowerPlateletsCount': 0,                          # Integer                            :   yes
    'donationElapsedTime': 0,                          # Integer                            :   ~
    'isDonorSqueezingDuringReturn': 'no',              # Yes or no                          :   ~
    'donorSex': '',                                    # M or F                             :   yes
    'needleAdjusted': 0,                               # Integer                            :   ~
    'isCuffpressureHigh': 'no',                        # Yes or no                          :   ~
    'isNeedleVibrating': 'no',                         # Yes or no                          :   ~
    'donorHasLowDrawHistory': 'no',                    # Yes or no                          :   no
    'donorHasHighReturnHistory': 'no',                 # Yes or no                          :   yes
    'donorEBV': 0,                                     # Float                              :   yes
    'heathpadON': 'yes',                               # Yes or no                          :   ~
    'machineRedSquaresVarianceInCicles': 0,            # Integer                            :   ~
    'pressureBarGoingToYellow': 'no',                  # Yes or no                          :   ~
}

# Model outputs that require the donation to stop immediately
FATAL_OUTCOMES = [
    'End donation, AC toxicity reached',
    'End donation, AC toxicity buildup',
    'End donation, give rinseback'
]

# Load the model components from the file
MODEL = joblib.load('model_collection/XXXL_model_2/tuned_random_forest_model.joblib')
LE_DICT = joblib.load('model_collection/XXXL_model_2/label_encoders.joblib')
FEATURES_NAMES = joblib.load('model_collection/XXXL_model_2/feature_names.joblib')
LLM_MODEL = OllamaLLM(model='gpt-oss:latest')
CLASS_NAMES = MODEL.classes_ 


# Generate the pressure display to monitor the donation
def create_pressure_screen(line_number: int, 
                            colors: list) -> None:
    # Initialise first building block
    building_block = Turtle()
    building_block.shape('square')
    building_block.shapesize(stretch_wid=BLOCK_SIZE, stretch_len=BLOCK_SIZE, outline=1)
    building_block.penup()
    line_distances = [
        200, 160, 120, 80, 40
    ]
    if line_number < 5: # Bottom screen
        building_block.right(90)
        building_block.fd(line_distances[line_number]) # Move based on the line number
        building_block.left(90)
        building_block.bk(80) # Put building block into starting position
        if line_number == 0: # Red
            building_block.color(colors[0])
        elif line_number > 0 and line_number < 3: # Yellow
            building_block.color(colors[1])
        elif line_number > 2:
            building_block.color(colors[2])
        create_line(building_block, 0)
    elif line_number == 5: # Middle line
        building_block.bk(80) # Put building block into starting position
        building_block.color(colors[2])
        create_line(building_block, 0)
    elif line_number > 5: # Top screen
        building_block.left(90)
        building_block.fd(line_distances[::-1][line_number-6]) # Move based on the line number
        building_block.right(90)
        building_block.bk(80) # Put building block into starting position
        if line_number == 10: # Red
            building_block.color(colors[0])
        elif line_number > 7 and line_number < 10: # Yellow
            building_block.color(colors[1])
        elif line_number < 8: # Green
            building_block.color(colors[2])
        create_line(building_block, 0)

    line_number += 1
    # Draw outline
    if line_number == 11:
        border = Turtle()
        writer = Turtle()
        writer.hideturtle()
        writer.penup()
        writer.goto(215, 235)
        title_font = ("Arial", 50, "bold")
        writer.write("Time", align="center", font=title_font)
        border.hideturtle()
        border.penup()
        border.bk(105)
        border.pendown()
        border.pensize(10)
        border.right(90)
        distances_main = [
            225, 210, 450, 210
        ]
        distances_side = [
            91, 450, 91, 450
        ]
        for i in range(4):
            border.fd(distances_main[i])
            border.left(90)
        border.fd(230)

        # Donation time bar set up
        border.penup()
        border.goto(170, -225)
        border.pendown()
        border.left(90)
        for i in range(4):
            border.fd(distances_side[i])
            border.left(90)

        screen.update()
        return None
    else:
        # Recursively generate line after line, in different colors
        return create_pressure_screen(line_number, colors)


# Initialise the turtle objects required for the pressure bar
def pressure_line_starter() -> list:
    # Initialise the main turtle unit for the pressure bar
    pressure_line = Turtle()
    pressure_line.penup()
    pressure_line.shape('square')
    pressure_line.shapesize(stretch_wid=BLOCK_SIZE, stretch_len=BLOCK_SIZE, outline=1)
    pressure_line.bk(80)
    pressure_line.color('cyan')

    # Create clones
    pressure_line_members = [pressure_line]
    # Create a list of turtles that will act as a single unit
    for _ in range(4):
        clone = pressure_line_members[-1].clone()
        clone.fd(STEP_SIZE)
        pressure_line_members[-1].left(90)
        pressure_line_members.append(clone)
    pressure_line_members[-1].left(90)
    screen.update()
    return pressure_line_members


# Handle user inputs, and run the assistant or the RAG
def get_user_input(params: dict = PARAMETERS, 
                    model=LLM_MODEL) -> None:
    # When 'i' key is pressed, the input box is displayed
    user_input = screen.textinput(title='Chat box', 
                                prompt='Share your symptoms / chat with Plato')
    # Template used by the LLM to decide if the user is trying to ask a general question or share symptoms
    template_for_decision = '''
    You are a healthcare professional directly supervising a platelets donor during their donation.
    Your task is to analyse the donor message and understand if they want to share symptoms or have a conversation.

    Donor message: {user_input}

    Return only one word answer:
    assist (in case of the donor sharing symptoms or sensations)

    OR

    chat (if the donor just wants to make conversation or ask general questions)
    '''
    # Template for the LLM to update the right parameters when donor shares symptoms
    template_for_symptoms = '''
    Given the following parameters:

    'donorHasACReactionSymptoms': 'no',                # Yes or no                          :   ~
    'reactionSeverity': 'none',                        # 'none', 'low', 'medium', 'high'    :   ~

    User input: {user_input}
    Donor sex: {donor_sex}

    Your task is to extract from the user input any relevant information regarding the parameters, by assessing their symptoms.

    Here are some examples of an occurring Anti Coagulant reaction and its severity:

    severity: low - tingling on lips with no intensity specified, metallic taste with no intensity specified
    severity: medium - intense tingling on lips, intense metallic taste, tingling on entire body parts (eg: entire leg, feet, entire arm)
    severity: high - same symptoms as medium BUT the donor is a female, donor sed: F.

    You then need to answer with only the code necessary to change the parameters found in the user input, in text format.
    The dictionary variable name is 'params'.
    '''
    # Template used to generate an answer to the donor query using RAG
    template_for_retieval = '''
    You are a healthcare professional directly supervising a platelets donor during their donation.
    Your task is to answer the donor questions regarding platelets donations and general knowledge about platelets.

    Answer the <Question> below only by using the given information in the solution.

    <Question>: {question}

    Here is the solution: {info} 

    If the provided information does not contain the exact answer to the question, 
    respond with "Information not found in the booklet" and then answer with your own knowledge.
    Always prioritize the booklet knowledge.

    Otherwise, provide a complete answer without commentary or repeating the question.
    '''
    # Set templatre to prompt the model
    prompt = ChatPromptTemplate.from_template(template_for_decision)
    # Concatenates the prompt to the model
    chain = prompt | model
    # If the donor types anything, run the LLM
    if user_input:
    # Generate answer
        answer = chain.invoke({'user_input': user_input})
        # Update parameters
        if answer == 'assist':
            print('\n*** Running assistant ***\n')
            # Set templatre to prompt the model
            prompt = ChatPromptTemplate.from_template(template_for_symptoms)
            # Concatenates the prompt to the model
            chain = prompt | model
            # Generate answer
            answer = chain.invoke({'user_input': user_input, 'donor_sex': params['donorSex']})
            # Format answer
            clean_answer = answer.replace('`', '')
            coded_answer = [code for code in clean_answer.split('\n') if 'params' in code]
            # Execute code from LLM answer
            exec(coded_answer[0])
            exec(coded_answer[1])
            # Update reaction count parameters
            current_reactions = int(params['donorCurrentDonationReaction'])
            params['donorCurrentDonationReaction'] = (current_reactions + 1)
        # Run RAG
        elif answer == 'chat':
            print('\n*** Running RAG ***\n')
            # Initialise embedding function
            nomic = NomicEmbeddingFunction()
            client = chromadb.PersistentClient(path='embeddings/platelets_general')
            platelets_general_knowledge = client.get_or_create_collection(
                name='general', 
                embedding_function=nomic
            )
            query_embedding = nomic(user_input)
            query_result = platelets_general_knowledge.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            retrieved_chunks = query_result['documents'][0]
            query_info = ''
            for chunk in retrieved_chunks:
                query_info += (' ' + chunk)
            # Set templatre to prompt the model
            prompt = ChatPromptTemplate.from_template(template_for_retieval)
            # Concatenates the prompt to the model
            chain = prompt | model
            # Generate answer
            answer = chain.invoke({'question': user_input, 'info': query_info})
            print(f'\nYou: {user_input}\nPlato: {answer}\n')


# Function to control and display the draw phase
def draw_cycle(writer: Turtle, 
                title_font: tuple, 
                pressure_line_members: list, 
                time_variance: float, 
                scenario: list) -> None:
    writer.clear()
    status_writer(0, writer, title_font)
    # Normal scenario
    if 'standard_draw' in scenario:
        # Bring the bar to the bottom
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)
        # Start oscillating
        for i in range(28): # 28
            for pressure_line_point in pressure_line_members:
                if i % 2 == 0:
                    pressure_line_point.fd(STEP_SIZE)
                else: pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)
        # Bring the bar to the middle for the return
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.fd(STEP_SIZE)
            screen.update()
            time.sleep(time_variance)
    # Low draw speed scenario
    elif 'low_draw' in scenario:
        # Bring the bar to the bottom
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.bk(STEP_SIZE)
            screen.update()
            time.sleep(time_variance)
        # Start oscillating - 28 total loops
        for i in range(10):
            for pressure_line_point in pressure_line_members:
                if i % 2 == 0:
                    pressure_line_point.fd(STEP_SIZE)
                else: pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)

        for i in range(6):
            if i < 2: # First 2 loops
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.bk(STEP_SIZE)
                    pressure_line_point.color('khaki')
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)
            elif i == 2:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.bk(STEP_SIZE)
                    pressure_line_point.color('red')
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance*6)
            elif i > 2 and i < 5:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.fd(STEP_SIZE)
                    pressure_line_point.color('khaki')
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)
            elif i == 5:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.fd(STEP_SIZE)
                    pressure_line_point.color('cyan')
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)

        for i in range(10):
            for pressure_line_point in pressure_line_members:
                if i % 2 == 0:
                    pressure_line_point.fd(STEP_SIZE)
                else: pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)
        # Bring the bar to the middle for the return
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.fd(STEP_SIZE)
            screen.update()
            time.sleep(time_variance)


# Function to control and display the return phase
def return_cycle(writer: Turtle, 
                title_font: tuple, 
                pressure_line_members: list, 
                time_variance: float, 
                scenario: list) -> None:
    writer.clear()
    status_writer(1, writer, title_font)
    # Normal scenario
    if 'standard_return' in scenario:
        # Bring the bar to the top
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.fd(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)
        # Start oscillating, 2 different normal scenarios
        sub_scenario = np.random.randint(0, 2)
        if sub_scenario == 0:
            # Scenario 1
            for i in range(28): # 28
                for pressure_line_point in pressure_line_members:
                    if i % 2 == 0:
                        pressure_line_point.bk(STEP_SIZE)
                    else: pressure_line_point.fd(STEP_SIZE)
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)
        else:
            # Scenario 2
            for i in range(10): # 28
                for pressure_line_point in pressure_line_members:
                    if i % 2 == 0:
                        pressure_line_point.bk(STEP_SIZE)
                    else: pressure_line_point.fd(STEP_SIZE)
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)

            for i in range(4):
                for pressure_line_point in pressure_line_members:
                    if i % 2 == 0:
                        pressure_line_point.color('khaki')
                        pressure_line_point.fd(STEP_SIZE)
                    else:
                        pressure_line_point.color('cyan')
                        pressure_line_point.bk(STEP_SIZE)
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                if i % 2 == 0:
                    time.sleep(time_variance*4)
                else:
                    time.sleep(time_variance)

            for i in range(10): # 28
                for pressure_line_point in pressure_line_members:
                    if i % 2 == 0:
                        pressure_line_point.bk(STEP_SIZE)
                    else: pressure_line_point.fd(STEP_SIZE)
                screen.listen()
                screen.onkey(get_user_input, "i")
                screen.update()
                time.sleep(time_variance)
        # Bring the bar to the middle for the next draw
        for _ in range(2):
            for pressure_line_point in pressure_line_members:
                pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)
    elif 'high_return' in scenario:
        # Bring the bar to the top
        for i in range(3):
            if i == 2:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.color('khaki')
                    pressure_line_point.fd(STEP_SIZE)
            else:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.color('cyan')
                    pressure_line_point.fd(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)

        # Oscillate in the yellow area
        for i in range(28): # 28
            for pressure_line_point in pressure_line_members:
                if i % 2 == 0:
                    pressure_line_point.fd(STEP_SIZE)
                else: pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)

        # Bring the bar to the middle for the next draw
        for i in range(3):
            if i > 0:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.color('cyan')
                    pressure_line_point.bk(STEP_SIZE)
            else:
                for pressure_line_point in pressure_line_members:
                    pressure_line_point.bk(STEP_SIZE)
            screen.listen()
            screen.onkey(get_user_input, "i")
            screen.update()
            time.sleep(time_variance)


# Encode parameters with numbers that the model understands
def params_encoder(params: dict) -> pd.DataFrame:
    # Convert to DataFrame
    data_df = pd.DataFrame([params])
    df_encoded = data_df.copy()
    # Apply the loaded encoders
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            if col in LE_DICT:
                le = LE_DICT[col] # Get the original encoder
                df_encoded[col] = le.transform(df_encoded[col])
    # Ensure column order matches training using the loaded list
    df_encoded = df_encoded[FEATURES_NAMES]
    return df_encoded


# Run inference on the given parameters
def run_inference(params: dict) -> str:
    # Ensure column order matches training
    params = params[FEATURES_NAMES] # X_train.columns is from your training cell

    # Get the outcome label from the original outcome encoder
    predicted_index = MODEL.predict(params)[0]
    predicted_label = LE_DICT['outcome'].inverse_transform([predicted_index])[0]
    return predicted_label


# Get model prediction with a tie-breaker
def get_prediction_with_priority(probabilities: list, 
                                model_classes: list = CLASS_NAMES, 
                                label_encoder: list = LE_DICT['outcome'], 
                                priority_order: list = PRIORITY_ORDER) -> str:
    # Find the maximum probability score
    max_score = np.max(probabilities)

    # Find all indices that are effectively equal to the max score
    # (Using isclose handles float precision issues like 0.33333 vs 0.33334)
    tied_indices = np.where(np.isclose(probabilities, max_score))[0]

    # If there is only one winner, return it normally
    if len(tied_indices) == 1:
        class_index = model_classes[tied_indices[0]]
        return label_encoder.inverse_transform([class_index])[0]

    # HANDLING TIES
    print(f"\n>>> TIE DETECTED between {len(tied_indices)} classes with score {max_score:.4f}")

    # Convert the tied indices to actual string labels
    tied_labels = []
    for idx in tied_indices:
        class_index = model_classes[idx]
        label = label_encoder.inverse_transform([class_index])[0]
        tied_labels.append(label)

    print(f"    Candidates: {tied_labels}")

    # Check against priority order
    for priority_label in priority_order:
        if priority_label in tied_labels:
            print(f"    >>> Tie-breaker winner: {priority_label}\n")
            return priority_label

    # Fallback (should not happen if priority list covers all classes)
    return tied_labels[0]


# Update parameters as the simulation runs
def parameters_updater(scenarios: list = None, 
                        red_squares: list = None, 
                        donor_data: list = None, 
                        donation_data: dict = None, 
                        status = 'initial', 
                        params = PARAMETERS,
                        donation_time: int = 90,
                        loop_count: int = 0) -> dict:
    # To run once, before starting the simulation
    if status == 'initial':
        # Donor data initialisation
        params['donorAge'] = donor_data[0]
        if donor_data[1] == 'male':
            params['donorSex'] = 'M'
        else:
            params['donorSex'] = 'F'
        params['lowerPlateletsCount'] = donor_data[5]
        if int(donor_data[6]) != 1:
            params['firstDonation'] = 'no'
            # Add random history
            params['donorHasReactionHistory'] = np.random.choice(['yes', 'no'])
            params['donorHasLowDrawHistory'] = np.random.choice(['yes', 'no'])
            params['donorHasHighReturnHistory'] = np.random.choice(['yes', 'no'])
        else:
            params['firstDonation'] = 'yes'
        # Donation data initialisation
        params['donorEBV'] = donation_data['tbv_ml'] / 1000
        return params
    # To run while the simulationm is running
    else:
        # Idenify number of red squares
        red_square_instances = len([val for val in red_squares if val == 1])
        params['machineRedSquares'] = red_square_instances
        # Get variance of red squares
        params['machineRedSquaresVarianceInCicles'] = red_squares_variance(red_squares)
        # Update donation elapsed time for every cycle
        params['donationElapsedTime'] = int((float(donation_time) // 22) * loop_count)
        if 'high_return' in scenarios:
            params['pressureBarGoingToYellow'] = 'yes'
            params['timePressureBarGoingToYellowInSeconds'] = np.random.randint(0, 11)
        elif 'standard_return' in scenarios:
            params['pressureBarGoingToYellow'] = 'no'
            params['timePressureBarGoingToYellowInSeconds'] = 0
        return params


# Modifies the parameters based on the scenario
def scenario_starter(scenario: list, params: dict = PARAMETERS) -> dict:
    if 'draw' in scenario:
        params['isNeedleVibrating'] = 'yes'
    return params


# Guardrails to prevent the simulation to run wrong model outputs
def model_guardrails(adjustments: list, output: str, probabilities: list) -> str:
    # Return handler
    if output == 'Lower return speed from terminal' and PARAMETERS['pressureBarGoingToYellow'] == 'no':
            # 1. Get indices that would sort the array (ascending)
            sorted_indices = np.argsort(probabilities)
            # 2. Flip to make it descending (Highest -> Lowest)
            sorted_indices_desc = sorted_indices[::-1]
            # 3. Get the index of the SECOND best guess (Index 1)
            # Index 0 is the winner (which we are rejecting), Index 1 is the runner-up
            second_best_index = sorted_indices_desc[1]
            # 4. Get the actual class ID used by the model
            # (The model's internal class list might not map 0->0, 1->1 directly)
            class_id = MODEL.classes_[second_best_index]
            # 5. Decode the class ID back to a string
            output = LE_DICT['outcome'].inverse_transform([class_id])[0]
            print(f"\n>>> Switched to second best guess: {output}\n")
    adjustments.append(output)
    adjustment_instances = len([text for text in adjustments if text == 'First needle adjustment'])
    second_adjustment_instances = len([text for text in adjustments if text == 'Second needle adjustment'])
    if adjustment_instances > 1:
        output = 'No action'
    if second_adjustment_instances > 1:
        output = 'No action'
    adjustments.pop(-1)
    return output


# Set up screen variable and parameters
screen = Screen()
screen.setup(width=900, height=700)
screen.bgcolor("lightblue")
screen.title("Turtle")
screen.tracer(0)

# Set up cycle writer
writer = Turtle()
writer.hideturtle()
writer.penup()
writer.goto(0, 235)
title_font = ("Arial", 50, "bold")

# Set up time progress bar
time_progress_tracker = Turtle()
time_progress_tracker.color('gold')
time_progress_tracker.shape('square')
time_progress_tracker.penup()
time_progress_tracker.left(90)
time_progress_tracker.shapesize(stretch_wid=BLOCK_SIZE*2, stretch_len=BLOCK_SIZE//2, outline=1)
time_progress_tracker.goto(215, -210)

# Set up user generated donor data, and calculate donation data
donor_data, donation_data = machine_setup()
# Update parameters according to user input
PARAMETERS = parameters_updater(donor_data=donor_data, donation_data=donation_data)
# Write donor and donation data on screen
donor_info_writer(donor_data, donation_data)
# Set up pressure bar display
create_pressure_screen(0, PRESSURE_COLORS)
# Set up pressure bar line
pressure_bar = pressure_line_starter()
# Initialise cycle feedback list
feedback_list = []
feedback_status = []
# Sets the scenario for the simulation
scenario = ['draw']
# Corrects the parameters based on the scenario
PARAMETERS = scenario_starter(scenario)
# Initialise an empty list to store the adjustments predicted by the RF model
adjustments = []
# Initialise the output for the draw an return cycle, based on previous adjustments
orchestrator_output = scenario_orchestrator(scenario, adjustments)
# Draw and return cycle counter
cycle_counter = 0

# Simulation loop
for loop in range(11):
    cycle_counter += 1
    print(f'\n    //////////////// CYCLE {cycle_counter} ////////////////    \n')
    orchestrator_output = scenario_orchestrator(scenario, adjustments)
    if loop == 0:
        PARAMETERS = parameters_updater(
            status='running', red_squares=[0], scenarios=orchestrator_output, 
            donation_time=donation_data['predicted_time_min'], loop_count=cycle_counter
            )
    else:
        PARAMETERS = parameters_updater(
            status='running', red_squares=feedback_status, scenarios=orchestrator_output, 
            donation_time=donation_data['predicted_time_min'], loop_count=cycle_counter
            )
    pred = MODEL.predict_proba(params_encoder(PARAMETERS))
    model_output = get_prediction_with_priority(pred[0])
    model_output = model_guardrails(adjustments, model_output, pred[0])
    PARAMETERS = params_updater_from_model(model_output, PARAMETERS)
    adjustments.append(model_output)
    print(f'Orchestrator output: {orchestrator_output[0]} | {orchestrator_output[1]}')
    print(f'\nCycle {cycle_counter} output: {model_output}\n')
    # Handle orchestrator feedback
    if 'low_draw' in orchestrator_output:
        feedback_status.append(1)
    else: 
        feedback_status.append(0)
    if model_output in FATAL_OUTCOMES:
        break
    draw_cycle(writer, title_font, pressure_bar, .1, orchestrator_output)
    return_cycle(writer, title_font, pressure_bar, .1, orchestrator_output)
    time_progress_tracker = update_time(time_progress_tracker)
    feedback_motion(feedback_list)

    feedback_list = cycle_feedback(feedback_list, orchestrator_output)
    screen.update()

    print()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Second cycle
    cycle_counter += 1
    print(f'    //////////////// CYCLE {cycle_counter} ////////////////\n')
    orchestrator_output = scenario_orchestrator(scenario, adjustments)
    PARAMETERS = parameters_updater(
        status='running', red_squares=feedback_status, scenarios=orchestrator_output, 
        donation_time=donation_data['predicted_time_min'], loop_count=cycle_counter
        )
    # model_output = run_inference(params_encoder(PARAMETERS))
    pred = MODEL.predict_proba(params_encoder(PARAMETERS))
    model_output = get_prediction_with_priority(pred[0])
    model_output = model_guardrails(adjustments, model_output, pred[0])
    PARAMETERS = params_updater_from_model(model_output, PARAMETERS)
    # model_output_writer(agent_writer, model_output)
    adjustments.append(model_output)
    print(f'Orchestrator output: {orchestrator_output[0]} | {orchestrator_output[1]}')
    print(f'\nCycle {cycle_counter} output: {model_output}\n')
    # Handle orchestrator feedback
    if 'low_draw' in orchestrator_output:
        feedback_status.append(1)
    else: 
        feedback_status.append(0)
    if model_output in FATAL_OUTCOMES:
        break
    draw_cycle(writer, title_font, pressure_bar, .1, orchestrator_output)
    return_cycle(writer, title_font, pressure_bar, .1, orchestrator_output)
    # Skip very last cycle time update
    if loop != 10:
        time_progress_tracker = update_time(time_progress_tracker)
    feedback_motion(feedback_list)

    feedback_list = cycle_feedback(feedback_list, orchestrator_output)
    screen.update()

# Donation terminated early
if 'End' in adjustments[-1]:
    if 'rinseback' in adjustments[-1]:
        # Write 'rinseback'
        writer.clear()
        status_writer(2, writer, title_font)
    else:
        # Write 'Terminated'
        writer.clear()
        writer.color('red')
        status_writer(3, writer, title_font)
else:
    # Donation terminated normally
    writer.clear()
    status_writer(2, writer, title_font)
# Update screen to display final message
screen.update()
# When done, allows the user to exit the simulation with a click
screen.exitonclick()
