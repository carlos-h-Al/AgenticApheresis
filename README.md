# Agentic Components Donation (ACD)

## Introducing agents capable of assisting donors and healthcare staff during apheresis procedures

### The Goal

This project aims to tackle the lack of automation in the healthcare sector, especially in the blood components (apheresis) donations. I created a simulation environment where virtual donors donate platelets. The simulation design was inspired by the **Trima machine** from *Terumo*, which is the machine currently used by NHS Blood and Trasnplant to collect blood components. The goal is to further increase the machine automation by building an AI agent capable of requesting assistance from staff, answering donors queries, flagging donors side effects to nurses and staff members and performing the right adjustments when necessary.

The agent uses a `Random Forest` (RF) classifier, and its inference output correspond to what action the agent needs to take. The RF model was trained based on a set of rules written by an **NHSBT healthcare assistant**, specialised in platelets donations. 

The rules are not as strict as they would be for a live scenario, as this project is meant to be a step towards automation, and it is not meant to bridge the gap entirely.

https://github.com/user-attachments/assets/908561a4-2b60-41f4-b7b9-8b6e40804421

### Running the Simulation

Due to the lack of a real dataset, I generated a synthetic dataset with over 500.000 rows of data. The machine already has a level of automation that was not included in the simulation. As of now, the real machine can regulate draw flow based on each cycle feedback, reducing it or increasing it accordingly.

Once the donor data has been inserted, the simulation begins. The orchestrator is in charge of deciding if the next cycle will have high return and/or low draw. Before each cycle the agent run the RF model to check if any action is required. The orchestrator decision is impacted by the list of outputs that the model generates each cycle. 

**The orchestrator allows the simulation to mimic the impact of an action in a real world scenario.**


<img width="444" height="598" alt="Screenshot 2025-11-20 at 21 28 30" src="https://github.com/user-attachments/assets/b75d6433-9fed-4b8f-9b8e-4f26deb3daaf" />


The agent is also able to perform **RAG**, in case the donor intends to gain more knowldge about platelets donations, or asks questions related to blood components donations. The benefits of RAG include:

* More control over the shared content,
* Lowers the chances of the model hallucinating and giving false information,
* Generate answers with current and up-to-date information,
* Improved scalability,
* Improved trust and explainability.


<img width="882" height="154" alt="Screenshot 2025-11-20 at 21 27 23" src="https://github.com/user-attachments/assets/34a0926f-9395-46a0-ae09-cff1ac2c330e" />



### The Results

The model genralised quite well and achieved **99.59%** accuracy, with recall being extremely high for each class. The results obtained from testing different donor parameters ad the end of `DSS_randomForest.ipynb` indicates that the model performs well in simulation scenarios, as well as on standard accuracy tests.

The next steps to continue this project would be to gain access to an official dataset with donor data, and to add a number of labels. These steps would take this project another step closer to fully-automated donation processes.


<img width="900" height="700" alt="CM" src="https://github.com/user-attachments/assets/ad107fd9-8086-4bd2-be22-2f19e0dce242" />


### Technology Stack

* **Python 3**

* **Ollama:** For running the LLM.

* **Pandas:** For creating the dataset.

* **Matplotlib:** For plotting the confusion matrix scores.

* **Turtle (used in a separate script):** For visually displaying the simulation.

* **LangChain:** For creating a pipeline that connects information, prompts, and LLM together.

* **ChromaDB:** For storing the embeddings for the RAG function.
