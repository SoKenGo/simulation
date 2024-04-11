import gym
from gym import spaces
import time
from Basilisk.utilities import SimulationBaseClass
from Basilisk.simulation import groundLocation
from Basilisk.simulation import spaceToGroundTransmitter
from Basilisk.simulation import simpleInstrument
from Basilisk.simulation import partitionedStorageUnit
from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from simple_rl.agents import QLearningAgent 

class FederatedClient:
    def __init__(self, model):
        self.model = model
        self.local_updates = None

    def update_local_model(self, global_model):
        self.model.set_weights(global_model.get_weights())

    def train_local_model(self, local_data):
        self.model.fit(local_data['X'], local_data['y'])
        self.local_updates = self.model.get_weights()

    def get_local_updates(self):
        return self.local_updates

class SatelliteEnv(gym.Env):
    def __init__(self, num_clients):
        super(SatelliteEnv, self).__init__()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.create_simulation_task()  # Create simulation task before adding models
        self.scObject = spacecraft.Spacecraft()
        self.initialize_spacecraft()
        
        self.groundStation = groundLocation.GroundLocation()
        self.initialize_ground_station()

        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.instrument = simpleInstrument.SimpleInstrument()
        self.dataMonitor = partitionedStorageUnit.PartitionedStorageUnit()
        self.action_space = spaces.Discrete(5)  # 5 discrete actions
        self.global_model = self.create_global_model()  # Create the global model
        self.clients = [FederatedClient(self.create_local_model()) for _ in range(num_clients)]
        self.num_clients = num_clients
        self.local_data = self.generate_local_data()  # Generate local data for each client

    def create_simulation_task(self):
        task_name = "SimulationTask"
        process_name = "MainProcess"
        update_rate = macros.sec2nano(1)  # Update every 1 second
        proc = self.scSim.CreateNewProcess(process_name)
        proc.addTask(self.scSim.CreateNewTask(task_name, update_rate))

    def initialize_spacecraft(self):
        # Additional spacecraft initialization code placeholder
        pass

    def initialize_ground_station(self):
        self.groundStation.ModelTag = "BoulderGroundStation"
        self.groundStation.specifyLocation(np.radians(40.009971), np.radians(-105.243895), 1624)
        self.groundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.scSim.AddModelToTask("SimulationTask", self.groundStation)

    def step(self, action):
        action_methods = [self.power_level_1, self.power_level_2, self.transmit_data, self.switch_power_mode, self.perform_system_check]
        action_methods[action]()
        
        new_state = self.update_state(action)  # Pass action to update state
        reward = self.calculate_reward(new_state)
        done = self.check_done(new_state)

        return new_state, reward, done, {}


    def reset(self):
        self.scSim.InitializeSimulation()
        return self.get_initial_state()  # Implement this to return initial state

    def update_global_model(self):
        # Perform federated learning model aggregation
        client_updates = []
        for client in self.clients:
            client.update_local_model(self.global_model)
            client.train_local_model(self.local_data[client])
            client_updates.append(client.get_local_updates())

        # Aggregate the client updates to update the global model
        aggregated_updates = self.aggregate_updates(client_updates)
        self.global_model.set_weights(aggregated_updates)

    def create_global_model(self):
        # Create and return the global model architecture
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),  # Assuming input features size of 10
            Dense(64, activation='relu'),
            Dense(5, activation='softmax')  # Assuming 5 actions in the environment
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_local_model(self):
        # Create and return the local model architecture
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(64, activation='relu'),
            Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def aggregate_updates(self, client_updates, weights=None):
        # Perform aggregation of client updates (e.g., averaging)
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)  # Equal weighting if none provided
        new_weights = []
        for layer in range(len(client_updates[0])):  # Assume all clients have the same model architecture
            layer_mean = sum(weight * client_updates[i][layer] for i, weight in enumerate(weights))
            new_weights.append(layer_mean)
        return new_weights

    def generate_local_data(self):
        # Generate local data for each client
        local_data = {}
        for client in self.clients:
            X = np.random.rand(100, 10)  # Assuming 100 samples with 10 features
            y = np.random.randint(5, size=(100,))  # Assuming 5 classes
            local_data[client] = {'X': X, 'y': y}
        return local_data
    
    def get_initial_state(self):
    # Example of a simple initial state dictionary
    # The contents should be adjusted to match your simulation's state variables
        return {
            'spacecraft_position': np.zeros(3),  # Assume 3D position
            'spacecraft_velocity': np.zeros(3),  # Assume 3D velocity
            'ground_station_visibility': False,
            'battery_level': 100,  # Example battery level in percentage
            'data_storage_used': 0  # Example data storage usage in percentage
        }

    # Action methods
    def power_level_1(self):
        print("Battery mode 1!")
        time.sleep(10)

    def power_level_2(self):
        print("Battery mode 2!")
        time.sleep(3)

    def transmit_data(self):
        print("Transmitting data!")
        time.sleep(5)

    def switch_power_mode(self):
        print("Switching power mode!")
        time.sleep(6)

    def perform_system_check(self):
        print("Performing system check!")
        time.sleep(10)

    def update_state(self, action):
        # Get current state
        new_state = self.get_initial_state() 

        # Define effects of each action
        if action == 0:  # Power mode 1 (low power mode, conserves battery)
            new_state['battery_level'] -= 1  # Lower decrement as it saves power
            new_state['data_storage_used'] += 1  # Minimal data accumulation

        elif action == 1:  # Power mode 2 (high power mode, uses more battery)
            new_state['battery_level'] -= 5  # Higher decrement as it uses more power
            new_state['data_storage_used'] += 2  # More data might be processed and stored

        elif action == 2:  # Transmit data (reduces data storage, uses battery)
            new_state['battery_level'] -= 3  # Power needed for transmission
            new_state['data_storage_used'] = max(0, new_state['data_storage_used'] - 10)  # Clear some stored data

        elif action == 3:  # Switch power mode (adjusts power settings, minor battery use)
            new_state['battery_level'] -= 2  # Small power use for switching modes

        elif action == 4:  # Perform system check (uses battery, no change in data storage)
            new_state['battery_level'] -= 4  # Battery used for system checks

        # Update location or movement
        new_state['spacecraft_position'] += np.random.normal(0, 0.1, 3)  # Slight random drift in position

        return new_state


    def calculate_reward(self, new_state):
        # Calculate and return the reward for the new state
        # Placeholder reward calculation
        reward = 0
        if new_state['battery_level'] > 50:
            reward = 1  # Simplistic reward condition
        return reward
    
    def check_done(self, new_state):
        # Determine if the episode should end
        done = False
        if new_state['battery_level'] <= 0:
            done = True  # End episode if battery is depleted
        return done
    
def state_to_tuple(state):
    #Convert the state dictionary to a tuple sorted by keys, 
     #ensuring all numpy arrays are converted to tuples. 
    return tuple((k, tuple(state[k]) if isinstance(state[k], np.ndarray) else state[k]) for k in sorted(state))

# Setup for RL and FL
if __name__ == "__main__":
    num_clients = 5
    env = SatelliteEnv(num_clients)
    agent = QLearningAgent(actions=range(env.action_space.n))
    total_episodes = 100

    for episode in range(total_episodes):
        state = env.reset()  # Get initial state
        state_tuple = state_to_tuple(state)  # Convert state to tuple
        done = False
        reward = 0  # Initial reward

        while not done:
            action = agent.act(state_tuple, reward)  # Use tuple state
            next_state, reward, done, info = env.step(action)
            next_state_tuple = state_to_tuple(next_state)  # Convert next state to tuple
            agent.update(state_tuple, action, reward, next_state_tuple)
            state_tuple = next_state_tuple

        env.update_global_model()  # Update the global model with FL
