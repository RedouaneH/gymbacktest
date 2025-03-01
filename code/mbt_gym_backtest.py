
import sys
import gym    
import numpy as np
from typing import Optional
from collections import OrderedDict
from typing import Union, Tuple, Callable
sys.path.append("../")
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel
from mbt_gym.gym.ModelDynamics import ModelDynamics
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel
from mbt_gym.gym.info_calculators import InfoCalculator
from mbt_gym.rewards.RewardFunctions import RewardFunction
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX

class HistoricalMidpriceModel(StochasticProcessModel):
    def __init__(
        self,
        midprice_list: list, # length of the list : int(terminal_time/step_size)
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        if not midprice_list:
            raise ValueError("midprice_list must not be empty")
        self.midprice_list = midprice_list
        self.current_index = 0
        initial_price = midprice_list[0]

        
        super().__init__(
            min_value=np.array([[min(midprice_list)]]),
            max_value=np.array([[max(midprice_list)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    
    def reset(self):
        # Reset the pointer along with the state.
        self.current_index = 0
        super().reset()
    
    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        # Check if already at the last midprice
        if self.current_index >= len(self.midprice_list):
            raise ValueError("Cannot update: Already at the latest midprice in the list.")
        
        self.current_midprice = self.midprice_list[self.current_index]
        # Update the state to use the current midprice for all trajectories
        self.current_state = np.full((self.num_trajectories, 1), self.current_midprice)

        # Move to the next midprice
        self.current_index += 1


class BackTestingLimitOrderModelDynamics(ModelDynamics):
    """
    Instead of relying on an 'arrival_model' and a 'fill_probability_model' to determine 
    whether an order is executed, this model directly takes an 'executed' array. 
    The 'executed' array has the same shape as the action array and contains 1 if the order 
    was executed and 0 otherwise.
    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth: float = None,
    ):
        super().__init__(
            midprice_model=midprice_model,
            arrival_model=arrival_model,
            fill_probability_model=fill_probability_model,
            num_trajectories=num_trajectories,
            seed=seed,
        )
        self.max_depth = max_depth or self._get_max_depth()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def update_state(self, action: np.ndarray, executed: np.ndarray):
        """Updates the system state based on executed orders.

        Args:
            executed (np.ndarray): Array indicating order execution (1 if executed, 0 otherwise).
            action (np.ndarray): Action array, where the first two columns represent bid-ask spreads.
        """
        # Update inventory and cash, replacing (arrivals * fills) with 'executed'.
        self.state[:, INVENTORY_INDEX] += np.sum(executed * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
            self.fill_multiplier
            * executed
            * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
            axis=1,
        )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders, max_depth cannot be None."
        # The agent chooses bid and ask spreads.
        return gym.spaces.Box(low=np.float32(0.0), high=np.float32(self.max_depth), shape=(2,))

    def get_required_stochastic_processes(self):
        # No stochastic processes are required since execution is provided directly.
        return []

    def get_arrivals_and_fills(self, action: np.ndarray):
        # This method is not used in this model since execution is given via 'executed'.
        return None, None




class BackTestingTradingEnvironment(TradingEnvironment):
    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        reward_function: RewardFunction = None,
        model_dynamics: ModelDynamics = None,
        initial_cash: float = 0.0,
        initial_inventory: Union[int, Tuple[float, float]] = 0,  # Either a deterministic initial inventory, or a tuple
        max_inventory: int = 10_000,  # representing the mean and variance of it.
        max_cash: float = None,
        max_stock_price: float = None,
        start_time: Union[float, int, Callable] = 0.0,
        info_calculator: InfoCalculator = None,  # episode given as a proportion.
        seed: int = None,
        num_trajectories: int = 1,
        normalise_action_space: bool = True,
        normalise_observation_space: bool = True,
        normalise_rewards: bool = False,
    ):
        super().__init__(
            terminal_time=terminal_time,
            n_steps=n_steps,
            reward_function=reward_function,
            model_dynamics=model_dynamics,
            initial_cash=initial_cash,
            initial_inventory=initial_inventory,
            max_inventory=max_inventory,
            max_cash=max_cash,
            max_stock_price=max_stock_price,
            start_time=start_time,
            info_calculator=info_calculator,
            seed=seed,
            num_trajectories=num_trajectories,
            normalise_action_space=normalise_action_space,
            normalise_observation_space=normalise_observation_space,
            normalise_rewards=normalise_rewards,
        )

    def reset(self):
        """
        This reset function also returns the non-normalized model_dynamics state, which is necessary for plotting the real values.
        """
        for process in self.stochastic_processes.values():
            process.reset()
        self.model_dynamics.state = self.initial_state
        self.reward_function.reset(self.model_dynamics.state.copy())
        return self.normalise_observation(self.model_dynamics.state.copy()), self.model_dynamics.state.copy()

    def step(self, action: np.ndarray, executed: np.ndarray):
        """
        Steps the environment forward using both the action and the executed orders.

        Args:
            action (np.ndarray): The agent's action.
            executed (np.ndarray): An array of the same shape as action indicating order execution (1 if executed, 0 otherwise).

        Returns:
            normalized_observation (np.ndarray): The new normalized observation.
            observation (np.ndarray): The new observation.
            reward (np.ndarray): The (normalized) reward.
            done (np.ndarray): The done flag.
            info (dict): Additional info.
        """
        action = self.normalise_action(action, inverse=True)
        current_state = self.model_dynamics.state.copy()
        next_state = self._update_state(action, executed)
        dones = self._get_dones()
        rewards = self.reward_function.calculate(current_state, action, next_state, dones[0])
        infos = self._calculate_infos(current_state, action, rewards)
        return self.normalise_observation(next_state.copy()), next_state.copy(), self.normalise_rewards(rewards), dones, infos

    def _update_state(self, action: np.ndarray, executed: np.ndarray) -> np.ndarray:
        """
        Updates the state using executed orders instead of arrivals and fills.

        Args:
            action (np.ndarray): The agent's action.
            executed (np.ndarray): An array indicating order execution.
        
        Returns:
            np.ndarray: The updated state.
        """
        self._update_agent_state(None, None, action, executed)
        self._update_market_state(None, None, action)
        return self.model_dynamics.state

    def _update_market_state(self, arrivals, fills, action):
        """
        Updates the market state by calling the update method on each stochastic process.
        For processes, we pass the executed orders as a proxy for arrivals and fills.
        
        Args:
            executed (np.ndarray): An array indicating order execution.
            action (np.ndarray): The agent's action.
        """
        for process_name, process in self.stochastic_processes.items():
            process.update(action, arrivals, fills, self.model_dynamics.state)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            self.model_dynamics.state[:, lower_index:upper_index] = process.current_state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, executed: np.ndarray):
        """
        Updates the agent state using the executed orders.

        Args:
            executed (np.ndarray): An array indicating order execution.
            action (np.ndarray): The agent's action.
        """
        self.model_dynamics.update_state(action, executed)
        self._clip_inventory_and_cash()
        self.model_dynamics.state[:, TIME_INDEX] += self.step_size

    def _get_stochastic_processes(self):
        stochastic_processes = dict()
        for process_name in ["midprice_model"]:
            process: StochasticProcessModel = getattr(self.model_dynamics, process_name)
            if process is not None:
                stochastic_processes[process_name] = process
        return OrderedDict(stochastic_processes)

    def _get_stochastic_process_indices(self):
        process_indices = dict()
        count = 3
        for process_name, process in self.stochastic_processes.items():
            dimension = int(process.initial_vector_state.shape[1])
            process_indices[process_name] = (count, count + dimension)
            count += dimension
        return OrderedDict(process_indices)
