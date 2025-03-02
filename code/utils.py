
from hftbacktest import GTX, LIMIT

def get_midprices(hbt, dt, size=None):
    """
    Compute and return the list of mid-prices sampled at a given rate.

    """
    mid_prices = []
    asset_no = 0
    count = 1

    hbt.elapse(1_000_000_000 * 2) # Skip the two first second
    
    while hbt.elapse(1_000_000_000 * dt) == 0:  # Advance the simulation by dt seconds
        depth = hbt.depth(asset_no) 
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        mid_prices.append(mid_price)
        if size:
            if count >= size:
                break
        count+=1

    if size and count < size:
        print(f"Warning: Only {count} mid-prices were collected out of the requested {size}. Not enough price data available.")

    return mid_prices


def place_orders(hbt, action, asset_no, lot_size, tick_size, order_qty):

    depth = hbt.depth(asset_no)
    mid_price = (depth.best_bid + depth.best_ask) / 2.0

    lot_order_qty = order_qty * lot_size

    bid_spread = action[:, 0][0]
    ask_spread = action[:, 1][0]

    bid_price = mid_price - bid_spread
    ask_price = mid_price + ask_spread
    bid_order_id = int(bid_price / tick_size)
    ask_order_id = int(ask_price / tick_size)

    hbt.submit_buy_order(asset_no, bid_order_id, bid_price, lot_order_qty, GTX, LIMIT, False)
    hbt.submit_sell_order(asset_no, ask_order_id, ask_price, lot_order_qty, GTX, LIMIT, False)

    return bid_order_id, ask_order_id


import math

def get_executed_status(hbt, bid_order_id, ask_order_id, asset_no, trading_balance, num_trades):

    num_trades = hbt.state_values(asset_no).num_trades-num_trades
    trading_balance = hbt.state_values(asset_no).balance-trading_balance

    if num_trades == 2:
        return [1, 1]
    if num_trades == 0:
        hbt.cancel(asset_no, bid_order_id, False)
        hbt.cancel(asset_no, ask_order_id, False)
        return [0, 0]
    if num_trades == 1 and trading_balance < 4000:
        hbt.cancel(asset_no, ask_order_id, False)
        return [1, 0]
    if num_trades == 1 and trading_balance > 4000:
        hbt.cancel(asset_no, bid_order_id, False)
        return [0, 1]
    
    "Sometimes num_trades is not in [0, 1, 2] need to fix this"
    return [0, 0]
    

import matplotlib.pyplot as plt
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

class Recorder():
    def __init__(self, step_size, n_steps, order_qty):
        self.order_qty = order_qty
        self.step_size = step_size
        self.n_steps = n_steps
        self.mid_price = []
        self.inventory = []
        self.cash = []
        self.total_value = []
        self.total_value_with_fees = []
        self.time = []

    def record(self, state, hbt):

        asset_no = 0
        lot_size = hbt.depth(asset_no).lot_size
        lot_order_qty = self.order_qty * lot_size

        self.mid_price.append(state[:, ASSET_PRICE_INDEX][0])
        self.inventory.append(state[:, INVENTORY_INDEX][0])
        self.cash.append(state[:, CASH_INDEX][0])
        self.time.append(state[:, TIME_INDEX][0])
        self.total_value.append(state[:, CASH_INDEX][0] + state[:, INVENTORY_INDEX][0]*state[:, ASSET_PRICE_INDEX][0])
        self.total_value_with_fees.append(state[:, CASH_INDEX][0] + state[:, INVENTORY_INDEX][0]*state[:, ASSET_PRICE_INDEX][0] - hbt.state_values(0).fee)

    def plot(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        self.time = [self.step_size * i for i in range(1, self.n_steps + 1)]

        # Graph 1: Mid Price
        axes[0].plot(self.time, self.mid_price, label='Mid Price', color='blue')
        axes[0].set_ylabel('Mid Price')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_title('Mid Price Over Time')
        axes[0].legend()
        axes[0].grid(True)

        # Graph 2: Total Value & Total Value with Fees
        axes[1].plot(self.time, self.total_value, label='Total Value', color='red')
        axes[1].plot(self.time, self.total_value_with_fees, label='Total Value with Fees', color='purple', linestyle='dashed')
        axes[1].set_ylabel('Total Value')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Total Portfolio Value Over Time')
        axes[1].legend()
        axes[1].grid(True)

        # Graph 3: Cash and Inventory with Twin Axis
        ax3 = axes[2]
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Cash', color='green')
        ax3.plot(self.time, self.cash, label='Cash', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        ax4 = ax3.twinx()
        ax4.set_ylabel('Inventory', color='orange')
        ax4.plot(self.time, self.inventory, label='Inventory', color='orange', linestyle='dashed')
        ax4.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title('Cash and Inventory Over Time')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


