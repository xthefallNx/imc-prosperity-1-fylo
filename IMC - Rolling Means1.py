from ast import ParamSpec
from prosperity3bt.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import pandas as pd
import numpy as np
import string

#IM WRITING COMMENTS TO UNDERSTAND THE FINANCE -aileen

class Product:
     good1= 'KELP'
     good2 = 'RAINFOREST_RESIN'
     good3 = 'SQUID_INK'
params = {
    Product.good1: {
        #wait we definitely need to estimate fair_value based on the bid and ask prices/ microprices
        #and not just setting it at a value? i think it should be dynamic
        "fair_value": 2500,

        #basically if it dips 1 below market price then we buy immediately. if it goes 1 above market price we sell immediately
        "take_width": 1,

        #amt we are willing to buy/sell at more/less than market value to clear inventory
        #i feel like we should try making this 1-2 to get rid of more inventory-Aileen
        "clear_width": 0,

        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge

        "default_edge": 4,
        #risk control-- our position  can't be 25 lower or higher than market value
        "soft_position_limit": 25}
}
bids = []
asks = []
class Trader:

    def __init__(self, params=None):
        # Use module-level params as default
        self.params = params if params is not None else {
            Product.good1: {
                "fair_value": 2500,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 15
            },
            Product.good2: {  # Add default params for good2
                "fair_value": 1500,  # Example value
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 1,
                "default_edge": 2,
                "soft_position_limit": 25
            }
        }

        self.LIMIT = {Product.good1: 50, Product.good2: 50}
        self.bids = []
        self.asks = []

    def take_best_orders( # basically trades if above or below the fair value plus/minum take width
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume
    def take_orders( #runs the above function basically
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    def clear_position_order( #clears orders at a fair price
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    def clear_orders( #runs the above function
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    def make_orders( #market making algorithm
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def means_reversion_buy(self, price, orders, ask, position_size, order_size): #means reversion
        mavg_30 = np.mean(price[-30:]-10000)
        std_30 = np.std(price[-30:]-10000)
        z_score = (ask - mavg_30) / std_30
        if position_size < 50:
            if z_score < -0.5:
                orders.append(Order, ask, order_size)
            elif order_size > 50-position_size:
                orders.append(Order, ask, 50-position_size)
            elif abs(z_score) <= 0.1:
                orders.append(Order, ask, -(position_size))

        return orders

    def means_reversion_sell(self, price, orders, bid, position_size, order_size):
        mavg_30 = np.mean(price[-30:]-10000)
        std_30=np.std(price[-30:]-10000)
        z_score = (bid - mavg_30) / std_30
        if position_size >= -50:
            if z_score > 0.5:
                orders.append(Order, bid, -order_size)
            elif -order_size < -50 + position_size:
                orders.append(Order, bid, -50 + position_size)
            elif abs(z_score) <= 0.1:
                orders.append(Order, bid, -(position_size))
        return orders
    def run(self, state: TradingState): #runs code
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.good1 in self.params and Product.good1 in state.order_depths:
            good1_position = (
                state.position.get(Product.good1, 0)
            if state.position is not None
            else 0
            )
        if Product.good1 in self.params and Product.good1 in state.order_depths:
            good2_position = (
                state.position.get(Product.good2, 0)
            if state.position is not None
            else 0)

            good1_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.good1,
                    state.order_depths[Product.good1],
                    self.params[Product.good1]["fair_value"],
                    self.params[Product.good1]["take_width"],
                    good1_position,
                )
            )
            good1_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.good1,
                    state.order_depths[Product.good1],
                    self.params[Product.good1]["fair_value"],
                    self.params[Product.good1]["clear_width"],
                    good1_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            good1_make_orders, _, _ = self.make_orders(
                Product.good1,
                state.order_depths[Product.good1],
                self.params[Product.good1]["fair_value"],
                good1_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.good1]["disregard_edge"],
                self.params[Product.good1]["join_edge"],
                self.params[Product.good1]["default_edge"],
                True,
                self.params[Product.good1]["soft_position_limit"],
            )
            result[Product.good1] = (
                good1_take_orders + good1_clear_orders + good1_make_orders)
        if Product.good2 in self.params and Product.good2 in state.order_depths:
            good1_position = (
            state.position.get(Product.good2, 0)
            if state.position is not None
            else 0
            )
            
            
        means_buy_orders = []
        means_sell_orders = []
        self.bids.append(min(state.order_depths[Product.good2].sell_orders.values()))
        self.asks.append(max(state.order_depths[Product.good2].buy_orders.values()))
        if len(self.bids) > 30 and len(self.asks) > 30:
            if len(state.order_depths[Product.good2].sell_orders) and len(state.order_depths[Product.good2].buy_orders) != 0:
                for entry in [state.order_depths[Product.good2].sell_orders.items()]:
                    for best_ask, best_ask_amount in entry:
                        price = pd.Series(asks)
                        means_sell_orders.append(self.means_reversion_sell(price, means_sell_orders, best_ask, good2_position, best_ask_amount))
                       
		    # String value holding Trader state data required.
				# It will be delivered as TradingState.traderData on next execution.
                for entry in [state.order_depths[Product.good2].buy_orders.items()]:
                    for best_bid, best_bid_amount in entry:
                        price = pd.Series(bids)
                        means_buy_orders.append(self.means_reversion_buy(price, means_buy_orders, best_bid, good2_position, best_bid_amount))
            result[Product.good2] = (means_sell_orders + means_buy_orders)
        traderData = "SAMPLE"

				# Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
