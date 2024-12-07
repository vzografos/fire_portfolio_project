import pandas as pd
from typing import Dict, List, Optional, Tuple


class Portfolio:
    def __init__(
        self, name: str, initial_cash: float = 0.0, risk_free_rate: float = 0.02
    ):
        """
        Initialize a portfolio with a name, initial cash, and risk-free rate.

        :param name: Name of the portfolio
        :param initial_cash: Starting cash balance
        :param risk_free_rate: Annual risk-free rate for performance calculations
        """
        self.name = name
        self.holdings: Dict[str, Dict] = (
            {}
        )  # ETF symbol -> {shares, cost_basis, purchase_dates}
        self.cash: float = initial_cash
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.risk_free_rate = risk_free_rate
        self.transactions_log: List[Dict] = []

        # Performance tracking
        self.initial_value: float = initial_cash
        self.creation_date = pd.Timestamp.now()

    def add_holding(
        self,
        symbol: str,
        shares: float,
        cost_basis: float,
        commission: float = 0.0,
        purchase_date: Optional[pd.Timestamp] = None,
        affect_cash: bool = True,
    ):
        """
        Add or update a holding in the portfolio with commission consideration.

        :param symbol: ETF ticker symbol
        :param shares: Number of shares
        :param cost_basis: Purchase price per share
        :param commission: Total broker commission for the trade
        :param purchase_date: Date of purchase (defaults to current date)
        :param affect_cash: Whether to reduce cash balance for this transaction
        """
        if shares <= 0 or cost_basis <= 0:
            raise ValueError("Shares and cost basis must be positive.")

        purchase_date = purchase_date or pd.Timestamp.now()

        # Calculate total trade cost including commission
        total_trade_cost = (shares * cost_basis) + commission

        # Adjust effective cost basis to include commission spread across shares
        effective_cost_basis = total_trade_cost / shares

        # Determine transaction type based on affect_cash flag
        transaction_type = "buy" if affect_cash else "transfer_in"
        if symbol in self.holdings:
            transaction_type = "add" if affect_cash else "transfer_add"

        # Log the transaction with commission details
        transaction = {
            "type": transaction_type,
            "symbol": symbol,
            "shares": shares,
            "cost_basis": cost_basis,
            "commission": commission,
            "effective_cost_basis": effective_cost_basis,
            "date": purchase_date,
            "total_cost": total_trade_cost,
        }
        self.transactions_log.append(transaction)

        # Update or add holding with effective cost basis
        if symbol in self.holdings:
            total_shares = self.holdings[symbol]["shares"] + shares
            total_cost = (
                self.holdings[symbol]["shares"] * self.holdings[symbol]["cost_basis"]
            ) + total_trade_cost
            new_effective_cost_basis = total_cost / total_shares

            self.holdings[symbol] = {
                "shares": total_shares,
                "cost_basis": new_effective_cost_basis,
                "purchase_dates": self.holdings[symbol].get("purchase_dates", [])
                + [purchase_date],
            }
        else:
            self.holdings[symbol] = {
                "shares": shares,
                "cost_basis": effective_cost_basis,
                "purchase_dates": [purchase_date],
            }

        # Conditionally deduct commission from cash balance
        if affect_cash:
            self.cash -= total_trade_cost

    def remove_holding(
        self,
        symbol: str,
        shares: Optional[float] = None,
        current_price: float = 0.0,
        commission: float = 0.0,
        transaction_date: Optional[pd.Timestamp] = None,
        affect_cash: bool = True,
    ):
        """
        Remove all or part of a holding from the portfolio.

        :param symbol: ETF ticker symbol
        :param shares: Number of shares to remove (None removes entire holding)
        :param current_price: Current market price of the stock
        :param commission: Broker commission for the sale
        :param affect_cash: Whether to increase cash balance for this transaction
        :param transaction_date: Date of the transaction (defaults to current date)
        """
        if symbol not in self.holdings:
            raise ValueError(f"No holding found for {symbol}")

        if shares > self.holdings[symbol]["shares"]:
            raise ValueError(f"Not enough shares to sell for {symbol}.")

        if current_price <= 0:
            raise ValueError("Current price must be positive.")

        # Use provided date or current date
        transaction_date = transaction_date or pd.Timestamp.now()

        # Determine shares to sell
        if shares is None or shares >= self.holdings[symbol]["shares"]:
            # Full sale
            shares = self.holdings[symbol]["shares"]
            is_full_sale = True
        else:
            is_full_sale = False

        # Calculate sale proceeds and realize gains/losses
        cost_basis = self.holdings[symbol]["cost_basis"]

        sale_proceeds = shares * current_price
        net_proceeds = sale_proceeds - commission
        realized_gain = sale_proceeds - (shares * cost_basis)

        # Determine transaction type based on affect_cash flag
        transaction_type = "sell" if affect_cash else "transfer_out"

        # Prepare transaction log
        transaction = {
            "type": transaction_type,
            "symbol": symbol,
            "shares": shares,
            "cost_basis": cost_basis,
            "current_price": current_price,
            "commission": commission,
            "date": transaction_date,
            "net_proceeds": net_proceeds,
            "realized_gain": realized_gain,
        }
        self.transactions_log.append(transaction)

        # Conditionally update cash balance
        if affect_cash:
            self.cash += net_proceeds

        # Update holdings
        if is_full_sale:
            del self.holdings[symbol]
        else:
            # Partial sale
            self.holdings[symbol]["shares"] -= shares

            # Optionally, you might want to recalculate average cost basis for remaining shares
            # This is a simplified approach and might need more sophisticated handling
            self.holdings[symbol]["cost_basis"] = cost_basis

        return transaction

    def calculate_portfolio_performance(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics.

        :param current_prices: Current market prices for holdings
        :return: Dictionary of performance metrics
        """
        # Current portfolio value
        current_value = self.get_current_value(current_prices)

        # Calculate total return
        total_return = (current_value - self.initial_value) / self.initial_value

        # Calculate gains/losses
        unrealized_gains = {}
        for symbol, holding in self.holdings.items():
            current_value = holding["shares"] * current_prices[symbol]
            cost_basis = holding["shares"] * holding["cost_basis"]
            unrealized_gains[symbol] = current_value - cost_basis

        return {
            "total_value": current_value,
            "initial_value": self.initial_value,
            "total_return_percentage": total_return * 100,
            "unrealized_gains": unrealized_gains,
            "cash": self.cash,
            "holdings_count": len(self.holdings),
        }

    def rebalance(
        self, target_weights: Dict[str, float], current_prices: Dict[str, float]
    ):
        """
        Rebalance portfolio to target weights.

        :param target_weights: Desired portfolio weights
        :param current_prices: Current market prices
        """
        total_value = self.get_current_value(current_prices)

        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_value = self.holdings.get(symbol, {"shares": 0})[
                "shares"
            ] * current_prices.get(symbol, 0)

            if target_value > current_value:
                # Need to buy
                shares_to_buy = (target_value - current_value) / current_prices[symbol]
                self.add_holding(symbol, shares_to_buy, current_prices[symbol])
            elif target_value < current_value:
                # Need to sell
                shares_to_sell = (current_value - target_value) / current_prices[symbol]
                self.remove_holding(symbol, shares_to_sell)

    def get_current_value(
        self, current_prices: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate the current value and the cost basis of the portfolio."""

        # If no current prices provided, use placeholder
        if current_prices is None:
            current_prices = {
                symbol: holding["cost_basis"]
                for symbol, holding in self.holdings.items()
            }

        total_portfolio_value = self.cash
        total_cost_basis = 0

        # Iterate through holdings
        for symbol, holding in self.holdings.items():
            shares = holding["shares"]
            cost_basis = holding["cost_basis"]
            current_price = current_prices.get(symbol, cost_basis)

            market_value = shares * current_price
            total_cost_basis_for_holding = shares * cost_basis
            gain_loss = market_value - total_cost_basis_for_holding

            # Print holding details
            print(
                f"{symbol:<10} {shares:<10.2f} €{cost_basis:<14.2f} €{current_price:<14.2f} €{market_value:<14.2f} €{gain_loss:<14.2f}"
            )

            total_portfolio_value += market_value
            total_cost_basis += total_cost_basis_for_holding

        return total_portfolio_value, total_cost_basis

    def get_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate the current weights of each holding in the portfolio."""
        total_value = self.get_current_value(current_prices)
        return {
            symbol: (self.holdings[symbol]["shares"] * current_prices[symbol])
            / total_value
            for symbol in self.holdings
        }

    def add_cash(
        self,
        amount: float,
        transaction_date: Optional[pd.Timestamp] = None,
        source: Optional[str] = None,
    ):
        """
        Add cash to the portfolio.

        :param amount: Amount of cash to add
        :param transaction_date: Optional date of the transaction
        :param source: Optional description of cash source (e.g., 'dividend', 'deposit', 'sale')
        """
        if amount < 0:
            raise ValueError("Cash amount must be positive")

        self.cash += amount

        transaction_date = transaction_date or pd.Timestamp.now()

        # Log the cash addition
        transaction = {
            "type": "cash_deposit",
            "amount": amount,
            "source": source,
            "date": transaction_date,
        }
        self.transactions_log.append(transaction)

    def withdraw_cash(
        self,
        amount: float,
        transaction_date: Optional[pd.Timestamp] = None,
        purpose: Optional[str] = None,
    ):
        """
        Withdraw cash from the portfolio.

        :param amount: Amount of cash to withdraw
        :param transaction_date: Optional date of the transaction
        :param purpose: Optional description of cash withdrawal purpose
        :raises ValueError: If insufficient funds
        """
        if amount < 0:
            raise ValueError("Withdrawal amount must be positive")

        if amount > self.cash:
            raise ValueError(f"Insufficient funds. Current balance: {self.cash}")

        self.cash -= amount
        transaction_date = transaction_date or pd.Timestamp.now()

        # Log the cash withdrawal
        transaction = {
            "type": "cash_withdrawal",
            "amount": amount,
            "purpose": purpose,
            "date": transaction_date,
        }
        self.transactions_log.append(transaction)

    def display_portfolio(self, current_prices: Optional[Dict[str, float]] = None):
        """
        Display detailed portfolio information.

        :param current_prices: Optional dictionary of current market prices
        """
        print(f"Portfolio: {self.name}")
        print("=" * 50)

        # Display cash
        print(f"Cash Balance: €{self.cash:,.2f}")

        # Display Holdings
        print("\nHOLDINGS:")
        print("-" * 50)

        # Table header
        print(
            f"{'Symbol':<10} {'Shares':<10} {'Cost Basis':<15} {'Current Price':<15} {'Market Value':<15} {'Gain/Loss':<15}"
        )
        print("-" * 80)

        total_portfolio_value, total_cost_basis = self.get_current_value(current_prices)

        # Portfolio Summary
        print("\nPORTFOLIO SUMMARY:")
        print("-" * 50)
        print(f"Total Portfolio Value:  €{total_portfolio_value:,.2f}")
        print(f"Total Cost Basis:       €{total_cost_basis:,.2f}")
        print(
            f"Total Unrealized G/L:   €{total_portfolio_value - total_cost_basis:,.2f}"
        )
        print(f"Number of Holdings:     {len(self.holdings)}")

    def print_transaction_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        """
        Print the portfolio's transaction history with optional filtering.

        :param symbol: Optional stock symbol to filter transactions
        :param start_date: Optional start date for filtering transactions
        :param end_date: Optional end date for filtering transactions
        """
        print(f"Transaction History for Portfolio: {self.name}")
        print("=" * 60)

        # Filter transactions based on parameters
        filtered_transactions = self.transactions_log.copy()

        # Filter by symbol if provided
        if symbol:
            filtered_transactions = [
                transaction
                for transaction in filtered_transactions
                if transaction.get("symbol") == symbol
            ]

        # Filter by date range if provided
        if start_date:
            filtered_transactions = [
                transaction
                for transaction in filtered_transactions
                if transaction["date"] >= start_date
            ]

        if end_date:
            filtered_transactions = [
                transaction
                for transaction in filtered_transactions
                if transaction["date"] <= end_date
            ]

        # Convert all date strings to Timestamp objects
        for transaction in filtered_transactions:
            if isinstance(transaction["date"], str):
                transaction["date"] = pd.Timestamp(transaction["date"])

        # Sort the transactions
        filtered_transactions.sort(key=lambda x: x["date"])

        # Check if no transactions found
        if not filtered_transactions:
            print("No transactions found.")
            return

        # Print table header
        print(f"{'Date':<20} {'Type':<15} {'Symbol':<10} {'Details':<30}")
        print("-" * 75)

        # Print each transaction
        for transaction in filtered_transactions:
            # Format details based on transaction type
            if transaction["type"] in ["buy", "add", "transfer_in", "transfer_add"]:
                details = f"Shares: {transaction.get('shares', 'N/A'):.2f}, Cost Basis: €{transaction.get('cost_basis', 0):.2f}"
            elif transaction["type"] in ["sell", "transfer_out"]:
                details = f"Shares: {transaction.get('shares', 'N/A'):.2f}, Sale Price: €{transaction.get('current_price', 0):.2f}"
            elif transaction["type"] in ["cash_deposit", "cash_withdrawal"]:
                details = f"Amount: €{transaction.get('amount', 0):.2f}, Source: {transaction.get('source', transaction.get('purpose', 'N/A'))}"
            else:
                details = str(transaction)

            # Print transaction row
            print(
                f"{transaction['date'].strftime('%Y-%m-%d %H:%M'):<20} "
                f"{transaction['type']:<15} "
                f"{transaction.get('symbol', 'N/A'):<10} "
                f"{details:<30}"
            )

        # Print summary
        print("\nSUMMARY:")
        print(f"Total Transactions: {len(filtered_transactions)}")
