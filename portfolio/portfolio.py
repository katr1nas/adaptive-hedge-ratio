import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from data.loader import load_data


class Strategy:
    def __init__(self, symbols, start, end, entry=2.0, exit_=0.5, cost=0.0005, train_frac=0.6, stop=4.0, initial_capital=10000, use_kalman=False):
        if len(symbols) != 2:
            raise ValueError("Strategy requires exactly two symbols")
        self.symbols = symbols
        self.start = start
        self.end = end
        self.entry = entry
        self.exit_ = exit_
        self.cost = cost
        self.stop = stop
        self.train_frac = train_frac
        self.init_capital = initial_capital
        self.use_kalman = use_kalman

    def get_signals(self):
        data = load_data(self.symbols, self.start, self.end)
        close = data['Close'].squeeze()

        self.la = np.log(close[self.symbols[0]])
        self.lb = np.log(close[self.symbols[1]])
        self.split = int(len(self.la) * self.train_frac)

        X = sm.add_constant(self.lb[:self.split])
        self.beta = sm.OLS(self.la[:self.split], X).fit().params.iloc[1]
        self.spread = self.la - self.beta * self.lb

        train = self.spread[:self.split]
        self.adf_p = adfuller(train)[1]
        self.coint_p = coint(self.la[:self.split], self.lb[:self.split])[1]

        d = train.diff().dropna()
        lag = train.shift(1).dropna()
        phi = sm.OLS(d, sm.add_constant(lag)).fit().params.iloc[1]
        self.half_life = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.nan

        window = int(round(2 * self.half_life)) if np.isfinite(self.half_life) else 60
        self.window = max(window, 10)

        mean = self.spread.rolling(self.window).mean()
        std = self.spread.rolling(self.window).std()
        self.z_score = (self.spread - mean) / std

        self.signals = pd.Series(0, index=self.z_score.index)
        self.signals[self.z_score > self.entry] = -1
        self.signals[self.z_score < -self.entry] = 1

        if self.use_kalman:
            return self.kalman_filter()


        return self.z_score, self.signals

    def order(self):
        position = 0
        positions = []

        for z, signal in zip(self.z_score.values, self.signals.values):
            if position == 0:
                if signal != 0 and abs(z) < self.stop:
                    position = signal
            elif position == 1 and (z > -self.exit_ or z < -self.stop):
                position = 0
            elif position == -1 and (z < self.exit_ or z > self.stop):
                position = 0
            positions.append(position)

        self.positions = pd.Series(positions, index=self.z_score.index)
        return self.positions

    def backtest(self):
        beta = self.beta_t if self.use_kalman else self.beta
        ret = self.la.diff() - beta * self.lb.diff()
        pos = self.positions.shift(1)
        trades = self.positions.diff().abs().shift(1)
        self.pnl = (pos * ret - trades * self.cost).fillna(0)

        test = self.pnl.iloc[self.split:]
        sharpe = test.mean() / test.std() * np.sqrt(252) if test.std() > 0 else np.nan

        return {
            "beta": self.beta,
            "adf_p": self.adf_p,
            "coint_p": self.coint_p,
            "half_life": self.half_life,
            "window": self.window,
            "test_sharpe": sharpe,
            "test_return": test.sum(),
            "test_trades": int(self.positions.iloc[self.split:].diff().abs().sum() / 2),
        }
    

    def kalman_filter(self, delta=1e-4, warmup=50):
        la = self.la.values - self.la[:self.split].mean()
        lb = self.lb.values - self.lb[:self.split].mean()

        X = sm.add_constant(lb[:self.split])
        fitted = sm.OLS(la[:self.split], X).fit()
        state = np.array([fitted.params[0], fitted.params[1]])

        P = np.eye(2) * 1e-5
        Vw = np.diag([0.0, delta / (1 - delta)])
        Ve = fitted.resid.var()

        betas_prior = []
        errors = []
        Qs = []

        for t in range(len(la)):
            x = np.array([1.0, lb[t]])
            R = P + Vw

            betas_prior.append(state[1])

            e = la[t] - x @ state
            errors.append(e)

            Q = x @ R @ x + Ve
            Qs.append(Q)

            K = R @ x / Q
            state = state + K * e
            P = R - np.outer(K, x) @ R

        idx = self.la.index
        self.beta_t = pd.Series(betas_prior, index=idx)
        e_s = pd.Series(errors, index=idx)
        self.z_score = e_s / e_s.rolling(self.window).std()

        self.signals = pd.Series(0, index=idx)
        self.signals[self.z_score > self.entry] = -1
        self.signals[self.z_score < -self.entry] = 1

        return self.z_score, self.signals


        




    

    
    




