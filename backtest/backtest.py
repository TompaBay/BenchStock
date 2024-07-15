import numpy as np
import pandas as pd


class Backtest():
    def __init__(self, cash, topk=50, n_drop=5, trade_cost = 0.0005) -> None:
        self.position = None
        self.topk = topk
        self.n_drop = n_drop
        self.cash = cash
        self.trade_cost = trade_cost
        self.cum_value = cash
        self.date = None
        self.values = []
        self.positions = []
        self.buy_cost = 0.0015
        self.sell_cost = 0.0025


    def trade_us(self, df, full=False):
        df.sort_values(by=['pred'], inplace=True)
        df.dropna(inplace=True)
        date = df['date'].to_numpy()[0]
        
        if self.position is None:
            cash_per_stock = self.cash / self.topk
            df = df[df['DlyVol'] > 100000]
            df['share'] = cash_per_stock // (df['DlyPrc'] + self.trade_cost)
            df = df[df['share'] < df['DlyVol']]
            self.position = df.iloc[-self.topk:]
<<<<<<< HEAD
=======
            # self.position['share'] = cash_per_stock // (self.position['DlyPrc'] + self.trade_cost)
            # self.position = self.position[self.position['share'] < self.position['DlyVol']]
>>>>>>> 109947ec0ece403a1d40a194a92ed80e2db0d46f
            self.position['value'] = self.position['share'] * self.position['DlyPrc']
            self.position.fillna(0, inplace=True)
            self.values.append(self.cash)
            self.cash -= np.sum(self.position['value'].to_numpy())
            self.cash -= np.sum(self.position['share'].to_numpy()) * self.trade_cost
        else:
            #Update information of holding stocks
            self.position['value'] *= (1 + self.position['label'])
            self.position = self.position.loc[:, ['code', 'share', 'value']]
            self.position = self.position.merge(df, how="left", on="code")
            non_sell = self
            self.position.sort_values(by=['pred'], inplace=True)

            if full:
                sell = self.position[self.position['SecurityEndDt'] == date]
                self.cash += np.sum(sell['value'].to_numpy())
                self.cash -= np.sum(sell['share'].to_numpy()) * self.trade_cost
                self.position = self.position[self.position['SecurityEndDt'] != date]
                

            non_sell = self.position[self.position.isnull().any(axis=1)]
            for_sell = self.position[~self.position.isnull().any(axis=1)]
            

            # Sell stocks
            sell = for_sell.iloc[:self.n_drop]
            self.cash += np.sum(sell['value'].to_numpy())
            self.cash -= np.sum(sell['share'].to_numpy()) * self.trade_cost
            self.position = for_sell[self.n_drop:]
            if not non_sell.empty:
                self.position = pd.concat([self.position, non_sell])
            self.values.append(self.cash + np.sum(self.position['value'].to_numpy()))

            # Buy stocks
            df = df[df['DlyVol'] > 100000]
            buy_amount = self.topk - len(self.position)
            cash_per_stock = self.cash / buy_amount
            buy = df[~df['code'].isin(set(self.position['code']))]
            buy['share'] = cash_per_stock // (buy['DlyPrc'] + self.trade_cost)
            buy = buy[buy['share'] < buy['DlyVol']]
            buy = buy[-buy_amount:]
            buy['value'] = buy['share'] * buy['DlyPrc']
            self.cash -= np.sum(buy['value'].to_numpy())
            self.cash -= np.sum(buy['share'].to_numpy()) * self.trade_cost
            self.position = pd.concat([self.position, buy])
            self.position.fillna(0, inplace=True)

        self.cum_value = self.cash + np.sum(self.position['value'].to_numpy())
        self.positions.append(self.position)


    def trade_cn(self, df):
        df.sort_values(by=['pred'], inplace=True)
        df.dropna(inplace=True)

        if self.position is None:
            cash_per_stock = self.cash / self.topk
            # Filter suspended stocks and those trade volume that is too small
            df = df[(df['DlyVol'] > 100000) & (df['s_dq_tradestatus'] != '停牌')]
            df = df[df['close'] < 0.095]

            buy_share = cash_per_stock // (df['DlyPrc'].to_numpy() * (1 + self.buy_cost))
            buy_share -= buy_share % 100
            df['share'] = buy_share
            # 恢复
            df = df[df['share'] < df['DlyVol']]
            self.position = df.iloc[-self.topk:]
            self.position['value'] = self.position['share'] * self.position['DlyPrc']
            self.position.fillna(0, inplace=True)
            self.values.append(self.cash)
            self.cash -= np.sum(self.position['value'].to_numpy()) * (1 + self.buy_cost)

        else:
            #Update information of holding stocks
            self.position['value'] *= (1 + self.position['label'])
            self.position = self.position.loc[:, ['code', 'share', 'value']]

            self.position = self.position.merge(df, how="left", on="code")
            self.position.fillna(0, inplace=True)
            self.position.sort_values(by=['pred'], inplace=True)
            
            # Sell stocks
            non_sell = self.position[(self.position['s_dq_tradestatus'] == '停牌') | (self.position['close'] <= -0.095)]
            for_sell = self.position[(self.position['s_dq_tradestatus'] != '停牌') & (self.position['close'] > -0.095)]
             
            sell = for_sell.iloc[:self.n_drop]
            self.cash += np.sum(sell['value'].to_numpy())
            self.cash -= np.sum(sell['value'].to_numpy()) * self.sell_cost
            self.position = pd.concat([for_sell.iloc[self.n_drop:], non_sell])
            self.values.append(self.cash + np.sum(self.position['value'].to_numpy()))

            # Buy stocks
            df = df[(df['DlyVol'] > 100000) & (df['s_dq_tradestatus'] != '停牌')]
            df = df[df['close'] < 0.095]
            buy_amount = self.topk - len(self.position)
            cash_per_stock = self.cash / buy_amount
            buy = df[~df['code'].isin(set(self.position['code']))]           

            # # A-share can only buy multiples of 100 shares before 2023
            buy_share = cash_per_stock // (buy['DlyPrc'].to_numpy() * (1 + self.buy_cost))
            buy_share -= buy_share % 100
            buy['share'] = buy_share
            #恢复
            buy = buy[buy['share'] < buy['DlyVol']]
            buy = buy[-buy_amount:]
            buy['value'] = buy['share'] * buy['DlyPrc']
            self.cash -= np.sum(buy['value'].to_numpy()) * (1 + self.buy_cost)
            self.position = pd.concat([self.position, buy])
            self.position.fillna(0, inplace=True)

        self.cum_value = self.cash + np.sum(self.position['value'].to_numpy())
        self.positions.append(self.position)