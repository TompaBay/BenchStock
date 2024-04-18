import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def sharpe_ratio(returns, value, rf = 0.01):
    sr_month = (value - 1 - rf) / (np.std(returns) * np.sqrt(len(returns)))
    return sr_month


def r2(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    dif = real - pred
    return 1 - np.sum(dif * dif) / np.sum(real * real)


def adj_convert(o):
    n = o.shape[0]
    o_sum = np.sum(o, axis=0)
    indice = np.where(o_sum > 1)[0]
    adj = np.zeros((n, n))

    for ind in indice:
        edge = o[:, ind].reshape(-1)
        indice = np.where(edge == 1)[0]
        indices = np.transpose([np.tile(indice, len(indice)), np.repeat(indice, len(indice))])
        for i in indices:
            adj[i[0]][i[1]] = 1
    
    return adj

def max_drawdown(values):
    peak = values[0]  # Assume the first value is the initial peak
    max_drawdown = 0

    for value in values:
        if value > peak:
            peak = value
        else:
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return -max_drawdown / peak
