import argparse
import numpy as np
import pandas as pd

def us(args):
    df = pd.read_feather("us_con.feather")
    df = df.loc[:, ['PERMNO', 'SICCD']]
    df.drop_duplicates(inplace=True)
    stock = df['PERMNO'].unique()
    industry = df['SICCD'].unique()
    stock.sort()
    stock_dict = {stock[i]: i for i in range(len(stock))}
    ind_dict = {industry[i]: i for i in range(len(industry))}

    
    if args.graph == "hypergraph":
        hypergraph = np.zeros((len(stock), len(industry)))
        for i in range(len(df)):
            data = df.iloc[i]
            row = stock_dict[data['PERMNO']]
            col = ind_dict[data['SICCD']]
            hypergraph[row][col] = 1
        np.save("us_hypergraph.npy", hypergraph)
    elif args.graph == "graph":
        graph = np.zeros((len(stock), len(stock)))
        gb = df.groupby("SICCD")
        for name, group in gb:
            codes = group['PERMNO'].values
            for i in range(len(codes)):
                node1 = codes[i]
                index1 = stock_dict[node1]
                for j in range(i+1, len(codes)):
                    node2 = codes[j]
                    index2 = stock_dict[node2]
                    graph[index1][index2] = 1
                    graph[index2][index1] = 1
        np.save("us_graph.npy", graph)
    # else:
        
    
def cn(args):
    return 


def main():
    parser = argparse.ArgumentParser(description='Constructing graph for the market')

    parser.add_argument('--market', type=str, required=True, default="us", help='Stock market, options: [us, cn]')
    parser.add_argument('--graph', type=str, required=True, default='graph', help='Type of graph, options: [graph, hypergraph]')

    args = parser.parse_args()

    if args.market == "us":
        us(args)
    elif args.market == "cn":
        cn(args)


if __name__ == "__main__":
    main()
    


