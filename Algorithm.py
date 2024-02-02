
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import skfuzzy as fuzz
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.inference import DecompositionalInference
from pygad import GA
import time
import math
pd.set_option('mode.chained_assignment', None) # turns off warnings




folder = os.getcwd()
subfolder = ''
#subfolder = '\PART IV\\'
series = ['Ftsemib', 'Djow', 'Vix', 'Nasdaq', 'Sp500']
series = series[0]
df = pd.read_csv(folder + subfolder + f'\{series}_weekly.csv', sep=",")

RSI_range = [0, 100]
STO_range = [0, 100]


def train_test_split(df, split_percentage):
    index = int(split_percentage * len(df))

    return df[:index], df[index:] # train set, test set

split_percentage = 0.8
train_frame, test_frame = train_test_split(df, split_percentage)




###############################################################################################
###################################    [1] - FUZZY LOGIC    ###################################
###############################################################################################




### FUZZY VARIABLES ###
def fuzzy_variables(parameters):
    variables = {
        "RSI": FuzzyVariable(
            universe_range=(RSI_range[0], RSI_range[1]),
            terms={
                "Oversold": ('trapmf', 0, 0, parameters[2], parameters[3]),
                "Risky Trading": ('trapmf', 0, parameters[2], parameters[4], 100),
                "Overbought": ('trapmf', parameters[3], parameters[4], 100, 100)
            },
        ),
        "MACD_Crossover": FuzzyVariable(
            universe_range=(-1, 1),
            terms={
                "Positive": ('trimf', -1, -1, -0.5),
                "Stable": ('trimf', -0.5, 0, 0.5),
                "Negative": ('trimf', 0.5, 1, 1)
            },
        ),
        "STOCHk": FuzzyVariable(
            universe_range=(STO_range[0], STO_range[1]),
            terms={
                "Overbought": ('trapmf', parameters[5], parameters[6], 100, 100),
                "Mid": ('trapmf', parameters[7], parameters[8], parameters[6], parameters[9]),
                "Oversold": ('trapmf',  0, 0, parameters[8], parameters[10])
            },
        ),
        "STOCHd": FuzzyVariable(
            universe_range=(STO_range[0], STO_range[1]),
            terms={
                "Overbought": ('trapmf', parameters[11], parameters[12], 100, 100),
                "Mid": ('trapmf', parameters[13], parameters[14], parameters[12], parameters[15]),
                "Oversold": ('trapmf',  0, 0, parameters[14], parameters[16])
            },
        ),
        "STOCH_Crossover": FuzzyVariable(
            universe_range=(-1, 1),
            terms={
                "Positive": ('trimf', -1, -1, -0.5),
                "Stable": ('trimf', -0.5, 0, 0.5),
                "Negative": ('trimf', 0.5, 1, 1)
            },
        ),
        "Decision": FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "Strong Sell": ('trapmf', 0, 0, parameters[17], parameters[18]),
                "Sell": ('trapmf', parameters[17], parameters[18], parameters[19], parameters[20]),
                "Hold": ('trapmf', parameters[19], parameters[20], parameters[21], parameters[22]),
                "Buy": ('trapmf', parameters[21], parameters[22], parameters[23], parameters[24]),
                "Strong Buy": ('trapmf', parameters[23], parameters[24], 10, 10)
            },
        )
    }
    return variables
# plt.figure(figsize=(10, 2.5))
# variables["Decision"].plot()


### FUZZY RULES ###
rules = [
    ### STRONG BUY ###
    FuzzyRule(
        premise=[
            ("RSI", "Oversold"),
            ("AND", "MACD_Crossover", "Positive")
        ],
        consequence=[("Decision", "Strong Buy")],
    ),
    FuzzyRule(
        premise=[
            ("STOCHd", "Oversold"),
            ("AND", "STOCHk", "Oversold"),
            ("AND", "STOCH_Crossover", "Positive")
        ],
        consequence=[("Decision", "Strong Buy")],
    ),
    ### BUY ###
    FuzzyRule(
        premise=[
            ("RSI", "Oversold"),
            ("OR", "MACD_Crossover", "Positive"),
            ("OR", "STOCHd", "Oversold"),
            ("OR", "STOCHk", "Oversold")
        ],
        consequence=[("Decision", "Buy")],
    ),
    ### HOLD ###
    FuzzyRule(
        premise=[
            ("RSI", "Risky Trading"),
            ("OR", "MACD_Crossover", "Stable"),
            ("OR", "STOCHd", "Mid"),
            ("OR", "STOCHk", "Mid"),
            ("OR", "STOCH_Crossover", "Stable")
        ],
        consequence=[("Decision", "Hold")],
    ),
    ### SELL ###
    FuzzyRule(
        premise=[
            ("STOCHd", "Overbought"),
            ("OR", "STOCHk", "Overbought"),
            ("OR", "RSI", "Overbought"),
            ("OR", "MACD_Crossover", "Negative")
        ],
        consequence=[("Decision", "Sell")],
    ),
    ### STRONG SELL ###
    FuzzyRule(
        premise=[
            ("RSI", "Overbought"),
            ("AND", "MACD_Crossover", "Negative")
        ],
        consequence=[("Decision", "Strong Sell")],
    ),
    FuzzyRule(
        premise=[
            ("STOCHd", "Overbought"),
            ("AND", "STOCHk", "Overbought"),
            ("AND", "STOCH_Crossover", "Negative")
        ],
        consequence=[("Decision", "Strong Sell")],
    )
]
#print(rules)


### DECOMPOSITIONAL INFERENCE ###
model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator="Rc",
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog",
)


### DEFUZZIFICATION ###
def Defuzzification(df, parameters):
    ### TECHNICAL INDICATORS ###
    # RSI
    length = int(parameters[25])
    df.ta.rsi(close='Close', length=length, append=True)

    # MACD
    MACD_fast, MACD_slow, MACD_signal = int(parameters[26]), int(parameters[27]), int(parameters[28])
    df.ta.macd(close='Close', fast=MACD_fast, slow=MACD_slow, signal=MACD_signal, append=True)

    df["MACD_Crossover"] = np.select(
            condlist=[
                (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'] > 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(1) < 0) & \
                    (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-1) > 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-2) > 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-3) > 0),
                (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'] < 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(1) > 0) & \
                    (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-1) < 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-2) < 0) & (df[f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}'].shift(-3) < 0)
            ], choicelist=[1, -1], default=0
        )

    # STO
    STO_k, STO_d = int(parameters[29]), int(parameters[30])
    df.ta.stoch(high='High', low='Low', k=STO_k, d=STO_d, smooth_k=STO_d, append=True)

    df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'] = df[f'STOCHk_{STO_k}_{STO_d}_{STO_d}'] - df[f'STOCHd_{STO_k}_{STO_d}_{STO_d}']

    df["STOCH_Crossover"] = np.select(
            condlist=[
                (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'] > 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(1) < 0) & \
                    (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-1) > 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-2) > 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-3) > 0),
                
                (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'] < 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(1) > 0) & \
                    (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-1) < 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-2) < 0) & (df[f'STOCHh_{STO_k}_{STO_d}_{STO_d}'].shift(-3) < 0)
            ], choicelist=[1, -1], default=0
        )
    

    ### DEFUZZIFICATION ###

    #print(df.columns)
    rolling_window = MACD_slow + MACD_signal - 2
    df['Decision'] = np.nan
    df['Decision'].iloc[rolling_window:] = df.iloc[rolling_window:].apply(lambda row: round(model(
        variables=fuzzy_variables(parameters),
        rules=rules,
        MACD_Crossover = row['MACD_Crossover'],
        STOCHd = row[f'STOCHd_{STO_k}_{STO_d}_{STO_d}'],
        STOCHk = row[f'STOCHk_{STO_k}_{STO_d}_{STO_d}'],
        STOCH_Crossover = row['STOCH_Crossover'],
        RSI = row[f'RSI_{length}'])[0]['Decision'], 2), axis=1)
    
    return df




##############################################################################################
###################################    [3] OPTIMIZATION    ###################################
##############################################################################################




### FITNESS FUNCTION ###
def position_size_check(row, thresholds):
    global positions_counter
    global trades_counter

    direction = np.select(
        condlist=[
            (row['Decision'] <= thresholds[0]) & (positions_counter > 0),
            (row['Decision'] > thresholds[0]) & (row['Decision'] < thresholds[1]),
            row['Decision'] >= thresholds[1]
        ], choicelist=[-1, 0, 1], default=0     # sell, hold, buy
    )
    positions_counter += direction
    trades_counter += abs(direction)
    return direction, positions_counter, trades_counter


def gain(df, thresholds):
    global positions_counter
    global trades_counter
    positions_counter, trades_counter = 0, 0

    df = df.copy()
    df[['Direction', 'Positions_counter', 'Trades_counter']] = df.apply(lambda row: pd.Series(position_size_check(row, thresholds)), axis=1)

    #df['Enter_price'] = df.loc[df['Direction']!=0, 'Close']
    df['Enter_price'] = df['Close']
    df['Enter_price'] = df['Enter_price'].fillna(0)
    
    df['Long'] = np.select(
        condlist=[
            (df['Direction'] == 1)
        ], choicelist=[
            #((df["Enter_price"].shift(-1) - df["Enter_price"])/df["Enter_price"])*100               # next-day validation
            ((df["Enter_price"] - df["Enter_price"].shift(1))/df["Enter_price"].shift(1))*100       # previous-day validation
        ], default=0)
    df['Long'] = df['Long'].fillna(0)

    df['Short'] = np.select(
        condlist=[
            (df['Direction'] == -1)
        ], choicelist=[
            #((df["Enter_price"] - df["Enter_price"].shift(-1))/df["Enter_price"])*100               # next-day validation
            ((df["Enter_price"].shift(1) - df["Enter_price"])/df["Enter_price"].shift(1))*100       # previous-day validation
        ], default=0)
    df['Short'] = df['Short'].fillna(0)

    df['Equity_long'] = df['Long'].cumsum()
    df['Equity_short'] = df['Short'].cumsum()
    df['Gain'] = df['Equity_long'] + df['Equity_short']

    return df


def fitness(parameters, solution_idx):
    data_frame = Defuzzification(train_frame, parameters)

    g = gain(data_frame, parameters[:2])
    g['Date'] = pd.to_datetime(g['Date'])
    date_range = (g['Date'].max() - g['Date'].min()).days # in days
    yearly_trades = g['Trades_counter'].iloc[-1]/(date_range//365)
    #print(g[:][len(data_frame)-2:])

    ### penalty
    if yearly_trades < 10 and yearly_trades != 0: penalty = (100//yearly_trades)**2
    elif yearly_trades == 0: penalty = 1000000
    else: penalty = 0

    cumulative_gain = g['Gain'][len(data_frame)-1] if not math.isnan(g['Gain'][len(data_frame)-1]) else g['Gain'][len(data_frame)-2] # returns the second to last gain instead of the last if this last one is nan, which sometimes happened for an issue with "Short"
    return cumulative_gain - penalty


### PARAMETER RANGES ###
initial_population = [
    [      ### INITIAL SOLUTION 1 ###
    3.25, 6.75,     30, 50, 70,                     # thresholds, RSI   [0-4]
    70, 80,     10, 20,    90,      30,             # STOCHk            [5-10]
    70, 80,     10, 20,    90,      30,             # STOCHd            [11-16]
    1, 2,   3.25, 4.25,     5.75, 6.75,     8, 9,   # Decision          [17-24]
    14, 12, 26, 9, 14, 3                            # Technical Indicators: RSI length, fast MACD, slow MACD, MACD signal, fast STOCH, slow STOCH [25-30]
    ], [   ### INITIAL SOLUTION 2 ###
    4, 6,     20, 50, 80,                                # thresholds, RSI   [0-4]
    65, 75,     15, 25,    85,      35,                  # STOCHk            [5-10]
    65, 75,     15, 25,    85,      35,                  # STOCHd            [11-16]
    0.5, 1.5,   3, 4.75,     6.25, 7.25,     8.5, 9.5,   # Decision          [17-24]
    2, 5, 35, 5, 21, 5                                   # Technical Indicators: RSI length, fast MACD, slow MACD, MACD signal, fast STOCH, slow STOCH [25-30]
    ], [   ### INITIAL SOLUTION 3 ###
    4.5, 5.5,     35, 50, 65,                                 # thresholds, RSI   [0-4]
    74, 75,     20, 30,    80,      25,                       # STOCHk            [5-10]
    74, 75,     20, 30,    80,      25,                       # STOCHd            [11-16]
    0.75, 1.75,   3.74, 3.75,     5.25, 7.25,     8.4, 8.5,   # Decision          [17-24]
    5, 10, 25, 7, 5, 3                                        # Technical Indicators: RSI length, fast MACD, slow MACD, MACD signal, fast STOCH, slow STOCH [25-30]
    ], [   ### INITIAL SOLUTION 4 ###
    4.9, 5,     10, 50, 90,                                  # thresholds, RSI   [0-4]
    60, 85,     5, 15,    94,      40,                       # STOCHk            [5-10]
    60, 85,     5, 15,    94,      40,                       # STOCHd            [11-16]
    1.4, 2.4,   2.75, 4.25,     6.24, 6.25,     7.5, 9.25,   # Decision          [17-24]
    7, 7, 40, 5, 21, 7                                       # Technical Indicators: RSI length, fast MACD, slow MACD, MACD signal, fast STOCH, slow STOCH [25-30]
    ]]

gene_space = [
    # thresholds
    {'low': 2, 'high': 5}, {'low': 5, 'high': 8},
    # RSI
    {'low': 10, 'high': 40}, {'low': 40, 'high': 60}, {'low': 60, 'high': 91},
    # STOCHk
    {'low': 60, 'high': 75}, {'low': 75, 'high': 86},
    {'low': 5, 'high': 21}, {'low': 15, 'high': 31}, {'low': 80, 'high': 95},
    {'low': 25, 'high': 41},
    # STOCHd
    {'low': 60, 'high': 75}, {'low': 75, 'high': 86},
    {'low': 5, 'high': 21}, {'low': 15, 'high': 31}, {'low': 80, 'high': 95},
    {'low': 25, 'high': 41},
    # Decision
    {'low': 0.5, 'high': 1.5}, {'low': 1.5, 'high': 2.5},
    {'low': 2.75, 'high': 3.76}, {'low': 3.75, 'high': 4.76},
    {'low': 5.25, 'high': 6.26}, {'low': 6.25, 'high': 7.26},
    {'low': 7.5, 'high': 8.6}, {'low': 8.5, 'high': 9.6},
    # Technical Indicators
    {'low': 2, 'high': 30},
    {'low': 5, 'high': 20}, {'low': 20, 'high': 50}, {'low': 5, 'high': 20},
    {'low': 5, 'high': 25}, {'low': 3, 'high': 9}
]        

num_genes = len(initial_population[0])
num_parents_mating = 2
num_generations = 10
sol_per_pop = 1
gene_type = [float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float,
             int, int, int, int, int, int]


start_time = time.time()
print('Start train')
ga = GA(num_generations=num_generations,
        initial_population=initial_population,
        num_parents_mating=num_parents_mating,
        gene_space=gene_space,
        fitness_func=fitness,
        gene_type=gene_type,
        num_genes=num_genes)
ga.run()
end_time = time.time()
print(f'End train in {end_time-start_time} seconds')
solution, solution_fitness, solution_idx = ga.best_solution()
print(f'Best solution found:\n{solution}')


train = Defuzzification(train_frame, solution)
test = Defuzzification(test_frame, solution)




###############################################################################################
###################################    [4] OUTPUT & PLOTS   ###################################
###############################################################################################




final_frame = gain(pd.concat([train, test]), solution[:2])
#final_frame = gain(test, solution[:2])

final_frame['Decision'] = np.select(
        condlist=[
            final_frame["Direction"] == 1,
            final_frame["Direction"] == 0,
            final_frame["Direction"] == -1,
        ], choicelist=["Buy", "Hold", "Sell"], default="HOLD"   # the caps are just to distinguish when it was set by default or by design
    )

final_decision = final_frame['Decision'][len(train)+len(test)-1]
print(f'The decision for tomorrow is to: {final_decision}')


ga.plot_fitness()


# print(final_frame.iloc[-1:,-1:]) # see the final total gain
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
test_gain = int(final_frame['Gain'][len(train)+len(test)-1] - final_frame['Gain'][len(train)])
result = 'gain' if test_gain >= 0 else 'loss'
ax.set_title(f'{series}: {result} of {abs(test_gain)}%',fontweight ='bold',fontsize=20)
ax.set_ylabel('Gain', fontweight = 'bold', fontsize=20)
ax.set_xlabel('Time', fontweight = 'bold', fontsize=20)
time_col = pd.to_datetime(final_frame['Date'])
norm_factor = final_frame['Gain'].max()/final_frame['Close'].max()

plt.plot(time_col, final_frame['Gain'], alpha=0.4, color='red',linewidth=5, label = 'Equity')
plt.plot(time_col, final_frame['Close']*norm_factor, alpha=0.2, color='black',linewidth=5, label = "Close")
plt.plot(time_col, final_frame['Gain'].cummax(), alpha=0.2, color='red',linewidth=3, label = 'Maximum Equity Reached')
plt.plot(time_col, final_frame['Equity_long'], alpha=0.4, color='green',linewidth=3, label = 'Long Equity')
plt.plot(time_col, final_frame['Equity_short'], alpha=0.2, color='green',linewidth=3, label = 'Short Equity')
plt.axvline(x=pd.to_datetime(train['Date'])[len(train)-1], color='red', linestyle='--', label='Test-Train Split')
ax.legend()


results_folder = 'PreviousDay_Results'
try:
    plt.savefig(f'{results_folder}/Algotrading_on_{series}.png', dpi=150)
    # plt.close(fig)
    final_frame.to_csv(f'{results_folder}/Algotrading_on_{series}.csv')
except:
    os.mkdir(f'{results_folder}')
    plt.savefig(f'{results_folder}/Algotrading_on_{series}.png', dpi=150)
    # plt.close(fig)
    final_frame.to_csv(f'{results_folder}/Algotrading_on_{series}.csv')




# %%
