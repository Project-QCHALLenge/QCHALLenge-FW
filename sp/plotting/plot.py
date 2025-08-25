# Plotting routinen zum Vergleich mehrere Usecases

import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns


def plot_comparison(eval_data):

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(eval_data['solver'].unique()))

    
    eval_data = eval_data.reset_index().rename(columns={'index': 'id'})
    # Calculate the best found objective for each instance (ID) where violations are 0
    best_objective_zero_violations = eval_data[eval_data['violations'].apply(len)  == 0].groupby('usecase')['objective'].min().reset_index()
    best_objective_zero_violations.columns = ['usecase', 'best_objective']
    # Merge the best objectives back to the original DataFrame
    eval_data = eval_data.merge(best_objective_zero_violations, on='usecase', how='left')    
       
    
    for i, solver in enumerate(eval_data['solver'].unique()):
        subset = eval_data[eval_data['solver'] == solver]
        
        zero_violations = subset[subset['violations'].apply(len) == 0].copy()
        zero_violations.loc[:, 'adjusted_objective'] = zero_violations['best_objective'] - zero_violations['objective']
        sns.lineplot(data=zero_violations, x='usecase', y='adjusted_objective', marker='o', label=f'{solver} Objective (0 violations)', color=palette[i])

        non_zero_violations = subset[subset['violations'].apply(len) != 0].copy()
        non_zero_violations.loc[:, 'adjusted_objective'] = subset['violations']
        plt.scatter(len(non_zero_violations['usecase']), len(non_zero_violations['usecase']), marker='x', color=palette[i], label=f'{solver} Objective (violations)')

    plt.title('Objective over usecases')
    plt.xlabel('Usecase')
    plt.ylabel('Difference from best found objective (zero is optimal)')
    plt.xticks(rotation=30)
    plt.legend(title='Solver')
    plt.show()

 
    
    
