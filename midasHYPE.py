import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from arch.univariate import ZeroMean, ConstantMean, ARX, HARX, Normal, StudentsT, SkewStudent
from arch.univariate.volatility import MIDASHyperbolic

def prepare_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to regular intervals if necessary
    df = df.resample('1T').last().ffill()
    
    
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    return returns

def fit_midas_models(returns, m=4):
    models = {
        'Zero Mean': ZeroMean(returns, volatility=MIDASHyperbolic(m=m)),
        'Constant Mean': ConstantMean(returns, volatility=MIDASHyperbolic(m=m)),
        'ARX': ARX(returns, lags=1, volatility=MIDASHyperbolic(m=m)),
        'HARX': HARX(returns, lags=[1, 5, 22], volatility=MIDASHyperbolic(m=m))
    }
    
    distributions = {
        'Normal': Normal(),
        'Student-t': StudentsT(),
        'Skewed Student-t': SkewStudent()
    }
    
    results = {}
    
    for mean_name, model in models.items():
        for dist_name, dist in distributions.items():
            model.distribution = dist
            try:
                result = model.fit(disp='off', options={'maxiter': 1000})
                results[f"{mean_name} - {dist_name}"] = result
                print(f"Fitted: {mean_name} with {dist_name} distribution")
                print(result.summary())
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"Error fitting {mean_name} with {dist_name} distribution: {str(e)}")
    
    return results

def plot_conditional_volatility(results, save_dir):
    plt.figure(figsize=(14, 8))
    
    for name, result in results.items():
        cond_vol = result.conditional_volatility
        plt.plot(cond_vol, label=name)
    
    plt.title('Conditional Volatility Over Time')
    plt.xlabel('Time')
    plt.ylabel('Conditional Volatility')
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(save_dir, 'conditional_volatility.png'))
    plt.close()

def plot_aic_bic_comparison(results, save_dir):
    comparison = pd.DataFrame({
        'AIC': {name: result.aic for name, result in results.items()},
        'BIC': {name: result.bic for name, result in results.items()}
    })

    comparison.sort_values('AIC').plot(kind='bar', figsize=(14, 8))
    plt.title('Model Comparison (AIC and BIC)')
    plt.ylabel('Information Criterion Value')
    
    plt.savefig(os.path.join(save_dir, 'aic_bic_comparison.png'))
    plt.close()



if __name__ == "__main__":
    file_path = 'hf_data.csv' 
    save_dir = 'midas_results'
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    returns = prepare_data(file_path)
    results = fit_midas_models(returns)
    
    # Plotting results
    plot_conditional_volatility(results, save_dir)
    plot_aic_bic_comparison(results, save_dir)


    
