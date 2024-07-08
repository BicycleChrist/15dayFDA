import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate import ZeroMean, ConstantMean, ARX, HARX, Normal, StudentsT, SkewStudent
from arch.univariate.volatility import MIDASHyperbolic

def prepare_data(file_path):
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to regular intervals if necessary
    df = df.resample('1T').last().ffill()
    
    # Calculate log returns
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


if __name__ == "__main__":
    
    file_path = 'hf_data.csv'  # Replace with your actual file path
    returns = prepare_data(file_path)
    results = fit_midas_models(returns)
    
    # AIC vs BIC
    comparison = pd.DataFrame({
        'AIC': {name: result.aic for name, result in results.items()},
        'BIC': {name: result.bic for name, result in results.items()}
    })
    print("MIDAS Model Comparison per mean model/distribution (sorted by AIC):")
    print(comparison.sort_values('AIC'))

    
