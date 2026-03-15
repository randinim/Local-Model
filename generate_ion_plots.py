import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

sns.set(style='whitegrid')
ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, 'v2', 'data', 'full_dataset.csv')
MODELS = ['v2','v3','v4']
ION_TARGETS = [
    'Bittern_Mg_Concentration_gL',
    'Bittern_K_Concentration_gL',
    'Bittern_SO4_Concentration_gL',
    'Bittern_Ca_Concentration_gL'
]
OUT = os.path.join(ROOT, 'comparison_plots')
os.makedirs(OUT, exist_ok=True)


def _fe(df: pd.DataFrame):
    f = pd.DataFrame()
    f['production_volume'] = df['production_volume']
    f['rain_sum'] = df.get('rain_sum', pd.Series(0, index=df.index))
    f['temperature_mean'] = df.get('temperature_mean', pd.Series(0, index=df.index))
    f['humidity_mean'] = df.get('humidity_mean', pd.Series(0, index=df.index))
    f['wind_speed_mean'] = df.get('wind_speed_mean', pd.Series(0, index=df.index))
    if 'Month' in df.columns:
        f['month_sin'] = np.sin(2*np.pi*df['Month']/12)
        f['month_cos'] = np.cos(2*np.pi*df['Month']/12)
    return f.fillna(0)


def load_payload(path, model_dir=None):
    if model_dir:
        train_py = os.path.join(model_dir,'train.py')
        if os.path.exists(train_py):
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location('train_tmp', train_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['__main__'] = mod
    with open(path,'rb') as f:
        return pickle.load(f)


def predict_payload(payload, df):
    X = _fe(df)
    scaler = payload.get('scaler')
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values
    if 'model' in payload and hasattr(payload['model'],'predict'):
        preds = payload['model'].predict(Xs)
        cols = payload.get('output_cols')
        return pd.DataFrame(preds, columns=cols, index=df.index)
    if 'models' in payload and isinstance(payload['models'], dict):
        models = payload['models']
        # if models are ensemble members returning full outputs
        member_preds = []
        for name,m in models.items():
            try:
                out = m.predict(Xs)
            except Exception:
                out = m.predict(df)
            arr = np.asarray(out)
            if arr.ndim==1:
                arr = arr.reshape(-1,1)
            # if arr has same number of outputs as payload output_cols, use them
            if payload.get('output_cols') and arr.shape[1]==len(payload.get('output_cols')):
                member_preds.append(pd.DataFrame(arr, columns=payload.get('output_cols'), index=df.index))
            elif arr.shape[1]==1:
                # single-col per model -> assume corresponds to model name
                member_preds.append(pd.DataFrame(arr, columns=[name], index=df.index))
            else:
                member_preds.append(pd.DataFrame(arr, index=df.index))
        # If each member produced full output, average them by equal weight
        if member_preds and list(member_preds[0].shape)[1]==len(payload.get('output_cols',[])):
            stacked = np.mean([m.values for m in member_preds], axis=0)
            return pd.DataFrame(stacked, columns=payload.get('output_cols'), index=df.index)
        # If members are single-col per-target, concat sideways
        concat = pd.concat(member_preds, axis=1)
        # try to align column names to payload output_cols
        if set(ION_TARGETS).issubset(set(concat.columns)):
            return concat
        # fallback: return concat
        return concat
    if hasattr(payload,'predict'):
        return payload.predict(df)
    raise RuntimeError('Unknown model payload format')


def binned_calibration(y_true,y_pred,bins=10):
    df = pd.DataFrame({'y':y_true,'p':y_pred})
    df['bin'] = pd.qcut(df['p'], q=bins, duplicates='drop')
    grp = df.groupby('bin').agg({'y':'mean','p':'mean','y':'count'}).reset_index()
    return grp


def main():
    df = pd.read_csv(DATA, comment='#')
    preds_all = {}
    for m in MODELS:
        pth = os.path.join(ROOT,m,f'waste_predictor_{m}.pkl')
        if not os.path.exists(pth):
            print('Model missing',pth); continue
        payload = load_payload(pth, os.path.join(ROOT,m))
        preds = predict_payload(payload, df)
        preds_all[m]=preds

    # For each ion target, create plots
    for ion in ION_TARGETS:
        plt.figure(figsize=(8,6))
        for m in preds_all.keys():
            if ion in preds_all[m].columns:
                plt.scatter(df[ion], preds_all[m][ion], alpha=0.5, s=10, label=m)
        mn = df[ion].min(); mx=df[ion].max()
        plt.plot([mn,mx],[mn,mx],'k--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Parity: {ion}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,f'parity_{ion}.png'), dpi=150)
        plt.close()

        # Residual violin
        plt.figure(figsize=(8,4))
        resid_df = []
        for m in preds_all.keys():
            if ion in preds_all[m].columns:
                resid = df[ion] - preds_all[m][ion]
                resid_df.append(pd.DataFrame({'resid':resid, 'model':m}))
        if resid_df:
            rdf = pd.concat(resid_df, axis=0)
            sns.violinplot(x='model', y='resid', data=rdf)
            plt.title(f'Residual distribution: {ion}')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT,f'residuals_{ion}.png'), dpi=150)
            plt.close()

        # Error vs production volume
        plt.figure(figsize=(8,6))
        for m in preds_all.keys():
            if ion in preds_all[m].columns:
                err = np.abs(df[ion]-preds_all[m][ion])
                plt.scatter(df['production_volume'], err, alpha=0.4, s=10, label=m)
        plt.xscale('linear')
        plt.yscale('linear')
        plt.xlabel('Production volume')
        plt.ylabel('Absolute error')
        plt.title(f'Absolute error vs Production: {ion}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,f'err_vs_prod_{ion}.png'), dpi=150)
        plt.close()

        # Calibration plot (binned predicted vs true)
        plt.figure(figsize=(6,6))
        for m in preds_all.keys():
            if ion in preds_all[m].columns:
                grp = binned_calibration(df[ion].values, preds_all[m][ion].values, bins=10)
                plt.plot(grp['p'], grp['y'], marker='o', label=m)
        mn = df[ion].min(); mx=df[ion].max()
        plt.plot([mn,mx],[mn,mx],'k--')
        plt.xlabel('Mean predicted (binned)')
        plt.ylabel('Mean true')
        plt.title(f'Calibration (binned) for {ion}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,f'calibration_{ion}.png'), dpi=150)
        plt.close()

    print('Ion diagnostic plots saved to',OUT)

if __name__=='__main__':
    main()
