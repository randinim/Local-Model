import importlib.util, sys, pickle, pprint, os

def inspect(path, train_py=None):
    if train_py and os.path.exists(train_py):
        spec = importlib.util.spec_from_file_location('train_tmp', train_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules['__main__'] = mod

    with open(path, 'rb') as f:
        obj = pickle.load(f)

    print('PATH:', path)
    print('TYPE:', type(obj))
    if isinstance(obj, dict):
        print('DICT KEYS:', list(obj.keys()))
        for k in ['models','model','scaler','feature_names','output_cols']:
            if k in obj:
                print(k, '->', type(obj[k]))
    else:
        print('HAS PREDICT:', hasattr(obj, 'predict'))
        if hasattr(obj, 'models'):
            try:
                print('models keys:', list(obj.models.keys())[:10])
            except Exception as e:
                print('models attr error:', e)


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    train = sys.argv[2] if len(sys.argv) > 2 else None
    inspect(p, train)
