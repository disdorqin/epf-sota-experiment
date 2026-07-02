"""Check TiRex model API."""
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import torch
from tirex import TiRexZero, load_model, ForecastModel

print("TiRexZero:", TiRexZero)
print("ForecastModel:", ForecastModel)

# Check TiRexZero signature
import inspect
try:
    sig = inspect.signature(TiRexZero.__init__)
    print(f"TiRexZero.__init__: {sig}")
except:
    pass

# Try to initialize and load
print("\nLoading TiRex (zero-shot)...")
try:
    model = TiRexZero()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Model attributes: {[x for x in dir(model) if not x.startswith('_')][:20]}")
    
    # Check predict method
    if hasattr(model, 'predict'):
        pred_sig = inspect.signature(model.predict)
        print(f"predict signature: {pred_sig}")
    if hasattr(model, 'forecast'):
        pred_sig = inspect.signature(model.forecast)
        print(f"forecast signature: {pred_sig}")
except Exception as e:
    print(f"TiRexZero init failed: {e}")
    import traceback
    traceback.print_exc()
