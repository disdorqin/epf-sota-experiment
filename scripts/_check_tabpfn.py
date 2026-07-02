"""Check TabPFN v2 API."""
import tabpfn
print(f"version: {tabpfn.__version__}")

try:
    from tabpfn import TabPFNRegressor
    print("TabPFNRegressor: OK")
except Exception as e:
    print(f"TabPFNRegressor: {e}")

# Check if it has the same API
import inspect
from tabpfn import TabPFNRegressor
sig = inspect.signature(TabPFNRegressor.__init__)
print(f"TabPFNRegressor.__init__: {sig}")
sig_fit = inspect.signature(TabPFNRegressor.fit)
print(f"TabPFNRegressor.fit: {sig_fit}")
sig_pred = inspect.signature(TabPFNRegressor.predict)
print(f"TabPFNRegressor.predict: {sig_pred}")
