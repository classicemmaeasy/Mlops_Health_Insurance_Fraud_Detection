### create virtual env
py -3.11 -m venv .your_env
.\.your_env\Scripts\Activate.ps1

### create project structure
```
mkdir -p \
 data/raw data/processed data/external \
 notebooks \
 src/{data,features,models,utils} \
 app \
 configs \
 scripts \
 tests \
 .github/workflows \
 docker \
 great_expectations \
 mlruns \
 artifacts
```
