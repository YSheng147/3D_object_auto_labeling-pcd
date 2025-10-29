set -e

git config --global --add safe.directory .

pip install --no-build-isolation --config-settings editable_mode=compat -e .

cd tracker && pip install -e . --user

cd ..