
# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

conda activate "smttask"

# Add a pyvenv.cfg file prohibiting the use of user site packages.
# (This is the default for normal Python venvs, but not Conda environments)
# The pyvenv.cfg file must be placed one directory above the Python executable
# (see https://docs.python.org/3/library/site.html)
ENVDIR="$( cd "$( dirname "$( dirname "$(which python)" )" )" && pwd )"
echo "include-system-site-packages = false" >> $ENVDIR/pyvenv.cfg
    # Unlikely that the pyvenv.cfg file exists, but append (instead of write) in case it does
