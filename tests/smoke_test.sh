#! /bin/sh

example_dirs="fft_example"

testenv="smttask"

# See https://stackoverflow.com/a/5947802
ORANGE='\033[0;33m'
NOCOLOR='\033[0m'

echo ""
echo -e "The ${ORANGE}smoke test for smttask${NOCOLOR} runs through all packaged examples, thus ensuring that they execute without errors, but without checking that they executed correctly."
echo "This requires that within each example directory, 'git init' and 'smttask project init' were run, and that there be no uncommitted changes."
echo "This also assumes that a test environment '${testenv}' with installed with conda."
echo ""
echo "Within each example directory, this test will do the following:"
echo "  - Delete the 'data' directory."
echo "  - Execute 'python run.py' twice, to test both with and without existing data from previous runs. "
echo ""
echo "This list of example directories is hard-coded. Its values are:"
echo "  $example_dirs"
echo ""

## Preliminaries

# Change to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Switching to directory $DIR"
echo ""
cd "$DIR"

# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

## Ensure the smttask test environment exists
# Get the list of current conda environments
envlist="$(conda env list | tr "*" " ")"

testenvexists=n
for env in $envlist; do
  if [ "$env" == "$testenv" ]; then
    testenvexists=y
    break
  fi
done

if [ $testenvexists == "n" ]; then
  echo "The test environment '${testenv}' is not known to conda."
  exit
fi

## Activate environment, and loop over examples

conda activate $testenv

EXAMPLES_DIR="$DIR/../examples"

num_errors=0
for example in $example_dirs; do
  cd "$EXAMPLES_DIR/$example"
  if [ -d data ]; then
    rm -r data
  fi
  echo -e "${ORANGE}Running example $example${NOCOLOR} with an empty cache..."
  python run.py
  if [ $? -eq 0 ]; then
    echo "Done"
  else
    let "num_errors+=1"
  fi
  echo ""  # Add a blank line

  echo -e "${ORANGE}Re-running example $example${NOCOLOR}. Cache should be full..."
  python run.py
  if [ $? -eq 0 ]; then
    echo "Done"
  else
    let "num_errors+=1"
  fi
  echo ""
done

if [ $num_errors -eq 0 ]; then
  echo "All examples executed successfully."
elif [ $num_errors -eq 1 ]; then
  echo "1 example raised an exception"
else
  echo "$num_errors examples raised an exception."
fi
