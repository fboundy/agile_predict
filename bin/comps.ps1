# run_script.ps1

# Name of your conda environment
$envName = "agile_predict"

# Path to your Python script
$scriptPath = "U:\django\agile_predict\.local\its_a_competition.py"

# Initialize conda for PowerShell (important)
& conda shell.powershell hook | Out-String | Invoke-Expression

# Activate the environment
conda activate $envName

# Run the script
conda run -n $envName python $scriptPath