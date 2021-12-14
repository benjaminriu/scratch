!/bin/bash

# BEGIN SBATCH directives
SBATCH --job-name=test_seq
SBATCH --output=res_seq.txt

SBATCH --ntasks=1
SBATCH --time=00:10:00
SBATCH --partition=cpu_seq
SBATCH --account=sandbox
SBATCH --mail-type=ALL
SBATCH --mail-user=benjamin.riu@polytechnique.edu
# END SBATCH directives

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/10.2.0

# Execution
./hello.seq