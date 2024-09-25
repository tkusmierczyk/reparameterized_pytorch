import re, sys, os
import subprocess
from datetime import datetime


def usage():
    print(f"Usage: {sys.argv[0]} <script_with_python_commands.sh> [path_to_env]")
    sys.exit(1)


PARITIONS = ["rtx2080", "rtx3080", "rtx4090"]


def main():
    print(
        "This script generates sbatch files for each Python command found in a specified input script."
    )
    print("The sbatch files will be stored in a directory named 'sbatch_files'.")
    print("An additional script will be created to submit all generated sbatch files.")
    print(f"Usage: {sys.argv[0]} <script_with_python_commands.sh> [path_to_env]")
    print(
        "If the path to the .env environment is not provided, the script will use '.env' in the current directory."
    )
    print()

    if len(sys.argv) < 2:
        usage()

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)

    if len(sys.argv) < 3:
        env_path = os.path.join(os.getcwd(), ".env")
        print(f"No ENV_PATH provided. Using default: {env_path}")
    else:
        env_path = sys.argv[2]
        print(f"Using provided ENV_PATH: {env_path}")

    sbatch_dir = f"slurm_sbatch_files_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(sbatch_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(script_dir, "slurm_template.sbatch"), "r") as template_file:
        sbatch_template = template_file.read()

    counter = 1

    with open(input_file, "r") as f:
        for line in f:
            if line.strip().startswith("python"):

                script = re.sub(r"\s*&\s*$", "", line.strip())
                job_name = f"train_{counter}"
                partition = PARITIONS[counter % len(PARITIONS)]

                sbatch_content = sbatch_template.replace("<JOB_NAME>", job_name)
                sbatch_content = sbatch_content.replace("<SBATCH_DIR>", sbatch_dir)
                sbatch_content = sbatch_content.replace("<ENV_PATH>", env_path)
                sbatch_content = sbatch_content.replace("<COMMAND>", script)
                sbatch_content = sbatch_content.replace("<PARTITION>", partition)

                sbatch_file = os.path.join(sbatch_dir, f"{job_name}.sbatch")
                with open(sbatch_file, "w") as sf:
                    sf.write(sbatch_content)

                print(f"Created {sbatch_file}")

                counter += 1

    execution_script = "slurm_execute_all_sbatch.sh"
    with open(execution_script, "w") as es:
        es.write("#!/bin/bash\n")
        for sbatch_file in os.listdir(sbatch_dir):
            if sbatch_file.endswith(".sbatch"):
                es.write(f"sbatch {os.path.join(sbatch_dir, sbatch_file)}\n")

    os.chmod(execution_script, 0o755)

    print(f"Generated sbatch files in {sbatch_dir}")
    print(f"Execute all sbatch files with: bash {execution_script}")


if __name__ == "__main__":
    main()
