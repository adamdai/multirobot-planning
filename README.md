# multirobot-planning

Multi-robot Reachability-based Trajectory Planning

## Setup
For Windows, conda environment may not work on other OSes.

Clone the GitHub repository:

    git clone https://github.com/adamdai/multirobot-planning.git

Create conda environment:

    conda create -n multirtd python=3.9

Active the environment:
   
    conda activate multirtd
    
Install dependencies:

    pip install numpy scipy matplotlib
   
Install `multirtd` locally from directory containing `setup.py`
   
    pip install -e .
