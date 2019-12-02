#!/bin/bash 
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=InceptionV3_color       #Set the job name to "JobExample2"
#SBATCH --time=25:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=2                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=75G                     #Request 8GB per node 
#SBATCH --output=output_logs/inception_output_log.%j      #Send stdout/err to "Example2Out.[jobID]" 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122749846190	       #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=skumar55@tamu.edu      #Send all emails to email_address 

#User's Executable Line
module load Anaconda/3-5.0.0.1
source activate kumarenv
python final_inception_v3.py

