description: Use this file to know the commands of the job being used for reserving gpus

A brief overview of the above file named “test.job”. The first line is important as this file should be run as a bash script. The next few lines are important slurm parameters that configure the resources allocated to the job.

--job-name is the name of the job.
--output is the stdout for all the scripts that run via slurm. Since we do not have an interactive shell with slurm, all your print/cout/echo outputs go to this file.
--error is the output file for all the errors.
--time sets the maximum time that your job should take before being killed (format is dd-hh:mm, so 2-00:00 means 2 days). Set this to a reasonable limit so that in case of logical errors you get to know that your job did not terminate and the cluster is free for other users.
--mem sets the main memory required for your code in MB (so 50000 in this example is 50 GB main memory). Set it to be more than the anticipated usage for safety. However please do not set it to be arbitrarily large like 300 GB as it will severely impact everyone else in the queue and result in longer wait times.
--gres specifies the generic resources (for us GPUs). In this example I request 2 GPUs. You can request upto 8 GPUs (the maximum available in the servers ai03 or ai05). Please be mindful of other users and request only what you need. Also note that instead of running code that uses 8 GPUs and utilises 4 GB memory on each of them, instead use 1 GPU and use all 32GB present in that single GPU.
--nodelist can be used to specify the specific node in case you have preference for a specific type of GPU (Tesla nodes have 32 GB Tesla V100 GPUs and Quadro nodes have 48 GB Quadro RTX GPUs). However, if you do not care what GPU you need and are sure that your job will fit in the smaller GPU, it is best to remove this line from your job file and let Slurm allocate you the node that is available. This will result in a shorter wait time.
In case you need to use the Nvidia CUDA Compiler (nvcc), I include this because the path for nvcc is not the same as other AI servers.
You can run command line tools in a job file (like nvidia-smi) and the outputs will be shown in the --output file.
The cluster uses the same network based home directory that you use in all of our servers so you can access all of your data, code and conda environments. In this example I simply activate an environment called py_env (which has pytorch installed) and run a .py file called run.py.


Once you submit a job, the job will get a jobID and go to the slurm queue. Take note of the jobID. You can see the status of your job in the queue by typing “squeue” on the command line.


Submit a job using the following:

sbatch test.job


You can get your job history by:

sacct -u <netid>


You can get the job details using:

scontrol -d show job <jobID>

