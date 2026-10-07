# Getting Started on TPU VMs

This guide will walk you through the steps to get started with Levanter on TPU VMs.

## Overview

An important thing to know about TPU VMs is that they are not a single machine (for more than a vX-8). Instead, they
are a collection of workers that are all connected to the same TPU pod. Each worker manages a set of 8 TPUs.
This means that you can't just run a single process on a TPU VM instance, you need to run a distributed process,
and you can't just set up one machine, but a whole cluster. The script `infra/spin-up-vm.sh` creates a TPU VM
instance and installs Levanter on every worker.

## Installation

### Install Levanter

First, you need to clone the Levanter repository and install the dependencies. You can do this with the following commands:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### Google Cloud setup

First you need gcloud installed and configured. You can find instructions for that [here](https://cloud.google.com/sdk/docs/quickstarts)
or if you're a conda person you can just run `conda install -c conda-forge google-cloud-sdk`.

Second, you need to follow some steps to enable Cloud TPU VM. You can follow Google's directions [here](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
but the gist of it is you need to enable the TPU API and the Compute Engine API. You can do this by running:

```bash
gcloud auth login  # if you haven't already
gcloud auth application-default login  # on your local machine
gcloud components install alpha
gcloud services enable tpu.googleapis.com
gcloud config set account your-email-account
gcloud config set project your-project
```

You can follow more steps there to get used to things like creating instances and such, but we'll only discuss the
most important details here.

Google recommends not running those first two commands on a VM, and instead using tokens and service accounts. You can
find more information about that [here](https://cloud.google.com/docs/authentication/production#auth-cloud-implicit-python).
Honestly, if you're working outside of a corp environment and not dealing with private data, I don't bother...

You may also need to create an SSH key and add it to your Google Cloud account. Consider using
[gcloud's guide on ssh keys](https://cloud.google.com/compute/docs/connect/add-ssh-keys#metadata) (or OS Login if you do that)
to set up ssh keys and [using `ssh-agent`](https://kb.iu.edu/d/aeww) to make executing the SSH commands easier.

## Common Issues

### Can't find TPUs

If you get the warning `No GPU/TPU found, falling back to CPU.` then something else might be using the TPU, like a zombie python
process. You can kill all python processes with `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'pkill python'`

If that fails, you can try rebooting the TPU VM instance. This is harder than it sounds, because you can't just reboot the instance
from the console or with a special command. You have to do it with ssh:

```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'sudo reboot'
```

**and then you have to ctrl-c this after about 10 seconds**. Otherwise, gcloud will think the command failed and will
try again, and get stuck in a loop forever. (You can ctrl-c it at any point after 10 seconds.)

## Creating a TPU VM Instance



### Automatic Setup

You can use `infra/spin-up-vm.sh` to create a TPU VM instance. In addition to creating the instance, it will set up
the venv on each worker, and it will clone the repo to `~/levanter/`.

```bash
bash infra/spin-up-vm.sh <name> -z <zone> -t <type> -n <subnetwork> [--preemptible] [--use-alpha]
```

Defaults are:
- `zone`: `us-east1-d`
- `type`: `v3-32`
- `subnetwork`: `default` (set to custom VPC subnet, useful for tpuv4 configs)
- `preemptible`: `false`
- `use-alpha`: `false` (mark `true` for tpuv4s in alpha zones like `us-central2`)

**Notes**:

* This uploads setup scripts via scp. If the ssh-key that you used for Google Cloud requires passphrase or your ssh key
path is not `~/.ssh/google_compute_engine`, you will need to modify the script.
* The command will spam you with a lot of output, sorry.


## Useful commands

### SSHing into one TPU VM worker

`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=0`

### Running a command on all workers (in parallel)
`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=all --command="echo hello"`

### SCPing a file to all workers
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=all`

### SCPing a file to one worker
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=0`

### SCPing a file from one worker
`gcloud compute tpus tpu-vm scp $name:path/to/file my_file --zone us-east1-d --worker=0`
