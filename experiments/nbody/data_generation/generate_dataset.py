from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
import subprocess
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--name', type=str, default='new',
                    help='Add string to suffix of filename.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--dim', type=int, default=3,
                    help='Spatial simulation dimension (2 or 3).')
parser.add_argument('--boxsize', type=float, default=5.0,
                    help='Size of a surrounding box. If 0, then no box.')

args = parser.parse_args()
args_dict = vars(args)
git_commit = subprocess.check_output(["git", "describe", "--always"]).strip()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0,
                    n_balls=args.n_balls,
                    box_size=args.boxsize,
                    dim=args.dim)
    suffix = '_springs_' + str(args.dim) + 'D_'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0,
                              n_balls=args.n_balls,
                              box_size=args.boxsize,
                              dim=args.dim)
    suffix = '_charged_' + str(args.dim) + 'D_'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
suffix += '_' + str(args.name)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    ds = {
        "points": list(),
        "vel": list(),
        "edges": list(),
        "clamp": list(),
        "E": list(),
        "U": list(),
        "K": list(),
        "delta_T": sim._delta_T,
        "sample_freq": sample_freq,
    }
    if args.simulation == 'charged':
        ds["charges"] = list()

    for i in range(num_sims):
        t = time.time()
        if args.simulation == 'springs':
            loc, vel, edges, clamp = sim.sample_trajectory(
                T=length, sample_freq=sample_freq)
        elif args.simulation == 'charged':
            loc, vel, edges, charges, clamp = sim.sample_trajectory(
                T=length, sample_freq=sample_freq)

        energies = np.array(
            [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
             range(loc.shape[0])])

        ds["E"].append(energies[..., 0])
        ds["U"].append(energies[..., 1])
        ds["K"].append(energies[..., 2])

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        ds["points"].append(loc)
        ds["vel"].append(vel)
        ds["edges"].append(edges)
        if args.simulation == 'charged':
            ds["charges"].append(charges)
        ds["clamp"].append(clamp)

    for key in ["points", "vel", "edges", "E", "U", "K"]:
        ds[key] = np.stack(ds[key])
    for key in ["E", "U", "K"]:
        ds[key] = np.mean(ds[key], axis=0)
    if args.simulation == 'charged':
        ds["charges"] = np.stack(ds["charges"])
    ds["clamp"] = np.stack(ds["clamp"])

    return ds

# Generate training and test dataset.
ds = dict()
print("Generating {} training simulations".format(args.num_train))

ds["train"] = generate_dataset(args.num_train,
                               args.length,
                               args.sample_freq)
ds["train"]["git_commit"] = str(git_commit)
ds["train"]["args"] = args_dict

print("Generating {} test simulations".format(args.num_test))
ds["test"] = generate_dataset(args.num_test,
                              args.length_test,
                              args.sample_freq)
ds["test"]["git_commit"] = str(git_commit)
ds["test"]["args"] = args_dict

# Save dataset to file.
for ds_type in ["train", "test"]:
    filename = "ds_" + ds_type + suffix + ".pkl"
    with open(filename, "wb") as file:
        pickle.dump(ds[ds_type], file)

