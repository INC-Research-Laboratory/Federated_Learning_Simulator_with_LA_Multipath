import Simulator
import argparse

parser = argparse.ArgumentParser(description="Simulation options")
parser.add_argument("--use_multipaths", action="store_true", help="Whether to use Multipaths")

args = parser.parse_args()
use_multipaths = args.use_multipaths

multipaths = True
singlepath = False

server, edge_device = Simulator.Device()
usingClient_list = Simulator.target_list(edge_device)

if use_multipaths:
    # multi path
    Simulator.simulator_multipaths(server, edge_device,usingClient_list,multipaths)

else:
    # single path
    Simulator.simulator_single(server, edge_device,usingClient_list,singlepath)