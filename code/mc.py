# First commit
import json
import numpy as np

if __name__== '__main__':
    ### Prepare arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    sample = np.random.normal(size=10000)
    bundle = {
        "J": 1, "h": 0.08,
        "sample": sample.tolist()
    }
    with open(args.output, 'w') as file: json.dump(bundle, file)
    print(f"Simulations saved to {args.output}")

# import json
# with open(path, 'r+') as file: bundle = json.load(file)
# print("J is:", bundle["J"])
