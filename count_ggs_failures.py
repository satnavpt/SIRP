import os

successes = 0
failures = -1
seen_success = False
with open('./results/posediffusionGGSCROP224/ggs_results.txt', mode='r') as f:
    for l in f:
        l = l.split(' ')[-1].strip()
        if l.isdecimal():
            if int(l) != 0:
                seen_success = True
        else:
            if seen_success:
                successes += 1
                seen_success = False
            else:
                failures += 1

print(f'successes: {successes}')
print(f'failures: {failures}')
