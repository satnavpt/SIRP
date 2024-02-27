import os
import numpy as np

successes = []
failures = []
seen_success = False
last_test = None
with open('./results/posediffusionGGSCROP224/ggs_results.txt', mode='r') as f:
    for l in f:
        n = l.split(' ')[-1].strip()
        if n.isdecimal():
            if int(n) != 0:
                seen_success = True
        else:
            if seen_success:
                # successes += 1
                if last_test:
                    successes.append(last_test)
                seen_success = False
            else:
                if last_test:
                    failures.append(last_test)
                # failures += 1
            scene_r, q = l.split(' ')
            s, r = scene_r.split('][')
            s = s[2:-1]
            r = r[2:-4]
            q = q[2:-5]
            last_test = f"{s}.{r}.{q}"


# print(f'successes: {(successes)}')
# print(f'failures: {(failures)}')

trans_errs = {'s':[], 'f':[]}
rot_errs = {'s':[], 'f':[]}
reproj_errs = {'s':[], 'f':[]}

with open('./results/posediffusionGGSCROP224/detailed_res.txt', mode='r') as f:
    for l in f:
        s, r, q, trans_err, rot_err, reproj_err = l.split(',')
        key = f"{s}.{r}.{q}"
        if key in successes:
            trans_errs['s'].append(trans_err)
            rot_errs['s'].append(rot_err)
            reproj_errs['s'].append(reproj_err)
        elif key in failures:
            trans_errs['f'].append(trans_err)
            rot_errs['f'].append(rot_err)
            reproj_errs['f'].append(reproj_err)
        else:
            pass

print(f'mean trans_err for successes: {np.mean(np.array(trans_errs["s"]).astype("float64"))}')
print(f'mean trans_err for failures: {np.mean(np.array(trans_errs["f"]).astype("float64"))}')
print(f'mean rot_err for successes: {np.mean(np.array(rot_errs["s"]).astype("float64"))}')
print(f'mean rot_err for failures: {np.mean(np.array(rot_errs["f"]).astype("float64"))}')
print(f'mean reproj_err for successes: {np.mean(np.array(reproj_errs["s"]).astype("float64"))}')
print(f'mean reproj_err for failures: {np.mean(np.array(reproj_errs["f"]).astype("float64"))}')