from benchmark.mapfree_combiner import compute_scene_metrics, aggregate_results
from config.default import cfg
from zipfile import ZipFile
from pathlib import Path

def main(submission_path_gs, nogs=None, rot_diff_cutoff=0, trans_diff_cutoff=0, conf_cutoff_est=0, conf_cutoff_gs=0):
    cfg.merge_from_file('config/mapfree.yaml')
    dataset_path = Path(cfg.DATASET.DATA_ROOT)
    split = 'val'
    dataset_path = dataset_path / split

    scenes = tuple(f.name for f in dataset_path.iterdir() if f.is_dir())

    submission_zip_gs = ZipFile(submission_path_gs, 'r')
    if nogs is not None:
        submission_zip_nogs = ZipFile(nogs, 'r')
    else:
        submission_zip_nogs = None

    all_results = dict()
    all_failures = 0
    for scene in scenes:
        metrics, failures = compute_scene_metrics(
            dataset_path, submission_zip_gs, submission_zip_nogs, scene, rot_diff_cutoff, trans_diff_cutoff, conf_cutoff_est, conf_cutoff_gs)

        all_results[scene] = metrics
        all_failures += failures

    # unexpected_scene_count = count_unexpected_scenes(scenes, submission_zip)
    # if unexpected_scene_count > 0:
    #     logging.warning(
    #         f'Submission contains estimates for {unexpected_scene_count} scenes outside the {args.split} set')

    if all((len(metrics) == 0 for metrics in all_results.values())):
        logging.error( 
            f'Submission does not have any valid pose estimates')
        return

    output_metrics = aggregate_results(all_results, all_failures)

    name = "results"
    if rot_diff_cutoff:
        name += ("_r" + str(rot_diff_cutoff))
    else:
        name += ("_r" + str(0))
    if trans_diff_cutoff:
        name += ("_t" + str(trans_diff_cutoff))
    else:
        name += ("_t" + str(0))
    if conf_cutoff_est:
        name += ("_e" + str(conf_cutoff_est))
    else:
        name += ("_e" + str(0))
    if conf_cutoff_gs:
        name += ("_g" + str(conf_cutoff_gs))
    else:
        name += ("_g" + str(0))
        
    # with open(f"results/{str(args.submission_path_gs).split('/')[1]}/{name}.txt", mode='w', encoding='utf-8') as f:
        # json.dump(output_metrics, f, ensure_ascii=False, indent=4)
    print(name)
    for k,v in output_metrics.items():
        print(f"{k}: {v}")

for r in range(0, 100, 10):
    # main("results/posediffusionPGS2D3D/submission.zip", rot_diff_cutoff=r)
    main("results/sg_emat_dptkitti/submission.zip", rot_diff_cutoff=r)

    # ""