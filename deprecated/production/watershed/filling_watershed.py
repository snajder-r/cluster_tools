#! /usr/bin/python

import time
import os
import argparse
import json
import subprocess
from concurrent import futures

import numpy as np
import z5py
import vigra
import nifty
import luigi


# TODO more clean up (job config files)
# TODO computation with rois
class FillingWatershedTask(luigi.Task):
    """
    Run watersheds to fill up components
    """

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    seeds_key = luigi.Parameter()
    mask_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    # the task that makes the seeds
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)
    # TODO optional parameter to just run a subset of blocks

    def requires(self):
        return self.dependency

    # TODO there must be a more efficient way to do this
    def _make_checkerboard(self, blocking):
        blocks_a = [0]
        blocks_b = []
        all_blocks = [0]

        def recurse(current_block, insert_list):
            other_list = blocks_a if insert_list is blocks_b else blocks_b
            for dim in range(3):
                ngb_id = blocking.getNeighborId(current_block, dim, False)
                if ngb_id != -1:
                    if ngb_id not in all_blocks:
                        insert_list.append(ngb_id)
                        all_blocks.append(ngb_id)
                        recurse(ngb_id, other_list)

        recurse(0, blocks_b)
        all_blocks = blocks_a + blocks_b
        expected = set(range(blocking.numberOfBlocks))
        assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
        assert len(set(all_blocks) - expected) == 0
        assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
        return blocks_a, blocks_b

    def _prepare_jobs(self, n_jobs, block_list, config, prefix):
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'config': config,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'filling_watershed_config_%s_job%i.json' % (prefix,
                                                                                                    job_id))
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id, prefix):
        script_path = os.path.join(self.tmp_folder, 'filling_watershed.py')
        assert os.path.exists(script_path)
        config_path = os.path.join(self.tmp_folder, 'filling_watershed_config_%s_job%i.json' % (prefix,
                                                                                                job_id))
        command = '%s %s %s %s %s %i %s %s' % (script_path, self.path, self.aff_key,
                                               self.seeds_key, self.mask_key,
                                               job_id, config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_filling_watershed_%s_%i' % (prefix, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_filling_watershed_%s_%i.err' % (prefix, job_id))
        bsub_command = 'bsub -J filling_watershed_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                                   self.time_estimate,
                                                                                   log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _submit_jobs(self, n_jobs, prefix):
        from .. import util
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id, prefix)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id, prefix)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

    def _collect_outputs(self, n_blocks):
        times = []
        processed_blocks = []
        # support integeres and lists as input
        block_list = range(n_blocks) if isinstance(n_blocks, int) else n_blocks
        for block_id in block_list:
            res_file = os.path.join(self.tmp_folder, 'filling_watershed_block%i.json' % block_id)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                    times.append(res['t'])
                processed_blocks.append(block_id)
                os.remove(res_file)
            except Exception:
                continue
        return processed_blocks, times

    # normal run of the workflow
    # TODO support ROI
    def _normal_run(self, config, block_shape):
        print("Starting normal run")
        # get the shape
        f5 = z5py.File(self.path)
        shape = f5[self.seeds_key].shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)

        # TODO need to divide the blocks in 2 parts, defining a checkerboard pattern
        blocks_a, blocks_b = self._make_checkerboard(blocking)

        # find the actual number of jobs and prepare job configs
        n_jobs = min(len(blocks_a), self.max_jobs)
        self._prepare_jobs(n_jobs, blocks_a, config, 'a')
        # add halo to config for second block list
        config.update({'second_pass_filling_ws': True})
        self._prepare_jobs(n_jobs, blocks_b, config, 'b')

        # submit the jobs
        print("Start blocks a")
        self._submit_jobs(n_jobs, 'a')
        print("Start blocks b")
        self._submit_jobs(n_jobs, 'b')

        n_blocks = blocking.numberOfBlocks
        processed_blocks, times = self._collect_outputs(n_blocks)
        assert len(processed_blocks) == len(times)
        success = len(processed_blocks) == n_blocks

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with block?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'filling_watershed_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("FillingWatershedTask failed, %i / %i blocks processed," % (len(processed_blocks),
                                                                                           n_blocks)
                               + "serialized partial results to %s" % log_path)

    # second run to process failed blocks
    def _second_run(self, config, block_list):
        # number of failed blocks
        n_blocks = len(block_list)
        print("Starting salvage run for", n_blocks, "blocks")

        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)

        # prepare and run the jobs
        # add halo to the config
        config.update({'second_pass_filling_ws': True})
        self._prepare_jobs(n_jobs, block_list, config, 'second_run')
        self._submit_jobs(n_jobs, 'second_run')

        # check the outputs
        processed_blocks, times = self._collect_outputs(block_list)
        assert len(processed_blocks) == len(times)
        success = len(processed_blocks) == n_blocks
        n_processed = len(processed_blocks)

        # load and update previous results
        log_path = os.path.join(self.tmp_folder, 'filling_watershed_partial.json')
        with open(log_path) as f:
            results = json.load(f)
        times = results['times'] + times
        processed_blocks = results['processed_blocks'] + processed_blocks

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with block?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'filling_watershed_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("FillingWatershedTask failed, %i / %i blocks processed," % (n_processed,
                                                                                           n_blocks)
                               + "serialized partial results to %s" % log_path)

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'filling_watershed.py'),
                              os.path.join(self.tmp_folder, 'filling_watershed.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True
            if 'failed_blocks_filling_ws' in config:
                second_run = True
                block_list = config['failed_blocks_filling_ws']
            else:
                second_run = False

        if second_run:
            self._second_run(config, block_list)
        else:
            self._normal_run(config, block_shape)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'filling_watershed.log'))


def compute_max_seeds(hmap, boundary_threshold,
                      sigma, offset):

    # we only compute the seeds on the smaller crop of the volume
    seeds = np.zeros_like(hmap, dtype='uint64')

    for z in range(seeds.shape[0]):
        # compute distance transform on the full 2d plane
        dtz = vigra.filters.distanceTransform((hmap[z] > boundary_threshold).astype('uint32'))
        if sigma > 0:
            vigra.filters.gaussianSmoothing(dtz, sigma, out=dtz)
        # compute local maxima of the distance transform, then crop
        seeds_z = vigra.analysis.localMaxima(dtz, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        seeds_z = vigra.analysis.labelImageWithBackground(np.isnan(seeds_z).view('uint8'))
        # add offset to the seeds
        seeds_z = seeds_z.astype('uint64')
        seeds_z[seeds_z != 0] += offset
        offset = int(seeds_z.max()) + 1
        # write seeds to the corresponding slice
        seeds[z] = seeds_z
    return seeds


def run_2d_ws(hmap, seeds, mask, size_filter, offset):
    # iterate over the slices
    for z in range(seeds.shape[0]):

        # we need to remap the seeds consecutively, because vigra
        # watersheds can only handle uint32 seeds, and we WILL overflow uint32
        # however, we still need to seperate the additional from the extended seeds,
        # so we offset them
        additional_seeds_mask = seeds[z] >= offset
        seeds_z, offz, old_to_new = vigra.analysis.relabelConsecutive(seeds[z],
                                                                      start_label=1,
                                                                      keep_zeros=True)
        new_to_old = {new: old for old, new in old_to_new.items()}
        additional_seeds_ids = np.unique(seeds_z[additional_seeds_mask])
        ws_z = vigra.analysis.watershedsNew(hmap[z], seeds=seeds_z.astype('uint32'))[0]

        # apply size_filter
        if size_filter > 0:
            ids, sizes = np.unique(ws_z, return_counts=True)
            filter_ids = ids[sizes < size_filter]
            # do not filter ids that belong to the extended seeds
            filter_ids = filter_ids[np.in1d(filter_ids, additional_seeds_ids)]
            filter_mask = np.ma.masked_array(ws_z, np.in1d(ws_z, filter_ids)).mask
            ws_z[filter_mask] = 0
            vigra.analysis.watershedsNew(hmap[z], seeds=ws_z, out=ws_z)

        # set the invalid mask to zero
        ws_z[mask[z]] = 0

        # map bad to original ids
        ws_z = ws_z.astype('uint64')
        ws_z = nifty.tools.takeDict(new_to_old, ws_z)

        # write the watershed to the seeds
        seeds[z] = ws_z
    return seeds, int(seeds.max())


def ws_block(ds_affs, ds_seeds, ds_mask,
             blocking, block_id, block_config,
             empty_blocks, tmp_folder, offset,
             second_pass):

    print("Processing block", block_id)
    res_file = os.path.join(tmp_folder, 'filling_watershed_block%i.json' % block_id)

    t0 = time.time()
    if block_id in empty_blocks:
        with open(res_file, 'w') as f:
            json.dump({'t': 0}, f)
        return

    # get offset to make new seeds unique between blocks
    # (we need to relabel later to make processing efficient !)
    offset += block_id * np.prod(blocking.blockShape)

    boundary_threshold = block_config['boundary_threshold2']
    sigma_maxima = block_config['sigma_maxima']
    size_filter = block_config['size_filter']

    # halo is hard-coded for now / 100 pixel should be enough
    halo = [0, 100, 100]
    block = blocking.getBlockWithHalo(block_id, halo)

    outer_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
    local_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

    mask = ds_mask[outer_bb]

    # load affinities and make heightmap for the watershed
    bb_affs = (slice(1, 3),) + outer_bb
    affs = ds_affs[bb_affs]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = np.mean(1. - affs, axis=0)

    # load the mask and make the invalid mask by inversion
    inv_mask = np.logical_not(mask)
    affs[inv_mask] = 1

    # get the maxima seeds on 2d distance transform to fill gaps
    # in the extended seeds
    max_seeds = compute_max_seeds(affs, boundary_threshold, sigma_maxima, offset)

    # load surrounding seeds in the second pass and
    # add them to the new seeds, to get a segmentatin without blocking artefacts
    if second_pass:
        # FIXME weird runtime error I don't understand
        # where seeds cannot be read from the larger bounding box
        # this only happens very rarely, so as a quick fix, we just
        # restrict the block to the inner bounding box.
        # This will produce blocking artifacts for this block, but
        # won't harm the overall results otherwise
        try:
            seeds = ds_seeds[outer_bb]
            seed_bb = local_bb
        except RuntimeError:
            seeds = ds_seeds[inner_bb]
            mask = mask[local_bb]
            affs = affs[(slice(None),) + local_bb]
            seed_bb = np.s_[:]
        # add maxima seeds where we don't have seeds from the distance transform components
        # and where we are not in the invalid mask
        unlabeled_in_seeds = np.logical_and(seeds == 0, mask)
        seeds[unlabeled_in_seeds] += max_seeds[unlabeled_in_seeds]

    else:
        seeds = max_seeds
        seed_bb = local_bb

    # run the watershed
    seeds, max_id = run_2d_ws(affs, seeds, inv_mask, size_filter, offset)

    # write the result
    ds_seeds[inner_bb] = seeds[seed_bb]
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


def filling_ws(path, aff_key, seed_key, mask_key,
               job_id, config_file, tmp_folder):

    f5 = z5py.File(path)
    ds_affs = f5[aff_key]
    ds_seeds = f5[seed_key]
    ds_mask = f5[mask_key]

    with open(config_file) as f:
        input_config = json.load(f)
        block_list = input_config['block_list']
        config = input_config['config']
        block_shape = config['block_shape']

    # TODO shouldn't hardcode the path
    offsets_path = os.path.join(tmp_folder, 'block_offsets.json')
    with open(offsets_path) as f:
        offset_config = json.load(f)
        empty_blocks = offset_config['empty_blocks']
        offset = offset_config['n_labels']
        second_pass = 'second_pass_filling_ws' in config

    shape = ds_seeds.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    [ws_block(ds_affs, ds_seeds, ds_mask,
              blocking, int(block_id), config,
              empty_blocks, tmp_folder, offset, second_pass)
     for block_id in block_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('seed_key')
    parser.add_argument('mask_key', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('config_file', type=str)
    parser.add_argument('tmp_folder', type=str)
    args = parser.parse_args()

    filling_ws(args.path, args.aff_key,
               args.seed_key, args.mask_key,
               args.job_id, args.config_file,
               args.tmp_folder)
