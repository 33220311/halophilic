# Copyright 2022 Rostlab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import json
import numpy as np
import argparse as argp
import subprocess as cmd
from shutil import which
import sys

from math import exp
from math import ceil
from pathlib import Path
from tempfile import mkdtemp

from mmseqs2_uniqueprot.clustering import get_clusters


def load_config(json_file):
    defaults = {'e_value': 10,
                'sensitivity': 7.5,
                'num_iterations': 2,
                'realign': 0,
                'min_aln_len': 11,
                'max_seqs': 10000000,
                'alt_ali': 999,
                'split_size': 2500,
                'db_load_mode': 3}

    if not json_file.exists():
        return defaults

    with json_file.open('r') as jf:
        config = json.load(jf)

    for param, value in defaults.items():
        if param in config:
            if type(config[param]) != type(value):
                raise TypeError(f'{param} must be {type(value)}.')
        else:
            config[param] = value

    return config


def setup_dirs(config):
    job_name = config['input_file'].stem

    config['work_dir'].mkdir(parents=True, exist_ok=True)

    config['work_dir'] = Path(mkdtemp(prefix=job_name, dir=config['work_dir']))

    config['db_dir'] = Path(config['work_dir'], 'db')
    config['fasta_dir'] = Path(config['work_dir'], 'fasta')
    config['results_dir'] = Path(config['work_dir'], 'results')

    config['db_dir'].mkdir(parents=True, exist_ok=True)
    config['fasta_dir'].mkdir(parents=True, exist_ok=True)
    config['results_dir'].mkdir(parents=True, exist_ok=True)


def read_fasta(fasta_file):
    sequences = []

    with fasta_file.open('r') as ff:
        header = ''
        sequence = []

        for line in ff:
            line = line.strip()

            if line.startswith('>'):
                if header and sequence:
                    sequences.append((header, ''.join(sequence)))

                header = line[1:]
                sequence = []
            else:
                sequence.extend(line.split())

        if header and sequence:
            sequences.append((header, ''.join(sequence)))

    return sequences


def write_fasta(fasta_file, sequences, idx_header=False):
    with fasta_file.open('w') as ff:
        for i, (header, sequence) in enumerate(sequences):
            if idx_header:
                ff.write(f'>{i}\n')
            else:
                ff.write(f'>{header}\n')
            ff.write(f'{sequence}\n')


def split_fasta(sequences, root_dir, split_size):
    num_seqs = len(sequences)
    num_splits = num_seqs // split_size

    if num_seqs % split_size != 0:
        num_splits = num_splits + 1

    for n in range(num_splits):
        offset = n * split_size
        split_seq = sequences[offset:(offset + split_size)]

        with Path(root_dir, f'query_{n:09}.fasta').open('w') as ff:
            for i, (header, sequence) in enumerate(split_seq):
                ff.write(f'>{i + offset}\n')
                ff.write(f'{sequence}\n')


def create_db(fasta_file, db_file, work_dir, with_index=False, touch_db=False):
    cmd.run(['mmseqs', 'createdb', str(fasta_file), str(db_file)])

    if with_index:
        cmd.run(['mmseqs', 'createindex', str(db_file), str(work_dir)])

    if touch_db:
        cmd.run(['mmseqs', 'touchdb', str(db_file)])
        # cmd.run(['vmtouch', '-l', '-d', '-t', f'{db_file}.idx'])


def search(query_db, target_db, result_db, work_dir, config):
    base_cmd = ['mmseqs',
                'search',
                str(query_db),
                str(target_db),
                str(result_db),
                str(work_dir)]

    # fixed parameters
    base_cmd.extend(['-a', '1'])
    base_cmd.extend(['--alignment-mode', '3'])

    # configurable parameters
    base_cmd.extend(['-e', str(config['e_value'])])
    base_cmd.extend(['-s', str(config['sensitivity'])])
    base_cmd.extend(['--realign', str(config['realign'])])
    base_cmd.extend(['--alt-ali', str(config['alt_ali'])])
    base_cmd.extend(['--max-seqs', str(config['max_seqs'])])
    base_cmd.extend(['--min-aln-len', str(config['min_aln_len'])])
    base_cmd.extend(['--db-load-mode', str(config['db_load_mode'])])
    base_cmd.extend(['--num-iterations', str(config['num_iterations'])])

    if not config['keep_temp']:
        base_cmd.extend(['--remove-tmp-files', '1'])

    cmd.run(base_cmd)


def convert(query_db, target_db, result_db, result_file):
    base_cmd = ['mmseqs',
                'convertalis',
                str(query_db),
                str(target_db),
                str(result_db),
                str(result_file)]

    params = ['--format-mode', '0']
    fields = ['--format-output', 'query,target,nident,mismatch,qlen,tlen']

    base_cmd.extend(params)
    base_cmd.extend(fields)

    cmd.run(base_cmd)


def get_hval(seq_identity, aln_length):
    if aln_length < 12:
        return (seq_identity - 100)
    elif aln_length > 450:
        return (seq_identity - 19.5)
    else:
        exponent = -0.32 * (1.0 + exp(-1.0 * (aln_length / 1000.0)))
        return (seq_identity - (480.0 * (aln_length ** exponent)))


def read_results(result_file):
    alignments = []

    with result_file.open('r') as rf:
        for line in rf:
            data = line.split()

            assert len(data) == 6

            query, target, nident, mismatch, qlen, tlen = data

            qlen = int(qlen)
            tlen = int(tlen)
            query = int(query)
            target = int(target)
            nident = int(nident)
            mismatch = int(mismatch)

            m_len = min(qlen, tlen)
            aln_length = nident + mismatch
            seq_identity = 100.0 * nident / aln_length

            alignments.append((query, target, seq_identity, aln_length, m_len))

    return alignments


def write_mapping(matrix_file, sequences):
    map_file = Path(matrix_file.parent, f'{matrix_file.stem}.idx.map')

    with map_file.open('w') as mf:
        for i, (header, sequence) in enumerate(sequences):
            seq_id, *_ = header.split(maxsplit=1)

            mf.write(f'{i}\t{seq_id}\n')


def write_results(fasta_file, clusters, sequences):
    representatives = [sequences[r] for r, m in clusters]

    write_fasta(fasta_file, representatives)


def write_clusters(cluster_file, clusters, sequences, hval_mat):
    with cluster_file.open('w') as cf:
        cf.write(f'cluster,member,hval\n')

        for rep, members in clusters:
            header, sequence = sequences[rep]
            rep_id, *_ = header.split(maxsplit=1)

            cf.write(f'{rep_id},{rep_id},{hval_mat[rep][rep]}\n')

            for mem in members:
                header, sequence = sequences[mem]
                mem_id, *_ = header.split(maxsplit=1)

                cf.write(f'{rep_id},{mem_id},{hval_mat[rep][mem]}\n')


def clean_up(root_path):
    for child in list(root_path.glob('*')):
        # Path.is_file() fails for empty files
        try:
            is_file = child.is_file()
        except:
            is_file = True

        if is_file or child.is_symlink():
            child.unlink()
        else:
            clean_up(child)

    root_path.rmdir()


def main(config):
    setup_dirs(config)

    sequences = read_fasta(config['input_file'])

    target_fasta = Path(config['fasta_dir'], 'target.fasta')

    write_fasta(target_fasta, sequences, idx_header=True)

    target_db = Path(config['db_dir'], 'target_db')

    create_db(target_fasta, target_db, config['work_dir'],
              with_index=True, touch_db=True)

    split_fasta(sequences, config['fasta_dir'], config['split_size'])

    query_files = list(config['fasta_dir'].glob('query_*.fasta'))

    for i, query_file in enumerate(sorted(query_files)):
        result_file = Path(config['results_dir'], f'result_{i}.tab')

        query_db = Path(config['db_dir'], f'query_db_{i}')
        result_db = Path(config['db_dir'], f'result_db_{i}')

        create_db(query_file, query_db, config['work_dir'])
        search(query_db, target_db, result_db, config['work_dir'], config)
        convert(query_db, target_db, result_db, result_file)

    n_seq = len(sequences)
    hval_mat = np.zeros(shape=(n_seq, n_seq), dtype=np.int8)

    hval_mat.fill(-100)

    for result_file in config['results_dir'].glob('result_*.tab'):
        alignments = read_results(result_file)

        for query, target, seq_identity, aln_length, m_len in alignments:
            if m_len >= 180 and aln_length < 50:
                continue

            hval = get_hval(seq_identity, aln_length)

            # round to next higher int with epsilon=0.005
            hval = ceil(hval - 0.005)

            if hval > hval_mat[query][target]:
                hval_mat[query][target] = hval
                hval_mat[target][query] = hval

    if config['matrix_file']:
        np.save(config['matrix_file'], hval_mat)
        write_mapping(config['matrix_file'], sequences)

    if config['output_file'] or config['cluster_file']:
        if not config['pass_sequences']:
            clusters = get_clusters(hval_mat, config['threshold'], None)
        else:
            clusters = get_clusters(hval_mat, config['threshold'], sequences)

    if config['output_file']:
        write_results(config['output_file'], clusters, sequences)

    if config['cluster_file']:
        write_clusters(config['cluster_file'], clusters, sequences, hval_mat)

    if not config['keep_temp']:
        clean_up(config['work_dir'])

def setup_parse_args():
    arg_parser = argp.ArgumentParser(description='MMseqs2-UniqueProt')

    arg_parser.add_argument('-i', '--input-file', required=True)

    arg_parser.add_argument('-o', '--output-file', default=None)
    arg_parser.add_argument('-m', '--matrix-file', default=None)
    arg_parser.add_argument('-c', '--cluster-file', default=None)

    arg_parser.add_argument('-t', '--threshold', default=0)
    arg_parser.add_argument('-p', '--pass-sequences',
                            action='store_true', default=False)

    arg_parser.add_argument('-w', '--work-dir', default=None)
    arg_parser.add_argument('-k', '--keep-temp',
                            action='store_true', default=False)
    arg_parser.add_argument('-j', '--config-file', default=None)

    args = arg_parser.parse_args()

    if args.config_file:
        config = load_config(Path(args.config_file))
    else:
        config = load_config(Path(__file__).parent/'defaults.json')


    config['input_file'] = Path(args.input_file)

    if not config['input_file'].exists():
        raise FileNotFoundError(config['input_file'])
    elif not config['input_file'].is_file():
        raise ValueError('Input file is not a file.')

    if args.output_file:
        config['output_file'] = Path(args.output_file)
    else:
        config['output_file'] = None

    if args.matrix_file:
        config['matrix_file'] = Path(args.matrix_file)
    else:
        config['matrix_file'] = None

    if args.cluster_file:
        config['cluster_file'] = Path(args.cluster_file)
    else:
        config['cluster_file'] = None

    if not any([config['output_file'],
                config['matrix_file'],
                config['cluster_file']]):
        raise ValueError('Please specify at least one type of output file.')

    config['threshold'] = int(args.threshold)
    config['pass_sequences'] = args.pass_sequences

    if args.work_dir:
        config['work_dir'] = Path(args.work_dir)
    else:
        config['work_dir'] = config['input_file'].parent

    config['keep_temp'] = args.keep_temp
    return config


def check_mmseqs_avail():
    """Check whether `MMSeqs2` is on PATH and marked as executable."""
    return which('mmseqs') is not None
    
def run():
    config = setup_parse_args()
    if not check_mmseqs_avail():
        print('MMseqs2 could not be found in your PATH variable. Please make sure it is installed!')
        print('You can check: https://github.com/soedinglab/MMseqs2 for installation details.')
        sys.exit(-1)

    main(config)
      

if __name__ == '__main__':
    run()
