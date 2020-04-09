#!/usr/bin/env python3

"""
Download CENDL 3.1 data from OECD NEA and convert it to a HDF5 library for
use with OpenMC.
"""

import argparse
import zipfile
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from urllib.parse import urljoin

import openmc.data
from utils import download, process_neutron


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=CustomFormatter
)
parser.add_argument('-d', '--destination', type=Path, default=None,
                    help='Directory to create new library in')
parser.add_argument('--download', action='store_true',
                    help='Download files from OECD-NEA')
parser.add_argument('--no-download', dest='download', action='store_false',
                    help='Do not download files from OECD-NEA')
parser.add_argument('--extract', action='store_true',
                    help='Extract tar/zip files')
parser.add_argument('--no-extract', dest='extract', action='store_false',
                    help='Do not extract tar/zip files')
parser.add_argument('--libver', choices=['earliest', 'latest'],
                    default='latest', help="Output HDF5 versioning. Use "
                    "'earliest' for backwards compatibility or 'latest' for "
                    "performance")
parser.add_argument('-r', '--release', choices=['3.1'],
                    default='3.1', help="The nuclear data library release "
                    "version. The only option currently supported is 3.1")
parser.add_argument('--cleanup', action='store_true',
                    help="Remove download directories when data has "
                    "been processed")
parser.add_argument('--no-cleanup', dest='cleanup', action='store_false',
                    help="Do not remove download directories when data has "
                    "been processed")
parser.set_defaults(download=True, extract=True, cleanup=False)
args = parser.parse_args()


library_name = 'cendl'

cwd = Path.cwd()

endf_files_dir = cwd.joinpath('-'.join([library_name, args.release, 'endf']))
download_path = cwd.joinpath('-'.join([library_name, args.release, 'download']))
# the destination is decided after the release is known
# to avoid putting the release in a folder with a misleading name
if args.destination is None:
    args.destination = Path('-'.join([library_name, args.release, 'hdf5']))

# This dictionary contains all the unique information about each release.
# This can be extended to accommodate new releases
release_details = {
    '3.1': {
        'base_url': 'https://www.oecd-nea.org/dbforms/data/eva/evatapes/cendl_31/',
        'compressed_files': ['CENDL-31.zip'],
        'neutron_files': endf_files_dir.glob('*.C31'),
        'metastables': endf_files_dir.glob('*m.C31'),
        'compressed_file_size': '0.03 GB',
        'uncompressed_file_size': '0.4 GB'
    }
}

download_warning = """
WARNING: This script will download {} of data.
Extracting and processing the data requires {} of additional free disk space.
""".format(release_details[args.release]['compressed_file_size'],
           release_details[args.release]['uncompressed_file_size'])

# ==============================================================================
# DOWNLOAD FILES FROM WEBSITE

if args.download:
    print(download_warning)
    for f in release_details[args.release]['compressed_files']:
        # Establish connection to URL
        download(urljoin(release_details[args.release]['base_url'], f),
                 output_path=download_path)


# ==============================================================================
# EXTRACT FILES FROM ZIP
if args.extract:
    for f in release_details[args.release]['compressed_files']:
        with zipfile.ZipFile(download_path / f) as zf:
            print('Extracting {0}...'.format(f))
            zf.extractall(path=endf_files_dir)

    if args.cleanup and download_path.exists():
        rmtree(download_path)

# ==============================================================================
# GENERATE HDF5 LIBRARY -- NEUTRON FILES

# Get a list of all ENDF files
neutron_files = release_details[args.release]['neutron_files']

# Create output directory if it doesn't exist
args.destination.mkdir(parents=True, exist_ok=True)

library = openmc.data.DataLibrary()

with Pool() as pool:
    results = []
    for filename in sorted(neutron_files):

        # this is a fix for the CENDL 3.1 release where the
        # 22-Ti-047.C31 and 5-B-010.C31 files contain non-ASCII characters
        if library_name == 'cendl' and args.release == '3.1' and filename.name in ['22-Ti-047.C31', '5-B-010.C31']:
            print('Manual fix for incorrect value in ENDF file')
            text = open(filename, 'rb').read().decode('utf-8', 'ignore').split('\r\n')
            if filename.name == '22-Ti-047.C31':
                text[205] = ' 8) YUAN Junqian,WANG Yongchang,etc.               ,16,(1),57,92012228 1451  205'
            if filename.name == '5-B-010.C31':
                text[203] = '21)   Day R.B. and Walt M.  Phys.rev.117,1330 (1960)               525 1451  203'
            open(filename, 'w').write('\r\n'.join(text))

        func_args = (filename, args.destination, args.libver)
        r = pool.apply_async(process_neutron, func_args)
        results.append(r)

    for r in results:
        r.wait()

# Register with library
for p in sorted((args.destination).glob('*.h5')):
    library.register_file(p)

# Write cross_sections.xml
library.export_to_xml(args.destination / 'cross_sections.xml')
