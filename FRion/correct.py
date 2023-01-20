#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for applying ionospheric Faraday rotation corrections to images and
image cubes. This file contains the functions needed to apply the correction
to data cubes (either in memory, or in FITS files.)

The complex polarization (Q + iU) is divided by the predicted ionospheric
modulation to produce corrected values that should have the effect of the
ionosphere removed. These can then be saved to new Stokes Q and U FITS files.

The main functions below do not do anything specific to handle
very large FITS files gracefully. It may not perform efficiently when file
sizes are comparable to the amount of available RAM. An alternative mode,
`apply_correction_large_cube()`, has been developed to reduce the memory
footprint required.

"""

import numpy as np
from math import floor, ceil
from astropy.io import fits as pf
import os
import sys
import queue
import logging
import multiprocessing as mp


logging.basicConfig(level=logging.INFO)


def apply_correction_to_files(
    Qfile, Ufile, predictionfile, Qoutfile, Uoutfile, overwrite=False
):
    """This function combines all the individual steps needed to apply a
    correction to a set of Q and U FITS cubes and save the results.
    The user should supply the paths to all the files as specified.

    Args:
        Qfile (str): filename of uncorrected Stokes Q FITS cube
        Ufile (str): filename of uncorrected Stokes U FITS cube
        predictionfile (str): path to ionospheric modulation prediction (from predict tools)
        Qoutfile (str): filename for corrected Stokes Q FITS cube.
        Uoutfile (str): filename for corrected Stokes U FITS cube.
        overwrite (bool): overwrite Stokes Q/U files if they already exist? [False]

    """

    # Get all data:
    frequencies, theta = read_prediction(predictionfile)
    Qdata, Udata, header = readData(Qfile, Ufile)

    # Checks for data consistency.
    if Qdata.shape != Udata.shape:
        raise Exception("Q and U files don't have same dimensions.")
    if Qdata.shape[0] != theta.size:
        raise Exception(
            "Prediction file does not have same number of channels as FITS cube."
        )
    # Currently this doesn't actually check that the frequencies are the same,
    # just that the number of channels is the same. Should this be a more
    # strict check?

    # Apply correction
    Qcorr, Ucorr = correct_cubes(Qdata, Udata, theta)

    # Save results
    write_corrected_cubes(Qoutfile, Uoutfile, Qcorr, Ucorr, header, overwrite=overwrite)


def read_prediction(filename):
    """Read in frequencies and ionospheric predictions from text file.

    Returns:
        tuple containing

        -frequencies (array): frequencies of each channel (Hz);

        -theta (array): ionospheric modulation for each channel

    """
    (frequencies, real, imag) = np.genfromtxt(filename, unpack=True)
    theta = real + 1.0j * imag
    return frequencies, theta


def find_freq_axis(header):
    """Finds the frequency axis in a FITS header.
    Input: header: a Pyfits header object.
    Returns the axis number (as recorded in the FITS file, **NOT** in numpy ordering.)
    Returns 0 if the frequency axis cannot be found.

    """
    freq_axis = 0  # Default for 'frequency axis not identified'
    # Check for frequency axes. Because I don't know what different formatting
    # I might get ('FREQ' vs 'OBSFREQ' vs 'Freq' vs 'Frequency'), convert to
    # all caps and check for 'FREQ' anywhere in the axis name.
    for i in range(1, header["NAXIS"] + 1):  # Check each axis in turn.
        try:
            if "FREQ" in header["CTYPE" + str(i)].upper():
                freq_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.
    return freq_axis


def readData(Qfilename, Ufilename):
    """Open the Stokes Q and U input cubes (from the supplied
    file names) and return data-access variables and the header.
    Axes are re-ordered so that frequency is first, beyond that the number
    and ordering of axes doesn't matter.
    Uses the memmap functionality so that data isn't read into data; variables
    are just handles to access the data on disk.
    Returns the header from the Q file, the U file's header is ignored.

    """

    hdulistQ = pf.open(Qfilename, memmap=True)
    header = hdulistQ[0].header
    Qdata = hdulistQ[0].data
    hdulistU = pf.open(Ufilename, memmap=True)
    Udata = hdulistU[0].data

    N_dim = header["NAXIS"]  # Get number of axes

    freq_axis = find_freq_axis(header)
    # If the frequency axis isn't the last one, rotate the array until it is.
    # Recall that pyfits reverses the axis ordering, so we want frequency on
    # axis 0 of the numpy array.
    if freq_axis != 0 and freq_axis != N_dim:
        Qdata = np.moveaxis(Qdata, N_dim - freq_axis, 0)
        Udata = np.moveaxis(Udata, N_dim - freq_axis, 0)

    return Qdata, Udata, header


def write_corrected_cubes(
    Qoutputname, Uoutputname, Qcorr, Ucorr, header, overwrite=False
):
    """Write the corrected Q and U data to FITS files. Copies the supplied
    header, adding a note to the history saying that the correction was applied.

    Inputs:
        Qoutputname (str): filename to write corrected Stoke Q data to.
        Uoutputname (str): filename to write corrected Stoke U data to.
        Qcorr (array): corrected Stokes Q data
        Ucorr (array): corrected Stokes U data
        header: Astropy FITS header object that describes the data
        overwrite (bool): overwrite Stokes Q/U files if they already exist? [False]

    """
    output_header = header.copy()
    output_header.add_history("Corrected for ionospheric Faraday rotation using FRion.")

    # Get data back to original axis order, if necessary.
    N_dim = output_header["NAXIS"]  # Get number of axes
    freq_axis = find_freq_axis(output_header)
    if freq_axis != 0:
        Qcorr = np.moveaxis(Qcorr, 0, N_dim - freq_axis)
        Ucorr = np.moveaxis(Ucorr, 0, N_dim - freq_axis)

    pf.writeto(Qoutputname, Qcorr, output_header, overwrite=overwrite)
    pf.writeto(Uoutputname, Ucorr, output_header, overwrite=overwrite)


def correct_cubes(Qdata, Udata, theta):
    """Applies the ionospheric Faraday rotation correction to the Stokes Q/U
    data, derotating the polarization angle and renormalizing to remove
    depolarization. Note that this will amplify the noise present in the data,
    particularly if the depolarization is large (\|theta\| is small).

    Inputs:
        Qdata (array): uncorrected Stokes Q data, frequency axis first
        Udata (array): uncorrected Stokes U data, frequency axis first
        theta (1D array): ionospheric modulation, per frequency

    Returns:
        Qcorr (array): corrected Stokes Q data, same axis ordering
        Ucorr (array): corrected Stokes U data, same axis ordering
    """

    Pdata = Qdata + 1.0j * Udata  # Input complex polarization
    arrshape = np.array(Pdata.shape)  # the correction needs the same number of
    arrshape[:] = 1  # axes as the input data
    arrshape[0] = theta.size  # (but they can all be degenerate)
    Pcorr = np.true_divide(Pdata, np.reshape(theta, arrshape))
    Qcorr = Pcorr.real
    Ucorr = Pcorr.imag

    return Qcorr, Ucorr


def progress(width, percent):
    """
    Print a progress bar to the terminal.
    Stolen from Mike Bell.
    """

    marks = floor(width * (percent / 100.0))
    spaces = floor(width - marks)
    loader = "  [" + ("=" * int(marks)) + (" " * int(spaces)) + "]"
    sys.stdout.write("%s %d%%\r" % (loader, percent))
    if percent >= 100:
        sys.stdout.write("\n")
    sys.stdout.flush()


def pool_correct_task(Qoutfile, Uoutfile, Qdata, Udata, theta, lock, task_queue, complete_queue):
    """Task for correct_cubes() function to be run with multiprocessing.

    """
    while True:
        try:
            # Retrieve task parameter
            region = task_queue.get_nowait()
        except queue.Empty:
            # Break if no more tasks
            break
        else:
            # Log process and job
            pid = os.getpid()
            logging.info(f'[{pid}] Working on region {region}')

            # Run job
            N_dim = len(region)
            if N_dim == 4:
                # Update data array
                Qin = Qdata[region[0], region[1], region[2], region[3]]
                Uin = Udata[region[0], region[1], region[2], region[3]]
                Qcorr, Ucorr = correct_cubes(Qin, Uin, theta)
            elif N_dim == 3:
                Qin = Qdata[region[0], region[1], region[2]]
                Uin = Udata[region[0], region[1], region[2]]
                Qcorr, Ucorr = correct_cubes(Qin, Uin, theta)

            # Write to file with lock
            with lock:
                logging.info(f'[{pid}]: Writing to output file')
                Qout_hdu = pf.open(Qoutfile, mode="update", memmap=True)
                Uout_hdu = pf.open(Uoutfile, mode="update", memmap=True)
                if N_dim == 4:
                    Qout_hdu[0].data[region[0], region[1], region[2], region[3]] = Qcorr
                    Uout_hdu[0].data[region[0], region[1], region[2], region[3]] = Ucorr
                elif N_dim == 3:
                    Qout_hdu[0].data[region[0], region[1], region[2]] = Qcorr
                    Uout_hdu[0].data[region[0], region[1], region[2]] = Ucorr
                Qout_hdu.flush()
                Uout_hdu.flush()
                Qout_hdu.close()
                Uout_hdu.close()

            complete_queue.put(region)
    return True


def apply_correction_large_cube(
    Qfile, Ufile, predictionfile, Qoutfile, Uoutfile, overwrite=False
):
    """Functions as apply_correction_to_files, but for files too large to
    hold in memory. Combines the correct_cubes() and write_corrected_cubes()
    steps into a single function so that it can operate on smaller pieces
    of data at one time.

    This function combines all the individual steps needed to apply a
    correction to a set of Q and U FITS cubes and save the results.
    The user should supply the paths to all the files as specified.
    This function is less flexible in terms of axis ordering: it assumes two
    spatial axes, and a frequency axis in position 3 or 4.

    Args:
        Qfile (str): filename of uncorrected Stokes Q FITS cube
        Ufile (str): filename of uncorrected Stokes U FITS cube
        predictionfile (str): path to ionospheric modulation prediction (from predict tools)
        Qoutfile (str): filename for corrected Stokes Q FITS cube.
        Uoutfile (str): filename for corrected Stokes U FITS cube.
        overwrite (bool): overwrite Stokes Q/U files if they already exist? [False]

    """

    # Get all data:
    frequencies, theta = read_prediction(predictionfile)

    hdulistQ = pf.open(Qfile, memmap=True)
    header = hdulistQ[0].header
    Qdata = hdulistQ[0].data
    hdulistU = pf.open(Ufile, memmap=True)
    Udata = hdulistU[0].data

    N_dim = header["NAXIS"]  # Get number of axes
    freq_axis = find_freq_axis(header)
    # Checks for data consistency.
    if Qdata.shape != Udata.shape:
        raise Exception("Q and U files don't have same dimensions.")
    if Qdata.shape[N_dim - freq_axis] != theta.size:
        raise Exception(
            "Prediction file does not have same number of channels as FITS cube."
        )
    # Currently this doesn't actually check that the frequencies are the same,
    # just that the number of channels is the same. Should this be a more
    # strict check?
    if Qdata.ndim != 3 and Qdata.ndim != 4:
        raise Exception(
            "Cube does not have 3 or 4 axes; only these are supported for large files."
        )

    # Add correction to header history
    output_header = header.copy()
    output_header.add_history("Corrected for ionospheric Faraday rotation using FRion.")

    # Deal with any existing output files:
    if (os.path.isfile(Qoutfile) or os.path.isfile(Uoutfile)) and not overwrite:
        raise Exception("Output file(s) aready exist.")
    if os.path.isfile(Qoutfile) and overwrite:
        os.remove(Qoutfile)
    if os.path.isfile(Uoutfile) and overwrite:
        os.remove(Uoutfile)

    # Create large blank files. This seems to produce file size complaints
    # sometimes, but those seem harmless so far.
    shape = tuple(
        output_header["NAXIS{0}".format(ii)]
        for ii in range(1, output_header["NAXIS"] + 1)
    )

    output_header.tofile(Qoutfile)
    with open(Qoutfile, "rb+") as fobj:
        fobj.seek(
            len(output_header.tostring())
            + (np.product(shape) * np.abs(output_header["BITPIX"] // 8))
            - 1
        )
        fobj.write(b"\0")

    output_header.tofile(Uoutfile)
    with open(Uoutfile, "rb+") as fobj:
        fobj.seek(
            len(output_header.tostring())
            + (np.product(shape) * np.abs(output_header["BITPIX"] // 8))
            - 1
        )
        fobj.write(b"\0")

    # Need to decide how to split the cube
    nsplit_x = 10
    nsplit_y = 10
    size_x = ceil(int(output_header["NAXIS1"]) / nsplit_x)
    size_y = ceil(int(output_header["NAXIS2"]) / nsplit_y)

    # multiprocessing
    n_processes = 4
    processes = []
    m = mp.Manager()
    lock = mp.Lock()
    task_queue = m.Queue()
    complete_queue = m.Queue()

    # Get subcubes
    # regions = []
    for i in range(nsplit_x):
        xmin = i * size_x
        xmax = (i + 1) * size_x
        if (i + 1 == nsplit_x):
            xmax = int(output_header["NAXIS1"])
        for j in range(nsplit_y):
            ymin = j * size_y
            ymax = (j + 1) * size_y
            if (j + 1 == nsplit_y):
                ymax = int(output_header["NAXIS2"])
            if N_dim == 4:
                region = (slice(None), slice(None), slice(ymin, ymax), slice(xmin, xmax))
            elif N_dim == 3:
                region = (slice(None), slice(ymin, ymax), slice(xmin, xmax))
            # regions.append(region)
            task_queue.put(region)
    logging.info(f'Number of regions to process in parallel: {task_queue.qsize()}')

    # Large sequential update regions
    # for r in regions:
    #     Qin = Qdata[r[0], r[1], r[2], r[3]]
    #     Uin = Udata[r[0], r[1], r[2], r[3]]
    #     Qcorr, Ucorr = correct_cubes(Qin, Uin, theta)
    #     Qdata[r[0], r[1], r[2], r[3]] = Qcorr
    #     Udata[r[0], r[1], r[2], r[3]] = Ucorr

    # Run in process pool
    logging.info('Starting parallel region')
    logging.info(f'Main process id: {os.getpid()}')
    for pid in range(n_processes):
        p = mp.Process(
            target=pool_correct_task,
            args=(Qoutfile, Uoutfile, Qdata, Udata, theta, lock, task_queue, complete_queue)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    logging.info('Finished parallel region')
    logging.info('Write complete')


def command_line():
    """When invoked from the command line, parse the input options to get the
    filenames and other parameters, then invoke apply_correction_to_files
    to run all the steps and save the output cubes.

    """

    import argparse
    import os

    descStr = """
    Apply correction for ionospheric Faraday rotation to Stokes Q and U FITS
    cubes. Requires the file names for the input cubes, output cubes, and the
    prediction file (which contains the ionospheric modulation per channel
    in the cubes).
    """

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "fitsQ",
        metavar="fitsQ",
        help="FITS cube containing (uncorrected) Stokes Q data.",
    )
    parser.add_argument(
        "fitsU",
        metavar="fitsU",
        help="FITS cube containing (uncorrected) Stokes U data.",
    )
    parser.add_argument(
        "predictionfile",
        metavar="predictionfile",
        help="File containing ionospheric prediction to be applied.",
    )
    parser.add_argument(
        "outQ",
        metavar="Qcorrected",
        help="Output filename for corrected Stokes Q cube.",
    )
    parser.add_argument(
        "outU",
        metavar="Ucorrected",
        help="Output filename for corrected Stokes U cube.",
    )
    parser.add_argument(
        "-o",
        dest="overwrite",
        action="store_true",
        help="Overwrite exising output files? [False]",
    )
    parser.add_argument(
        "-L",
        dest="large",
        action="store_true",
        help="Use large-file mode? (Reduced memory footprint) [False]",
    )

    args = parser.parse_args()

    # Check for file existence.
    if not os.path.isfile(args.fitsQ):
        raise Exception("Stokes Q file not found.")
    if not os.path.isfile(args.fitsU):
        raise Exception("Stokes U file not found.")

    # Pass file names into do-everything function (either basic or large-file, as
    # set by user.
    if args.large:
        apply_correction_large_cube(
            args.fitsQ,
            args.fitsU,
            args.predictionfile,
            args.outQ,
            args.outU,
            overwrite=args.overwrite,
        )
    else:
        apply_correction_to_files(
            args.fitsQ,
            args.fitsU,
            args.predictionfile,
            args.outQ,
            args.outU,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    command_line()
