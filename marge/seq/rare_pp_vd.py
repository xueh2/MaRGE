"""
Created on Thu June 2, 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import os
import sys
import csv


#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import marge.controller.experiment_gui as ex
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marge.marge_utils import utils

from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp
from marge.marge_tyger import tyger_rare
import marge.marge_tyger.tyger_config as tyger_conf
from marge.marge_tyger import tyger_denoising

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RarePyPulseqVD(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RarePyPulseqVD, self).__init__()
        # Input the parameters
        self.oversampling_factor = None
        self.decimation_factor = None
        self.add_rd_points = None
        self.nNoise = None
        self.tyger_denoising = None
        self.boFit_file = None
        self.tyger_recon = None
        self.recon_type = None
        self.image_orientation_dicom = None
        self.sequence_list = None
        self.unlock_orientation = None
        self.rdDephTime = None
        self.dummyPulses = None
        self.nScans = None
        self.standalone = None
        self.repetitionTime = None
        self.inversionTime = None
        self.preExTime = None
        self.sweepMode = None
        self.expt = None
        self.echoSpacing = None
        self.phGradTime = None
        self.rdGradTime = None
        self.parFourierFraction = None
        self.etl = None
        self.rfReFA = None
        self.rfExFA = None
        self.rfReTime = None
        self.rfExTime = None
        self.acqTime = None
        self.freqOffset = None
        self.nPoints = None
        self.dfov = None
        self.fov = None
        self.system = None
        self.echoMode = None
        self.axesOrientation = None
        # Variable flip angle parameters
        self.vfaMode = None
        self.rfReFAFinal = None
        self.vfaConstantEchoes = None
        # Variable density parameters
        self.densityMode = None
        self.accelerationFactor = None
        self.undersamplingType = None
        self.undersamplingAxis = None
        self.smoothTrajectory = None
        self.calibrationSize = None
        self.addParameter(key='seqName', string='RAREInfo', val='RarePyPulseqVD')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM') ## number of scans
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='echoMode', string='Echoes', val='All', field='SEQ', tip="'All', 'Odd', 'Even'")
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[12.0, 12.0, 12.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[120, 120, 20], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=4, field='SEQ') ## nm of peaks in 1 repetition
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='nNoise', string='Noise acquisitions', val=1, field='SEQ', tip="Number of noise acquisitions")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=0.7, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.addParameter(key='full_plot', string='Full plot', val=False, field='OTH',
                          tip="'True' or 'False' to plot odd and even images separately")
        self.addParameter(key='k_fill', string='Filling method', val='ZP', field='PRO',
                          tip="'ZP': Zero Padding, 'POCS': Projection Onto Convex Sets")
        self.addParameter(key='tyger_recon', string='Tyger reconstruction', val=0, field='PRO',
                          tip='To reconstruct with Tyger (0 = Disabled; 1 = Enabled)')
        self.addParameter(key='tyger_denoising', string='Denoising (SNRAware TEP)', val=0, field='PRO',
                          tip='To denoising with Tyger (0 = Disabled; 1 = Enabled)')
        self.addParameter(key='recon_type', string='Reconstruction type', val='cp', field='PRO',
                          tip='Options: cp or artpk.')
        self.addParameter(key='boFit_file', string='Bo Fit file', val='boFit_default.txt', field='PRO',
                          tip='Path to the Bo Fit file inside [b0_maps] folder.')
        self.addParameter(key='rd_direction', string='Rd direction', val=1, field='SEQ',
                          tip='Set the readout direction to positive (1) or negative (-1)')
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')
        # Variable Flip Angle parameters
        self.addParameter(key='vfaMode', string='VFA mode', val='Constant', field='RF',
                          tip="'Constant', 'ExponentialDecay', 'LinearDecay', 'ConstantThenDecay', 'LinearIncreaseThenConstant', 'ExponentialIncrease'")
        self.addParameter(key='rfReFAFinal', string='Final refocusing FA (º)', val=90, field='RF',
                          tip='Final refocusing flip angle for VFA decay schedules')
        self.addParameter(key='vfaConstantEchoes', string='VFA constant echoes', val=0, field='RF',
                          tip='Number of constant-FA echoes before decay (ConstantThenDecay mode)')
        # Variable Density parameters
        self.addParameter(key='densityMode', string='K-space density', val='Uniform', field='SEQ',
                          tip="'Uniform': standard. 'CenterOut': 2D center-out. 'EllipticalCenterOut': elliptical center-out. "
                              "'LinearCenterOut': center-out along k_phase within each train; trains traverse k_slice center-out.")
        self.addParameter(key='accelerationFactor', string='Acceleration factor', val=1.0, field='SEQ',
                          tip='Undersampling acceleration (1.0 = fully sampled)')
        self.addParameter(key='undersamplingType', string='Undersampling type', val='None', field='SEQ',
                          tip="'None': fully sampled. 'PoissonDisk': Poisson-disk. 'GaussianRandom': Gaussian random.")
        self.addParameter(key='undersamplingAxis', string='Undersampling axis', val='both', field='SEQ',
                          tip="'both': undersample phase and slice. 'phase': phase only. 'slice': slice only.")
        self.addParameter(key='calibrationSize', string='Calibration region size', val=16, field='SEQ',
                          tip='Size of fully-sampled calibration region (lines) at k-space center')
        self.addParameter(key='smoothTrajectory', string='Smooth k-space trajectory', val=2, field='SEQ',
                          tip='0: distance-sorted. 1: greedy NN. 2: NN + 2-opt (best). 3: distance-sorted within trains + inter-train smoothing.')

        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header = ismrmrd.xsd.ismrmrdHeader()

    def sequenceInfo(self):
        print("3D RARE sequence powered by PyPulseq (Enhanced)")
        print("Features: Variable Flip Angle, Variable Density Trajectory, Auto Echo Spacing")
        print("Based on original by: Dr. J.M. Algarín")


    def sequenceTime(self):
        n_scans = self.mapVals['nScans']
        n_points = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        repetition_time = self.mapVals['repetitionTime']
        par_fourier_fraction = self.mapVals['parFourierFraction']
        acceleration_factor = self.mapVals.get('accelerationFactor', 1.0)

        # check if rf amplitude is too high
        rf_ex_fa = self.mapVals['rfExFA'] / 180 * np.pi  # rads
        rf_re_fa = self.mapVals['rfReFA'] / 180 * np.pi  # rads
        rf_ex_time = self.mapVals['rfExTime']  # us
        rf_re_time = self.mapVals['rfReTime']  # us
        rf_ex_amp = rf_ex_fa / (rf_ex_time * hw.b1Efficiency)
        rf_re_amp = rf_re_fa / (rf_re_time * hw.b1Efficiency)
        if rf_ex_amp>1 or rf_re_amp>1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        seq_time = (n_points[1]/etl*n_points[2]*repetition_time*1e-3*n_scans
                    *par_fourier_fraction / acceleration_factor / 60)
        seq_time = np.round(seq_time, decimals=1)
        return seq_time  # minutes, scanTime

    def calculateMinEchoSpacing(self, rf_re_time, rd_grad_time, acq_time, ph_grad_time):
        """
        Calculate the minimum physically achievable echo spacing based on hardware and sequence parameters.

        All input times are in seconds (SI).

        Returns:
            float: Minimum echo spacing in seconds.
        """
        grad_rise_time = hw.grad_rise_time  # s
        rf_dead_time = hw.blkTime * 1e-6  # s

        rd_time = max(rd_grad_time, acq_time)

        # Time from center of refocusing to center of echo:
        #   rf_re/2 + dead_time + grad_rise + ph_grad + grad_rise + rd_time/2
        half_spacing = (rf_re_time / 2
                        + rf_dead_time
                        + grad_rise_time
                        + max(ph_grad_time, 0)
                        + grad_rise_time
                        + rd_time / 2)

        min_echo_spacing = 2 * half_spacing

        # Round up to gradient raster time
        raster = hw.grad_raster_time
        min_echo_spacing = np.ceil(min_echo_spacing / raster) * raster

        return min_echo_spacing

    def validateHardwareLimits(self):
        """
        Validate that all sequence gradient amplitudes and RF are within hardware limits.

        Reads per-axis gradient limits from hw_gradients.csv and checks:
        1. Each gradient axis amplitude against its per-axis max
        2. Slew rate (amplitude / rise_time) against max slew rate
        3. RF amplitude feasibility

        Prints a detailed PASS/FAIL report. Returns True if all checks pass.
        """
        # Read per-axis limits from hw_gradients.csv
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', 'configs', 'hw_gradients.csv')
        hw_limits = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        try:
                            hw_limits[row[0].strip()] = float(row[1])
                        except ValueError:
                            pass  # skip non-numeric rows (e.g., "GPA model,None")
        except Exception as e:
            print("WARNING: Could not read hw_gradients.csv: %s" % e)
            return True

        gx_max = abs(hw_limits.get('Gx max (mT/m)', 46.2))  # mT/m
        gy_max = abs(hw_limits.get('Gy max (mT/m)', 68.25))
        gz_max = abs(hw_limits.get('Gz max (mT/m)', 66.0))
        max_slew = hw_limits.get('Max slew rate (mT/m/ms)', 80.0)  # mT/m/ms
        grad_rise_time_us = hw_limits.get('Gradient rise time (us)', 500.0)
        grad_rise_time = grad_rise_time_us * 1e-6  # s

        per_axis_max = {'x': gx_max, 'y': gy_max, 'z': gz_max}

        # Map sequence axes (rd, ph, sl) to physical axes (x, y, z) via axesOrientation
        axes_map = {0: 'x', 1: 'y', 2: 'z'}
        rd_axis = axes_map[self.axesOrientation[0]]
        ph_axis = axes_map[self.axesOrientation[1]]
        sl_axis = axes_map[self.axesOrientation[2]]

        # Gradient amplitudes in T/m (from mapVals, set in sequenceRun Step 3)
        rd_amp = abs(self.mapVals.get('rd_grad_amplitude', 0))
        rd_deph_amp = abs(self.mapVals.get('rd_deph_amplitude', 0))
        ph_amp = abs(self.mapVals.get('ph_grad_amplitude', 0))
        sl_amp = abs(self.mapVals.get('sl_grad_amplitude', 0))

        # Maximum amplitude per physical axis (in mT/m)
        rd_max_mT = max(rd_amp, rd_deph_amp) * 1e3
        ph_max_mT = ph_amp * 1e3
        sl_max_mT = sl_amp * 1e3

        all_pass = True
        print("\n" + "=" * 60)
        print("  HARDWARE VALIDATION REPORT")
        print("=" * 60)

        checks = [
            ('Readout (%s)' % rd_axis, rd_max_mT, per_axis_max[rd_axis]),
            ('Phase (%s)' % ph_axis, ph_max_mT, per_axis_max[ph_axis]),
            ('Slice (%s)' % sl_axis, sl_max_mT, per_axis_max[sl_axis]),
        ]

        for label, amp, limit in checks:
            margin = (1 - amp / limit) * 100 if limit > 0 else 100
            status = 'PASS' if amp <= limit else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            print("  %-18s %6.1f / %6.1f mT/m  [%s, margin %.0f%%]" %
                  (label, amp, limit, status, margin))

        # Slew rate check per axis
        print("  ---")
        for label, amp_mT, limit in checks:
            slew = amp_mT / (grad_rise_time * 1e3)  # mT/m / ms
            margin = (1 - slew / max_slew) * 100 if max_slew > 0 else 100
            status = 'PASS' if slew <= max_slew else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            print("  Slew %-13s %6.1f / %6.1f mT/m/ms [%s, margin %.0f%%]" %
                  (label.split('(')[1], slew, max_slew, status, margin))

        # RF amplitude check
        print("  ---")
        rf_ex_amp = self.mapVals.get('rf_ex_amp', 0)
        rf_re_amp = self.mapVals.get('rf_re_amp', 0)
        rf_status = 'PASS' if rf_ex_amp <= 1 and rf_re_amp <= 1 else 'FAIL'
        if rf_status == 'FAIL':
            all_pass = False
        print("  RF excitation     %6.3f / 1.000      [%s]" % (rf_ex_amp, rf_status))
        print("  RF refocusing     %6.3f / 1.000      [%s]" % (rf_re_amp, rf_status))

        print("  ---")
        result = "ALL CHECKS PASSED" if all_pass else "*** VALIDATION FAILED ***"
        print("  %s" % result)
        print("=" * 60 + "\n")

        return all_pass

    def computeVFATrain(self, etl):
        """
        Compute the flip angle schedule for the refocusing echo train.

        Supports four modes:
        - 'Constant': All echoes use rfReFA.
        - 'ExponentialDecay': FA decays exponentially from rfReFA to rfReFAFinal.
        - 'LinearDecay': FA decays linearly from rfReFA to rfReFAFinal.
        - 'ConstantThenDecay': Constant rfReFA for vfaConstantEchoes, then exponential decay.

        Args:
            etl: Echo train length.

        Returns:
            np.ndarray: Flip angles in degrees of length etl.
        """
        fa_initial = self.rfReFA  # degrees
        fa_final = self.rfReFAFinal  # degrees
        mode = self.vfaMode
        n_const = self.vfaConstantEchoes

        if mode == 'Constant':
            fa_train = np.full(etl, fa_initial, dtype=float)

        elif mode == 'LinearDecay':
            if etl == 1:
                fa_train = np.array([fa_initial])
            else:
                fa_train = np.linspace(fa_initial, fa_final, etl)

        elif mode == 'ExponentialDecay':
            if etl == 1:
                fa_train = np.array([fa_initial])
            else:
                # Decay constant: at last echo, FA reaches fa_final
                # FA(n) = fa_final + (fa_initial - fa_final) * exp(-n / tau)
                # At n=etl-1: FA = fa_final => tau chosen so decay is smooth
                tau = (etl - 1) / 3.0  # ~95% decay over the train
                n = np.arange(etl, dtype=float)
                fa_train = fa_final + (fa_initial - fa_final) * np.exp(-n / tau)

        elif mode == 'ConstantThenDecay':
            fa_train = np.full(etl, fa_initial, dtype=float)
            n_decay = etl - n_const
            if n_decay > 1:
                tau = (n_decay - 1) / 3.0
                n = np.arange(n_decay, dtype=float)
                fa_train[n_const:] = fa_final + (fa_initial - fa_final) * np.exp(-n / tau)
            elif n_decay == 1:
                fa_train[n_const:] = fa_final

        elif mode == 'LinearIncreaseThenConstant':
            # Linear ramp from rfReFAFinal up to rfReFA over vfaConstantEchoes,
            # then constant rfReFA for the rest of the train.
            fa_train = np.full(etl, fa_initial, dtype=float)
            n_ramp = min(n_const, etl)
            if n_ramp > 1:
                fa_train[:n_ramp] = np.linspace(fa_final, fa_initial, n_ramp)
            elif n_ramp == 1:
                fa_train[0] = fa_final

        elif mode == 'ExponentialIncrease':
            # Exponentially converges from rfReFAFinal toward rfReFA.
            # FA(n) = fa_initial - (fa_initial - fa_final) * exp(-n / tau)
            # tau chosen so ~95% convergence over the train.
            if etl == 1:
                fa_train = np.array([fa_final])
            else:
                tau = (etl - 1) / 3.0
                n = np.arange(etl, dtype=float)
                fa_train = fa_initial - (fa_initial - fa_final) * np.exp(-n / tau)

        else:
            print(f"WARNING: Unknown VFA mode '{mode}', falling back to Constant.")
            fa_train = np.full(etl, fa_initial, dtype=float)

        # Clip to physical limits
        fa_train = np.clip(fa_train, 1.0, 180.0)
        return fa_train

    def _smooth_train_2opt(self, pts, _ph_s, _sl_s):
        """
        Smooth within-train trajectory using multi-start greedy NN + 2-opt.

        Uses a lexicographic bottleneck objective:
        1. Primary: minimize the maximum single step (worst gradient transient)
        2. Secondary: minimize total squared path distance

        Index 0 (center-most point) is always pinned.

        Args:
            pts: List of (ph, sl, dist) tuples for one echo train.
            _ph_s: Phase scaling factor for distance metric.
            _sl_s: Slice scaling factor for distance metric.

        Returns:
            list: Reordered list of (ph, sl, dist) tuples.
        """
        n = len(pts)
        if n <= 2:
            return sorted(pts, key=lambda x: x[2])

        pts = sorted(pts, key=lambda x: x[2])

        def _dsq(a, b):
            return ((a[0] - b[0]) * _ph_s) ** 2 + ((a[1] - b[1]) * _sl_s) ** 2

        def _path_cost(path):
            """Returns (max_step_sq, total_sq) for lexicographic comparison."""
            max_sq = 0.0
            total_sq = 0.0
            for k in range(len(path) - 1):
                d = _dsq(path[k], path[k + 1])
                total_sq += d
                if d > max_sq:
                    max_sq = d
            return (max_sq, total_sq)

        def _greedy_nn(first, second):
            """Build path with pinned first point and chosen second point."""
            ordered = [pts[first], pts[second]]
            remaining = set(range(n)) - {first, second}
            while remaining:
                last = ordered[-1]
                best = min(remaining, key=lambda r: _dsq(last, pts[r]))
                ordered.append(pts[best])
                remaining.remove(best)
            return ordered

        # Phase 1: Multi-start greedy NN
        n_starts = min(8, n - 1)
        candidates = sorted(range(1, n), key=lambda r: _dsq(pts[0], pts[r]))[:n_starts]

        best_path = None
        best_cost = (float('inf'), float('inf'))

        for second in candidates:
            path = _greedy_nn(0, second)
            cost = _path_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path

        # Phase 2: 2-opt local search (index 0 pinned)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_path = best_path[:i] + best_path[i:j + 1][::-1] + best_path[j + 1:]
                    new_cost = _path_cost(new_path)
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True

        return best_path

    def computeVariableDensityOrdering(self, n_ph, n_sl, etl, mask=None):
        """
        Compute a 2D variable-density k-space ordering over the phase-slice plane.

        For 'Uniform' mode, falls back to the standard 1D getIndex ordering.
        For 'CenterOut' and 'EllipticalCenterOut', lines are distributed across
        echo trains in an interleaved fashion: each train covers a spread from
        center to periphery, with the first echo in every train acquiring the
        closest-to-center line assigned to that train.

        Args:
            n_ph: Number of phase encoding steps.
            n_sl: Number of slice encoding steps.
            etl: Echo train length.
            mask: Optional boolean array (n_ph, n_sl). If provided, only points
                  where mask is True are included in the ordering.

        Returns:
            list of tuples: Ordered list of (ph_idx, sl_idx) pairs defining
            acquisition order. Every consecutive group of `etl` entries forms
            one echo train, with echo index 0 being closest to k-space center
            within that group.
        """
        mode = self.densityMode

        if mode == 'Uniform':
            ind = self.getIndex(etl, n_ph, self.sweepMode)
            ordering = []
            for sl_idx in range(n_sl):
                for ph_idx_ordered in ind:
                    if mask is None or mask[ph_idx_ordered, sl_idx]:
                        ordering.append((ph_idx_ordered, sl_idx))
            return ordering

        if mode == 'LinearCenterOut':
            # Linear ordering optimized for any mask:
            #   - Echo index within a train is strictly center-out in k_phase:
            #     echo e is the e-th closest to k_phase center among the train's
            #     echoes (global ranking across all acquired points).
            #   - Successive trains have monotonically increasing mean k_slice.
            #   - Works for any mask (fully sampled or undersampled), covering
            #     every acquired point exactly once.
            #
            # Algorithm:
            #   1. Collect all acquired (ph, sl) points.
            #   2. Sort globally by |ph - ph_center| ascending (ties: smaller
            #      |sl - sl_center| first, then ph then sl for determinism).
            #      This gives a center-out ranking.
            #   3. Form T = ceil(N / etl) trains by splitting the sorted list
            #      into etl "echo buckets" of T points each: bucket 0 holds
            #      the T most-central-k_phase points, bucket 1 the next T, etc.
            #   4. Within each echo bucket, sort by k_slice ascending and
            #      assign the t-th point to train t. This guarantees the mean
            #      k_slice of train t is monotonically non-decreasing.
            #   5. Emit trains 0..T-1, each listing echoes from bucket 0, 1, ...
            ph_center = (n_ph - 1) / 2.0
            sl_center = (n_sl - 1) / 2.0

            pts = []
            for sl_idx in range(n_sl):
                for ph_idx in range(n_ph):
                    if mask is not None and not mask[ph_idx, sl_idx]:
                        continue
                    pts.append((ph_idx, sl_idx))
            if not pts:
                return []

            pts.sort(key=lambda x: (abs(x[0] - ph_center),
                                    abs(x[1] - sl_center),
                                    x[0], x[1]))

            n_total = len(pts)
            n_trains = int(np.ceil(n_total / etl))

            # Split into echo buckets of up to n_trains points each.
            buckets = []
            for e in range(etl):
                start = e * n_trains
                if start >= n_total:
                    break
                end = min(start + n_trains, n_total)
                bucket = pts[start:end]
                # Sort within the bucket by k_slice ascending (ties: k_phase)
                bucket.sort(key=lambda x: (x[1], x[0]))
                buckets.append(bucket)

            # Emit trains: train t picks the t-th point from each bucket.
            ordering = []
            for t in range(n_trains):
                for bucket in buckets:
                    if t < len(bucket):
                        ordering.append(bucket[t])
            return ordering

        # Build all (ph, sl) pairs with their distance to center
        ph_center = (n_ph - 1) / 2.0
        sl_center = (n_sl - 1) / 2.0

        pairs = []
        for sl_idx in range(n_sl):
            for ph_idx in range(n_ph):
                if mask is not None and not mask[ph_idx, sl_idx]:
                    continue
                if mode == 'EllipticalCenterOut':
                    d_ph = (ph_idx - ph_center) / max(ph_center, 1.0)
                    d_sl = (sl_idx - sl_center) / max(sl_center, 1.0)
                else:  # CenterOut
                    d_ph = ph_idx - ph_center
                    d_sl = sl_idx - sl_center
                dist = np.sqrt(d_ph**2 + d_sl**2)
                pairs.append((ph_idx, sl_idx, dist))

        # Sort all lines by distance (center first)
        pairs.sort(key=lambda x: x[2])

        n_total = len(pairs)
        n_trains = int(np.ceil(n_total / etl))

        # Interleave into trains: distribute lines round-robin across trains,
        # so each train gets a spread from center to periphery.
        # Line 0 (closest) → train 0, line 1 → train 1, ..., line n_trains → train 0, etc.
        trains = [[] for _ in range(n_trains)]
        for i, (ph, sl, dist) in enumerate(pairs):
            trains[i % n_trains].append((ph, sl, dist))

        # Distance metric scaling
        if mode == 'EllipticalCenterOut':
            _ph_s = 1.0 / max(ph_center, 1.0)
            _sl_s = 1.0 / max(sl_center, 1.0)
        else:
            _ph_s = 1.0
            _sl_s = 1.0

        if self.smoothTrajectory >= 2:
            # --- Step 1: Smooth within-train using NN + 2-opt ---
            for t in range(n_trains):
                trains[t] = self._smooth_train_2opt(trains[t], _ph_s, _sl_s)

        elif self.smoothTrajectory == 1:
            # --- Step 1: Smooth within-train using greedy NN ---
            # Greedy nearest-neighbor path starting from center-most point.
            # Preserves center-out start, minimizes jumps between consecutive echoes.
            for t in range(n_trains):
                if len(trains[t]) <= 2:
                    trains[t].sort(key=lambda x: x[2])
                    continue
                pts = trains[t]
                pts.sort(key=lambda x: x[2])  # start from center
                ordered = [pts[0]]
                remaining = set(range(1, len(pts)))
                for _ in range(len(pts) - 1):
                    last_ph, last_sl, _ = ordered[-1]
                    best_idx = None
                    best_d = float('inf')
                    for r in remaining:
                        d = ((pts[r][0] - last_ph) * _ph_s)**2 + \
                            ((pts[r][1] - last_sl) * _sl_s)**2
                        if d < best_d:
                            best_d = d
                            best_idx = r
                    ordered.append(pts[best_idx])
                    remaining.remove(best_idx)
                trains[t] = ordered

        else:
            # Simple distance-sorted ordering (original behavior)
            for train in trains:
                train.sort(key=lambda x: x[2])

        # --- Step 2: Order trains to minimize inter-train jumps ---
        # Shared by modes 1 and 2. Greedy nearest-neighbor on train sequence.
        if self.smoothTrajectory >= 1 and n_trains > 2:
            def _train_cost(t_a, t_b):
                cost = 0.0
                for e in range(min(len(trains[t_a]), len(trains[t_b]))):
                    dph = (trains[t_a][e][0] - trains[t_b][e][0]) * _ph_s
                    dsl = (trains[t_a][e][1] - trains[t_b][e][1]) * _sl_s
                    cost += dph * dph + dsl * dsl
                return cost

            remaining_t = set(range(n_trains))
            first = min(range(n_trains), key=lambda t: trains[t][0][2])
            train_order = [first]
            remaining_t.remove(first)

            while remaining_t:
                last = train_order[-1]
                best_t = min(remaining_t, key=lambda t: _train_cost(last, t))
                train_order.append(best_t)
                remaining_t.remove(best_t)

            trains = [trains[t] for t in train_order]

        # Flatten: train 0 echoes, train 1 echoes, ...
        ordering = []
        for train in trains:
            for (ph, sl, _) in train:
                ordering.append((ph, sl))

        return ordering

    def generateUndersamplingMask(self, n_ph, n_sl):
        """
        Generate a variable-density undersampling mask for the phase-slice plane.

        When accelerationFactor > 1, creates a mask that samples k-space center
        more densely than the periphery. A fully-sampled calibration region is
        always preserved at the center.

        The undersamplingAxis parameter controls which axes are undersampled:
        - 'both' (default): 2D variable-density pattern across phase and slice
        - 'phase': undersample along phase only (all slices fully acquired)
        - 'slice': undersample along slice only (all phase steps fully acquired)

        Args:
            n_ph: Number of phase encoding steps.
            n_sl: Number of slice encoding steps.

        Returns:
            np.ndarray: Boolean mask of shape (n_ph, n_sl). True = acquired.
        """
        accel = self.accelerationFactor
        us_type = self.undersamplingType
        calib = self.calibrationSize
        us_axis = self.undersamplingAxis if self.undersamplingAxis else 'both'

        mask = np.ones((n_ph, n_sl), dtype=bool)

        if accel <= 1.0 or us_type == 'None':
            self.mapVals['undersampling_mask'] = mask
            return mask

        if us_axis == 'phase':
            # 1D undersampling along phase, all slices acquired
            mask_1d = self._generate1DMask(n_ph, accel, us_type, calib)
            mask = np.tile(mask_1d[:, np.newaxis], (1, n_sl))

        elif us_axis == 'slice':
            # 1D undersampling along slice, all phase steps acquired
            mask_1d = self._generate1DMask(n_sl, accel, us_type, calib)
            mask = np.tile(mask_1d[np.newaxis, :], (n_ph, 1))

        else:  # 'both'
            target_samples = int(np.ceil(n_ph * n_sl / accel))

            ph_center = n_ph // 2
            sl_center = n_sl // 2
            calib_ph_half = min(calib // 2, n_ph // 2)
            calib_sl_half = min(calib // 2, n_sl // 2)

            calib_mask = np.zeros((n_ph, n_sl), dtype=bool)
            calib_mask[ph_center - calib_ph_half:ph_center + calib_ph_half,
                       sl_center - calib_sl_half:sl_center + calib_sl_half] = True

            n_calib = int(np.sum(calib_mask))
            n_random = max(target_samples - n_calib, 0)

            elliptical = (self.densityMode == 'EllipticalCenterOut')

            if us_type == 'PoissonDisk':
                mask = self._poissonDiskMask(n_ph, n_sl, n_random, calib_mask, elliptical)
            elif us_type == 'GaussianRandom':
                mask = self._gaussianRandomMask(n_ph, n_sl, n_random, calib_mask, elliptical)
            else:
                print(f"WARNING: Unknown undersampling type '{us_type}', using full sampling.")

            mask |= calib_mask

        print(f"Undersampling ({us_axis}): {np.sum(mask)}/{n_ph*n_sl} lines "
              f"(acceleration: {n_ph*n_sl/np.sum(mask):.2f}x)")

        self.mapVals['undersampling_mask'] = mask
        return mask

    def _generate1DMask(self, n, accel, us_type, calib):
        """
        Generate a 1D variable-density undersampling mask along a single axis.

        Args:
            n: Number of encoding steps along this axis.
            accel: Acceleration factor.
            us_type: 'PoissonDisk' or 'GaussianRandom'.
            calib: Calibration region size.

        Returns:
            np.ndarray: Boolean mask of length n.
        """
        mask = np.zeros(n, dtype=bool)
        target = int(np.ceil(n / accel))
        center = n // 2
        calib_half = min(calib // 2, n // 2)

        # Always sample calibration region
        mask[center - calib_half:center + calib_half] = True
        n_calib = int(np.sum(mask))
        n_extra = max(target - n_calib, 0)

        # Variable-density probability for remaining points
        non_calib = np.where(~mask)[0]
        if len(non_calib) == 0 or n_extra == 0:
            return mask

        np.random.seed(42)
        center_f = (n - 1) / 2.0

        if us_type == 'GaussianRandom':
            sigma = n / 4.0
            probs = np.exp(-((non_calib - center_f) ** 2) / (2 * sigma ** 2))
            probs /= probs.sum()
            n_select = min(n_extra, len(non_calib))
            selected = np.random.choice(len(non_calib), size=n_select, replace=False, p=probs)
            mask[non_calib[selected]] = True
        else:  # PoissonDisk — concentric variable MD, exponential growth
            max_d = abs(center_f)
            smd_1d = 0.5  # smallest MD at center

            # Binary search for LMD (largest MD at periphery)
            def _count_1d(lmd_1d):
                count = 0
                for pos in non_calib:
                    d_norm = abs(pos - center_f) / max(max_d, 1)
                    md = smd_1d * np.exp(d_norm * np.log(max(lmd_1d / smd_1d, 1.01)))
                    count += 1.0 / max(md, 0.5)
                return int(count / max(1.0 / smd_1d, 1))

            lmd_lo, lmd_hi = smd_1d + 0.1, max(max_d * 0.5, 5.0)
            for _ in range(30):
                lmd_mid = (lmd_lo + lmd_hi) / 2
                if _count_1d(lmd_mid) > n_extra:
                    lmd_lo = lmd_mid
                else:
                    lmd_hi = lmd_mid
            lmd_1d = (lmd_lo + lmd_hi) / 2

            # Center-biased candidate order
            probs = 1.0 / (1.0 + 2.0 * np.abs(non_calib - center_f) / max(max_d, 1))
            probs /= probs.sum()
            candidate_order = np.random.choice(len(non_calib), size=len(non_calib),
                                               replace=False, p=probs)
            selected_pos = []
            for idx in candidate_order:
                if len(selected_pos) >= n_extra:
                    break
                pos = non_calib[idx]
                d_norm = abs(pos - center_f) / max(max_d, 1)
                md = smd_1d * np.exp(d_norm * np.log(max(lmd_1d / smd_1d, 1.01)))
                if any(abs(pos - sp) < md for sp in selected_pos):
                    continue
                selected_pos.append(pos)
                mask[pos] = True
            # Fill remainder if needed
            if len(selected_pos) < n_extra:
                remaining = [non_calib[i] for i in range(len(non_calib)) if not mask[non_calib[i]]]
                np.random.shuffle(remaining)
                for pos in remaining[:n_extra - len(selected_pos)]:
                    mask[pos] = True

        return mask

    @staticmethod
    def _poissonDiskMask(n_ph, n_sl, n_samples, calib_mask, elliptical=False):
        """
        Variable-density Poisson-disk mask using 9 concentric elliptical shells.

        Each shell has a minimum distance (MD) that increases pseudo-exponentially
        from 1 (center, fully dense) to 9 (periphery, sparsest). The MDs are
        scaled by the grid dimensions so the exclusion zones are elliptical when
        elliptical=True, matching the k-space aspect ratio.

        The 9 shells divide normalized distance [0, 1] into equal bands.
        Shell i has MD_base = exp(ln(1) + i/8 * ln(9)) ≈ [1, 1.3, 1.7, 2.3, 3, 3.9, 5.2, 6.8, 9].
        The base MDs are then scaled by (n_ph/n_max, n_sl/n_max) so that
        MD=1 means 1-pixel spacing along the largest dimension.

        Args:
            n_ph, n_sl: Grid dimensions.
            n_samples: Target number of samples (excluding calibration).
            calib_mask: Boolean array marking the calibration region.
            elliptical: If True, shell boundaries and exclusion use
                        dimension-normalized elliptical distances.
        """
        mask = calib_mask.copy()
        ph_c = n_ph / 2.0
        sl_c = n_sl / 2.0
        n_max = max(n_ph, n_sl)

        if n_samples < 1:
            return mask

        # Scaling: MD is in pixels along the largest dimension.
        # For exclusion, scale each axis so MD=1 means 1 pixel along n_max.
        if elliptical:
            ph_excl = n_max / max(n_ph, 1)  # >= 1 for the shorter axis
            sl_excl = n_max / max(n_sl, 1)
        else:
            ph_excl = 1.0
            sl_excl = 1.0

        # Normalized distance to center (elliptical: scaled by each half-dimension)
        def d_norm(ph, sl):
            return np.sqrt(((ph - ph_c) / max(ph_c, 1))**2 +
                           ((sl - sl_c) / max(sl_c, 1))**2)

        # 9 shells with pseudo-exponential MD: MD_i = exp(i/8 * ln(9))
        n_shells = 9
        md_base = np.array([np.exp(i / (n_shells - 1) * np.log(9)) for i in range(n_shells)])
        # md_base ≈ [1.0, 1.32, 1.73, 2.28, 3.0, 3.95, 5.20, 6.84, 9.0]

        shell_edges = np.linspace(0, np.sqrt(2) + 0.01, n_shells + 1)  # covers full grid

        # Precompute candidates per shell
        shell_cands = [[] for _ in range(n_shells)]
        for ph in range(n_ph):
            for sl in range(n_sl):
                if not calib_mask[ph, sl]:
                    dn = d_norm(ph, sl)
                    for s in range(n_shells):
                        if shell_edges[s] <= dn < shell_edges[s + 1]:
                            shell_cands[s].append((ph, sl))
                            break

        # Estimate samples per shell to compute budgets
        shell_counts = np.array([len(sc) for sc in shell_cands], dtype=float)
        expected_density = 1.0 / (md_base ** 2)  # relative density per shell
        raw_budgets = shell_counts * expected_density
        if raw_budgets.sum() > 0:
            budgets = np.round(raw_budgets * n_samples / raw_budgets.sum()).astype(int)
        else:
            budgets = np.zeros(n_shells, dtype=int)
        # Ensure at least 1 sample per non-empty shell
        for s in range(n_shells):
            if shell_counts[s] > 0 and budgets[s] < 1:
                budgets[s] = 1

        # Dart-throwing per shell (center outward)
        np.random.seed(42)
        sel_ph = []
        sel_sl = []
        n_sel = 0

        for s in range(n_shells):
            md = md_base[s]
            budget = int(budgets[s])
            cands = shell_cands[s]
            if not cands or budget == 0:
                continue

            # Shuffle candidates within shell
            perm = np.random.permutation(len(cands))
            shell_sel = 0

            for idx in perm:
                if shell_sel >= budget or n_sel >= n_samples:
                    break
                ph, sl = cands[idx]

                # Check exclusion distance (in scaled pixel space)
                if len(sel_ph) > 0:
                    dd = np.sqrt(((np.array(sel_ph) - ph) * ph_excl)**2 +
                                 ((np.array(sel_sl) - sl) * sl_excl)**2)
                    if np.any(dd < md):
                        continue

                mask[ph, sl] = True
                sel_ph.append(ph)
                sel_sl.append(sl)
                n_sel += 1
                shell_sel += 1

        # Relaxed fill: if under target, add from periphery inward
        if n_sel < n_samples:
            remaining = []
            for s in range(n_shells - 1, -1, -1):
                for ph, sl in shell_cands[s]:
                    if not mask[ph, sl]:
                        remaining.append((ph, sl))
            np.random.shuffle(remaining)
            for ph, sl in remaining[:n_samples - n_sel]:
                mask[ph, sl] = True

        return mask

    @staticmethod
    def _gaussianRandomMask(n_ph, n_sl, n_samples, calib_mask, elliptical=False):
        """
        Generate a Gaussian random variable density sampling mask.

        When elliptical=True, sigma is scaled per dimension so iso-probability
        contours are elliptical. When False, sigma is isotropic (circular).
        """
        mask = calib_mask.copy()
        ph_center = n_ph / 2.0
        sl_center = n_sl / 2.0

        if elliptical:
            sigma_ph = n_ph / 4.0
            sigma_sl = n_sl / 4.0
        else:
            sigma = max(n_ph, n_sl) / 4.0
            sigma_ph = sigma
            sigma_sl = sigma

        probs = np.zeros((n_ph, n_sl))
        for ph in range(n_ph):
            for sl in range(n_sl):
                if not calib_mask[ph, sl]:
                    probs[ph, sl] = np.exp(
                        -((ph - ph_center)**2 / (2 * sigma_ph**2)
                          + (sl - sl_center)**2 / (2 * sigma_sl**2))
                    )

        # Flatten and normalize
        non_calib_indices = np.argwhere(~calib_mask)
        prob_vals = np.array([probs[idx[0], idx[1]] for idx in non_calib_indices])
        if prob_vals.sum() > 0:
            prob_vals /= prob_vals.sum()

        n_to_select = min(n_samples, len(non_calib_indices))
        np.random.seed(42)

        selected_flat = np.random.choice(
            len(non_calib_indices), size=n_to_select, replace=False, p=prob_vals
        )

        for flat_idx in selected_flat:
            ph, sl = non_calib_indices[flat_idx]
            mask[ph, sl] = True

        return mask

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        """
        Runs the RARE MRI pulse sequence.

        This method orchestrates the execution of the RARE sequence by performing several key steps:
        1. Define the interpreter (FloSeq/PSInterpreter) to convert the sequence description into scanner instructions.
        2. Set system properties using PyPulseq (`pp.Opts`), which define hardware capabilities such as maximum gradient strengths and slew rates.
        3. Perform any necessary calculations for the sequence, such as timing, RF amplitudes, and gradient strengths.
        4. Define the experiment to determine the true bandwidth by using `get_sampling_period()` with an experiment defined as a class property (`self.expt`).
        5. Define sequence blocks including RF and gradient pulses that form the building blocks of the MRI sequence.
        6. Implement the `initializeBatch` method to create dummy pulses for each new batch.
        7. Define and populate the `createBatches` method, which accounts for the number of acquired points to determine when a new batch is needed.
        8. Run the batches and return the resulting data. Oversampled data is stored in `self.mapVals['data_over']`, and decimated data in `self.mapVals['data_decimated']`.

        Parameters:
        - plotSeq (bool): If True, plots the pulse sequence.
        - demo (bool): If True, runs the sequence in demo mode with simulated hardware.
        - standalone (bool): If True, runs the sequence as a standalone operation.

        Returns:
        - result (bool): The result of running the sequence, including oversampled and decimated data.
        """

        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        print('RARE run...')

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(np.abs(hw.gFactor)) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=np.max(np.abs(hw.gFactor)) * 1e3,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # Set the fov
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Check for used axes
        axes_enable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axes_enable.append(0)
            else:
                axes_enable.append(1)
        self.mapVals['axes_enable'] = axes_enable

        # Miscellaneous
        self.freqOffset = self.freqOffset*1e6  # MHz
        resolution = self.fov/self.nPoints
        rf_ex_amp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rf_re_amp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rf_ex_amp'] = rf_ex_amp
        self.mapVals['rf_re_amp'] = rf_re_amp
        self.mapVals['resolution'] = resolution
        self.mapVals['grad_rise_time'] = hw.grad_rise_time
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
        if rf_ex_amp > 1 or rf_re_amp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        # Set timing to multiples of gradient raster time for consistency (use us to get more accuracy)
        self.echoSpacing = (self.echoSpacing * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.repetitionTime = (self.repetitionTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.rdGradTime = (self.rdGradTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.phGradTime = (self.phGradTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.rdDephTime = (self.rdDephTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time

        # Matrix size
        n_rd = self.nPoints[0] + 2 * self.add_rd_points
        n_ph = self.nPoints[1]
        n_sl = self.nPoints[2]

        # ETL if etl>n_ph
        if self.etl>n_ph:
            self.etl = n_ph

        # Miscellaneous
        n_rd_points_per_train = self.etl * n_rd

        # par_acq_lines in case par_acq_lines = 0
        par_acq_lines = int(int(self.nPoints[2]*self.parFourierFraction)-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = par_acq_lines

        # BW
        bw = self.nPoints[0] / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw  # us

        # Readout gradient time
        if self.rdGradTime<self.acqTime:
            self.rdGradTime = self.acqTime
            print("Readout gradient time set to %0.1f ms" % (self.rdGradTime * 1e3))
        self.mapVals['rdGradTime'] = self.rdGradTime * 1e3  # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime == 0 or self.phGradTime > self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*hw.grad_rise_time:
            self.phGradTime = self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*hw.grad_rise_time
            print("Phase and slice gradient time set to %0.1f ms" % (self.phGradTime * 1e3))
        self.mapVals['phGradTime'] = self.phGradTime*1e3  # ms

        # Automatic minimum echo spacing validation (after rdGradTime and phGradTime auto-adjustments)
        min_es = self.calculateMinEchoSpacing(self.rfReTime, self.rdGradTime, self.acqTime, self.phGradTime)
        self.mapVals['minEchoSpacing'] = min_es
        print("Minimum echo spacing: %0.3f ms" % (min_es * 1e3))
        if self.echoSpacing < min_es:
            print("WARNING: Echo spacing (%0.3f ms) below minimum (%0.3f ms). Auto-adjusting." %
                  (self.echoSpacing * 1e3, min_es * 1e3))
            self.echoSpacing = min_es

        print("Echo spacing: %0.3f ms" % (self.echoSpacing * 1e3))
        print("Repetition time: %0.3f ms" % (self.repetitionTime * 1e3))

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acqTime)*axes_enable[0] * self.rd_direction
        ph_grad_amplitude = n_ph/(2*hw.gammaB*self.fov[1]*(self.phGradTime+hw.grad_rise_time))*axes_enable[1]
        sl_grad_amplitude = n_sl/(2*hw.gammaB*self.fov[2]*(self.phGradTime+hw.grad_rise_time))*axes_enable[2]
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude
        self.mapVals['ph_grad_amplitude'] = ph_grad_amplitude
        self.mapVals['sl_grad_amplitude'] = sl_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = 0.5*rd_grad_amplitude*(hw.grad_rise_time+self.rdGradTime)/(hw.grad_rise_time+self.rdDephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude
        print("Max rd gradient amplitude: %0.1f mT/m" % (max(rd_grad_amplitude, rd_deph_amplitude) * 1e3))
        print("Max ph gradient amplitude: %0.1f mT/m" % (ph_grad_amplitude * 1e3))
        print("Max sl gradient amplitude: %0.1f mT/m" % (sl_grad_amplitude * 1e3))

        # Validate hardware limits
        if not self.validateHardwareLimits():
            print("ERROR: Hardware limits exceeded. Aborting sequence.")
            return 0

        # Phase and slice gradient vector (full k-space, before partial Fourier)
        ph_gradients_full = np.linspace(-ph_grad_amplitude, ph_grad_amplitude, num=n_ph, endpoint=False)
        sl_gradients_full = np.linspace(-sl_grad_amplitude, sl_grad_amplitude, num=n_sl, endpoint=False)

        # Now fix the number of slices to partially acquired k-space
        n_sl = (int(self.nPoints[2]/2)+par_acq_lines)*axes_enable[2]+(1-axes_enable[2])
        print("Number of acquired slices: %i" % n_sl)

        # Compute Variable Flip Angle train
        fa_train = self.computeVFATrain(self.etl)
        self.mapVals['fa_train'] = fa_train.tolist()
        if self.vfaMode != 'Constant':
            print("VFA mode: %s" % self.vfaMode)
            print("FA train (degrees): %s" % np.array2string(fa_train, precision=1, separator=', '))

        # Check VFA RF amplitudes
        for echo_idx, fa_deg in enumerate(fa_train):
            fa_amp = (fa_deg / 180 * np.pi) / (self.rfReTime * 1e6 * hw.b1Efficiency)
            if fa_amp > 1:
                print("ERROR: RF amplitude too high for echo %d (FA=%.1f°), try longer RF time." %
                      (echo_idx, fa_deg))
                return 0

        # Generate undersampling mask
        undersampling_mask = self.generateUndersamplingMask(n_ph, n_sl)
        n_acquired_lines = int(np.sum(undersampling_mask))

        # Compute variable density ordering on acquired points only
        vd_ordering = self.computeVariableDensityOrdering(
            n_ph, n_sl, self.etl, mask=undersampling_mask)
        print("Acquired lines: %d / %d" % (len(vd_ordering), n_ph * n_sl))

        self.mapVals['vd_ordering'] = vd_ordering
        self.mapVals['n_acquired_lines'] = len(vd_ordering)

        # Build normalized gradient arrays indexed by original ph/sl indices
        # (these are used directly by create_batches via the vd_ordering)
        ph_gradients_norm = ph_gradients_full.copy()
        sl_gradients_norm = sl_gradients_full.copy()
        if ph_grad_amplitude != 0:
            ph_gradients_norm /= ph_grad_amplitude
        if sl_grad_amplitude != 0:
            sl_gradients_norm /= sl_grad_amplitude

        # For backward compatibility, keep the old-style arrays too
        # (used by initialize_batch for dummy pulses)
        ind = self.getIndex(self.etl, n_ph, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        ph_gradients = ph_gradients_full[ind]
        self.mapVals['ph_gradients'] = ph_gradients.copy()
        self.mapVals['sl_gradients'] = sl_gradients_full.copy()

        # Normalize gradient list for dummy pulses
        if ph_grad_amplitude != 0:
            ph_gradients /= ph_grad_amplitude
        if sl_grad_amplitude != 0:
            sl_gradients_full /= sl_grad_amplitude

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        '''
        # Step 4: Define the experiment to get the true bandwidth
        # In this step, student need to get the real bandwidth used in the experiment. To get this bandwidth, an
        # experiment must be defined and the sampling period should be obtained using get_sampling_period()
        # Note: experiment must be passed as a class property named self.expt
        '''

        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                      rx_t=sampling_period,  # us
                                      init_gpa=False,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      auto_leds=True,
                                      oversampling_factor=self.oversampling_factor)
            sampling_period = self.expt.get_sampling_period() # us
            bw = 1 / sampling_period  # MHz
            sampling_time = sampling_period * n_rd * 1e-6  # s
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            self.expt.__del__()
        else:
            sampling_time = sampling_period * n_rd * 1e-6  # s
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period
        self.mapVals['sampling_time_s'] = sampling_time

        '''
        # Step 5: Define sequence blocks.
        # In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.
        '''

        # First delay, sequence will start after 1 repetition time, this ensure gradient and ADC latency is not an issue.
        if self.inversionTime==0 and self.preExTime==0:
            delay = self.repetitionTime - self.rfExTime / 2 - system.rf_dead_time
        elif self.inversionTime>0 and self.preExTime==0:
            delay = self.repetitionTime - self.inversionTime - self.rfReTime / 2 - system.rf_dead_time
        elif self.inversionTime==0 and self.preExTime>0:
            delay = self.repetitionTime - self.preExTime - self.rfExTime / 2 - system.rf_dead_time
        else:
            delay = self.repetitionTime - self.preExTime - self.inversionTime - self.rfExTime / 2 - system.rf_dead_time
        delay_first = pp.make_delay(delay)

        # ADC to get noise
        delay = 100e-6
        block_adc_noise = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Pre-excitation pulse
        if self.preExTime>0:
            flip_pre = self.rfExFA * np.pi / 180
            delay = 0
            block_rf_pre_excitation = pp.make_block_pulse(
                flip_angle=flip_pre,
                system=system,
                duration=self.rfExTime,
                phase_offset=0.0,
                delay=0,
            )
            if self.inversionTime==0:
                delay = self.preExTime
            else:
                delay = self.rfExTime / 2 - self.rfReTime / 2 + self.preExTime
            delay_pre_excitation = pp.make_delay(delay)

        # Inversion pulse
        if self.inversionTime>0:
            flip_inv = self.rfReFA * np.pi / 180
            block_rf_inversion = pp.make_block_pulse(
                flip_angle=flip_inv,
                system=system,
                duration=self.rfReTime,
                phase_offset=0.0,
                delay=0,
            )
            delay = self.rfReTime / 2 - self.rfExTime / 2 + self.inversionTime
            delay_inversion = pp.make_delay(delay)

        # Excitation pulse
        flip_ex = self.rfExFA * np.pi / 180
        block_rf_excitation = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=0.0,
            use = 'excitation'
        )

        # De-phasing gradient
        delay = system.rf_dead_time + self.rfExTime / 2 + ((self.rfExTime / 2* 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time  # multiple of grad_raster_time
        block_gr_rd_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=system,
            amplitude=rd_deph_amplitude * hw.gammaB,
            flat_time=self.rdDephTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay to re-focusing pulse
        delay_preph = pp.make_delay(self.echoSpacing / 2 + self.rfExTime / 2 - self.rfReTime / 2)

        # Refocusing pulses — one per echo for Variable Flip Angle support
        block_rf_refocusing_list = []
        for echo_idx in range(self.etl):
            flip_re = fa_train[echo_idx] * np.pi / 180
            block_rf_refocusing_list.append(pp.make_block_pulse(
                flip_angle=flip_re,
                system=system,
                duration=self.rfReTime,
                phase_offset=np.pi / 2,
                delay=0,
                use='refocusing'
            ))
        # Keep a single block for dummy pulses (uses initial FA)
        block_rf_refocusing = block_rf_refocusing_list[0]

        # Delay to next refocusing pulse
        delay_reph = pp.make_delay(self.echoSpacing)

        # Phase gradient de-phasing
        delay = system.rf_dead_time + self.rfReTime / 2 + ((self.rfReTime / 2 * 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient de-phasing
        delay = system.rf_dead_time + self.rfReTime / 2 + ((self.rfReTime / 2 * 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Readout gradient
        delay = system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - self.rdGradTime / 2 - \
                hw.grad_rise_time
        block_gr_rd_reph = pp.make_trapezoid(
            channel=rd_channel,
            system=system,
            amplitude=rd_grad_amplitude * hw.gammaB,
            flat_time=self.rdGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # ADC to get the signal
        delay = system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - sampling_time / 2
        block_adc_signal = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Phase gradient re-phasing
        delay = (system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 +
                 ((self.nPoints[0] / 2 / bw) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time)
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient re-phasing
        delay = (system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 +
                 ((self.nPoints[0] / 2 / bw) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time)
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay TR
        delay = self.repetitionTime + self.rfReTime / 2 - self.rfExTime / 2 - (self.etl + 0.5) * self.echoSpacing - \
            self.inversionTime - self.preExTime
        if self.inversionTime > 0 and self.preExTime == 0:
            delay -= self.rfExTime / 2
        delay_tr = pp.make_delay(delay)

        '''
        # Step 6: Define your initializeBatch according to your sequence.
        # In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        # each new batch.
        '''

        def initialize_batch():
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - `batch` (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - `n_rd_points` (int): Total number of readout points in the batch.
                - `n_adc` (int): Total number of ADC acquisitions in the batch.

            Workflow:
            ---------
            1. **Create PyPulseq Sequence Object**:
                - Instantiates a new PyPulseq sequence object (`pp.Sequence`) and initializes counters for
                  readout points (`n_rd_points`) and ADC events (`n_adc`).

            2. **Set Gradients to Zero**:
                - Initializes slice and phase gradients (`gr_ph_deph`, `gr_sl_deph`, `gr_ph_reph`, `gr_sl_reph`) to zero
                  by scaling predefined gradient blocks with a factor of 0.

            3. **Add Initial Delay and Noise Measurement**:
                - Adds an initial delay block (`delay_first`) and a noise measurement ADC block (`block_adc_noise`)
                  to the sequence.

            4. **Generate Dummy Pulses**:
                - Creates a specified number of dummy pulses (`self.dummyPulses`) to prepare the system for data acquisition:
                    - **Pre-excitation Pulse**:
                        - If `self.preExTime > 0`, adds a pre-excitation pulse with a readout pre-phasing gradient.
                    - **Inversion Pulse**:
                        - If `self.inversionTime > 0`, adds an inversion pulse with a scaled readout pre-phasing gradient.
                    - **Excitation Pulse**:
                        - Adds an excitation pulse followed by a readout de-phasing gradient (`block_gr_rd_preph`).

                - For each dummy pulse:
                    - **Echo Train**:
                        - For the last dummy pulse, appends an echo train that includes:
                            - A refocusing pulse.
                            - Gradients for readout re-phasing, phase de-phasing, and slice de-phasing.
                            - ADC signal acquisition block (`block_adc_signal`).
                            - Gradients for phase and slice re-phasing.
                        - For other dummy pulses, excludes the ADC signal acquisition.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) to separate repetitions.

            5. **Return Results**:
                - Returns the configured sequence (`batch`), total readout points (`n_rd_points`), and number of ADC events (`n_adc`).

            """
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            # Add first delay and first noise measurement
            batch.add_block(delay_first, block_adc_noise)
            n_rd_points += n_rd
            n_adc += 1

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime>0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                            gr_rd_preex,
                                            delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime>0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batch.add_block(block_rf_inversion,
                                            gr_rd_inv,
                                            delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batch.add_block(block_gr_rd_preph,
                                        block_rf_excitation,
                                        delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses-1:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                    else:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)

                # Add time delay to next repetition
                batch.add_block(delay_tr)

            return batch, n_rd_points, n_adc

        def initialize_batch_0():
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - `batch` (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - `n_rd_points` (int): Total number of readout points in the batch.
                - `n_adc` (int): Total number of ADC acquisitions in the batch.

            Workflow:
            ---------
            1. **Create PyPulseq Sequence Object**:
                - Instantiates a new PyPulseq sequence object (`pp.Sequence`) and initializes counters for
                readout points (`n_rd_points`) and ADC events (`n_adc`).

            2. **Set Gradients to Zero**:
                - Initializes slice and phase gradients (`gr_ph_deph`, `gr_sl_deph`, `gr_ph_reph`, `gr_sl_reph`) to zero
                by scaling predefined gradient blocks with a factor of 0.

            3. **Add Initial Delay and Noise Measurement**:
                - Adds an initial delay block (`delay_first`) and a noise measurement ADC block (`block_adc_noise`)
                to the sequence.

            4. **Generate Dummy Pulses**:
                - Creates a specified number of dummy pulses (`self.dummyPulses`) to prepare the system for data acquisition:
                    - **Pre-excitation Pulse**:
                        - If `self.preExTime > 0`, adds a pre-excitation pulse with a readout pre-phasing gradient.
                    - **Inversion Pulse**:
                        - If `self.inversionTime > 0`, adds an inversion pulse with a scaled readout pre-phasing gradient.
                    - **Excitation Pulse**:
                        - Adds an excitation pulse followed by a readout de-phasing gradient (`block_gr_rd_preph`).

                - For each dummy pulse:
                    - **Echo Train**:
                        - For the last dummy pulse, appends an echo train that includes:
                            - A refocusing pulse.
                            - Gradients for readout re-phasing, phase de-phasing, and slice de-phasing.
                            - ADC signal acquisition block (`block_adc_signal`).
                            - Gradients for phase and slice re-phasing.
                        - For other dummy pulses, excludes the ADC signal acquisition.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) to separate repetitions.

            5. **Return Results**:
                - Returns the configured sequence (`batch`), total readout points (`n_rd_points`), and number of ADC events (`n_adc`).

            """
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            # Add first delay and first noise measurement
            for nNoise in range(self.nNoise):
                batch.add_block(delay_first, block_adc_noise)
                n_rd_points += n_rd
                n_adc += 1
            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime>0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                            gr_rd_preex,
                                            delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime>0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batch.add_block(block_rf_inversion,
                                            gr_rd_inv,
                                            delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batch.add_block(block_gr_rd_preph,
                                        block_rf_excitation,
                                        delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses-1:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                    else:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)

                # Add time delay to next repetition
                batch.add_block(delay_tr)

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def create_batches():
            """
            Creates and processes multiple batches of MRI sequence blocks.

            Uses variable density ordering (vd_ordering) to iterate over (ph, sl) pairs,
            and per-echo VFA refocusing pulses from block_rf_refocusing_list.

            Returns:
                tuple: (waveforms dict, n_rd_points_dict, n_adc total)
            """
            batches = {}
            waveforms = {}
            n_rd_points_dict = {}
            n_rd_points = 0
            seq_idx = 0
            n_adc = 0
            batch_num = "batch_0"

            # Group the VD ordering into ETL-sized echo trains
            line_idx = 0
            total_lines = len(vd_ordering)

            while line_idx < total_lines:
                # Determine how many lines this train will acquire
                train_lines = min(self.etl, total_lines - line_idx)
                n_rd_points_per_train_actual = train_lines * n_rd

                # Check if a new batch is needed
                if seq_idx == 0 or n_rd_points + n_rd_points_per_train_actual > hw.maxRdPoints:
                    if seq_idx > 0:
                        batches[batch_num].write(batch_num + ".seq")
                        waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
                        print(f"{batch_num}.seq ready!")

                    seq_idx += 1
                    n_rd_points_dict[batch_num] = n_rd_points
                    n_rd_points = 0
                    batch_num = f"batch_{seq_idx}"
                    print(batch_num)
                    if batch_num == "batch_1":
                        batches[batch_num], n_rd_points, n_adc_0 = initialize_batch_0()
                    else:
                        batches[batch_num], n_rd_points, n_adc_0 = initialize_batch()
                    n_adc += n_adc_0
                    print(f"Creating {batch_num}.seq...")

                # Pre-excitation pulse
                if self.preExTime > 0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=+1.0)
                    batches[batch_num].add_block(block_rf_pre_excitation,
                                                 gr_rd_preex,
                                                 delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime > 0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batches[batch_num].add_block(block_rf_inversion,
                                                 gr_rd_inv,
                                                 delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batches[batch_num].add_block(block_gr_rd_preph,
                                             block_rf_excitation,
                                             delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if line_idx >= total_lines:
                        # No more lines to acquire — play dummy echo (no ADC, zero gradients)
                        gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
                        gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
                        gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
                        gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)
                        batches[batch_num].add_block(block_rf_refocusing_list[echo],
                                                     block_gr_rd_reph,
                                                     gr_ph_deph,
                                                     gr_sl_deph,
                                                     delay_reph)
                        batches[batch_num].add_block(gr_ph_reph,
                                                     gr_sl_reph)
                        continue

                    ph_idx, sl_idx = vd_ordering[line_idx]

                    # Scale phase and slice gradients using normalized values
                    gr_ph_deph = pp.scale_grad(block_gr_ph_deph, ph_gradients_norm[ph_idx])
                    gr_sl_deph = pp.scale_grad(block_gr_sl_deph, sl_gradients_norm[sl_idx])
                    gr_ph_reph = pp.scale_grad(block_gr_ph_reph, -ph_gradients_norm[ph_idx])
                    gr_sl_reph = pp.scale_grad(block_gr_sl_reph, -sl_gradients_norm[sl_idx])

                    # Use per-echo VFA refocusing pulse
                    rf_refoc = block_rf_refocusing_list[echo]

                    # Determine if this echo should acquire data
                    acquire = False
                    if self.echoMode == 'All':
                        acquire = True
                    elif self.echoMode == 'Odd' and echo % 2 == 0:
                        acquire = True
                    elif self.echoMode == 'Even' and echo % 2 == 1:
                        acquire = True

                    if acquire:
                        batches[batch_num].add_block(rf_refoc,
                                                     block_gr_rd_reph,
                                                     gr_ph_deph,
                                                     gr_sl_deph,
                                                     block_adc_signal,
                                                     delay_reph)
                        batches[batch_num].add_block(gr_ph_reph,
                                                     gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                        line_idx += 1
                    else:
                        batches[batch_num].add_block(rf_refoc,
                                                     block_gr_rd_reph,
                                                     gr_ph_deph,
                                                     gr_sl_deph,
                                                     delay_reph)
                        batches[batch_num].add_block(gr_ph_reph,
                                                     gr_sl_reph)

                # Add time delay to next repetition
                batches[batch_num].add_block(delay_tr)

            # After final repetition, save and interpret the last batch
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired points in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = n_rd_points

            return waveforms, n_rd_points_dict, n_adc

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        '''
        waveforms, n_readouts, n_adc = create_batches()
        return self.runBatches(waveforms=waveforms,
                               n_readouts=n_readouts,
                               n_adc=n_adc,
                               frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               oversampling_factor=self.oversampling_factor,
                               decimation_factor=self.decimation_factor,
                               hardware=True,
                               )

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)

        # Get axes in strings
        axes = self.mapVals['axesOrientation']
        axesDict = {'x': 0, 'y': 1, 'z': 2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['', '', '']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1

        ## Tyger Reconstruction
        out_field = 'image3D_den'
        out_field_k = 'kSpace3D_den'
        result_Tyger = None
        if self.mapVals['axes_enable'] == [1,1,1] and self.tyger_denoising == 1:
            try:
                rawData_path = self.directory_mat + '/' + self.file_name+'.mat'
                imgTyger = tyger_denoising.denoisingTyger(rawData_path, out_field, out_field_k)
                imageTyger = np.abs(imgTyger[0])
                imageTyger = imageTyger/np.max(np.reshape(imageTyger,-1))*100

                ## Image plot
                # Tyger
                if self.mapVals['unlock_orientation'] == 0:
                    result_Tyger, _, _ = utils.fix_image_orientation(imageTyger, axes=self.axesOrientation)
                    result_Tyger['row'] = 0
                    result_Tyger['col'] = 1
                    result_Tyger['title'] = "Tyger"
                else:
                    result_Tyger = {'widget': 'image', 'data': imageTyger, 'xLabel': "%s" % axesStr[1],
                                    'yLabel': "%s" % axesStr[0], 'title': "Tyger", 'row': 0, 'col': 1}

            except Exception as e:
                print('Tyger reconstruction failed.')
                print(f'Error: {e}')

        ## Tyger Reconstruction
        if self.mapVals['axes_enable'] == [1, 1, 1] and self.tyger_recon == 1:
            if self.tyger_denoising == 1:
                input_field = out_field_k
            else:
                input_field =''
            print('Preparing Tyger enviroment...')
            rawData_path = self.directory_mat + '/' + self.file_name + '.mat'
            sign_rarepp = [-1, -1, -1, 1, 1, 1, 1, 1, tyger_conf.cp_batchsize_RARE]
            if self.recon_type == 'cp':
                output_field = 'imgTygerCP'
            # elif self.recon_type == 'art':
            #     output_field = 'imgTygerART'
            elif self.recon_type == 'artpk':
                output_field = 'imgTygerARTPK'
            elif self.recon_type == 'fft':
                output_field = 'imgTygerFFT'
            else:
                print('Reconstruction type not available in tyger. Reassigned to FFT.')
                var = self.recon_type == 'fft'
                output_field = 'imgTygerFFT'
            boFit_path = 'b0_maps/fits/' + self.boFit_file
            if self.tyger_denoising == 1:
                output_field = output_field + '_den'

            try:
                imgTyger = tyger_rare.reconTygerRARE(rawData_path, self.recon_type, boFit_path, sign_rarepp, output_field, input_field)
                imageTyger = np.abs(imgTyger[0])
                imageTyger = imageTyger / np.max(np.reshape(imageTyger, -1)) * 100

                ## Image plot
                # Tyger
                if self.unlock_orientation == 0:
                    result_Tyger, _, _ = utils.fix_image_orientation(imageTyger, axes=self.axesOrientation)
                    result_Tyger['row'] = 0
                    result_Tyger['col'] = 1
                    result_Tyger['title'] = "Tyger"
                else:
                    result_Tyger = {'widget': 'image', 'data': imageTyger, 'xLabel': "%s" % axesStr[1],
                                    'yLabel': "%s" % axesStr[0], 'title': "k-Space", 'row': 0, 'col': 0}

            except Exception as e:
                print('Tyger reconstruction failed.')
                print(f'Error: {e}')

        if result_Tyger is not None:
            self.output.append(result_Tyger)

        return self.output

    def save_ismrmrd(self):
        """
        Save the current instance's data in ISMRMRD format.

        This method saves the raw data, header information, and reconstructed images to an HDF5 file
        using the ISMRMRD (Image Storage and Reconstruction format for MR Data) format.

        Steps performed:
        1. Generate a timestamp-based filename and directory path for the output file.
        2. Initialize the ISMRMRD dataset with the generated path.
        3. Populate the header and write the XML header to the dataset. Informations can be added.
        4. Reshape the raw data matrix and iterate over scans, slices, and phases to write each acquisition. WARNING : RARE sequence follows ind order to fill the k-space.
        5. Set acquisition flags and properties.
        6. Append the acquisition data to the dataset.
        7. Reshape and save the reconstructed images.
        8. Close the dataset.

        Attribute:
        - self.data_full_mat (numpy.array): Full matrix of raw data to be reshaped and saved.

        Returns:
        None. It creates an HDF5 file with the ISMRMRD format.
        """

        directory_rmd = self.directory_rmd
        name = datetime.datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            self.raw_data_name = self.mapVals['seqName']
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)

        path= "%s/%s.h5" % (directory_rmd, file_name)

        dset = ismrmrd.Dataset(path, f'/dataset', True) # Create the dataset

        etl = self.mapVals['etl']
        axes_enable = self.mapVals['axes_enable']
        n_rd = self.nPoints[0]
        n_ph = self.nPoints[1]
        n_sl = (((self.nPoints[2] // 2) + self.mapVals['partialAcquisition']) * axes_enable[2] + (1 - axes_enable[2]))
        ind = self.getIndex(self.etl, n_ph, self.sweepMode)
        nRep = (n_ph//etl)*n_sl
        bw = self.mapVals['bw_MHz']

        axesOrientation = self.axesOrientation
        axesOrientation_list = axesOrientation.tolist()

        read_dir = [0, 0, 0]
        phase_dir = [0, 0, 0]
        slice_dir = [0, 0, 0]

        read_dir[axesOrientation_list.index(0)] = 1
        phase_dir[axesOrientation_list.index(1)] = 1
        slice_dir[axesOrientation_list.index(2)] = 1

        # Experimental Conditions field
        exp = ismrmrd.xsd.experimentalConditionsType()
        magneticFieldStrength = hw.larmorFreq * 1e6 / hw.gammaB
        exp.H1resonanceFrequency_Hz = hw.larmorFreq

        self.header.experimentalConditions = exp

        # Acquisition System Information field
        sys = ismrmrd.xsd.acquisitionSystemInformationType()
        sys.receiverChannels = 1
        self.header.acquisitionSystemInformation = sys


        # Encoding field can be filled if needed
        encoding = ismrmrd.xsd.encodingType()
        encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
        #encoding.trajectory =ismrmrd.xsd.trajectoryType[data.processing.trajectory.upper()]

        dset.write_xml_header(self.header.toXML()) # Write the header to the dataset

        new_data = np.zeros((n_ph * n_sl * self.nScans, n_rd + 2*self.add_rd_points))
        new_data = np.reshape(self.data_fullmat, (self.nScans, n_sl, n_ph, n_rd+ 2*self.add_rd_points))

        counter=0
        for scan in range(self.nScans):
            for slice_idx in range(n_sl):
                for phase_idx in range(n_ph):

                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, n_rd+2*self.add_rd_points))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)

                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (n_ph // etl))

                    acq.clearAllFlags()

                    if index_in_repetition == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)

                    if ind[phase_idx]== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif ind[phase_idx] == n_ph - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)

                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == n_sl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)

                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)

                    if scan == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                    elif scan == self.nScans-1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)


                    counter += 1

                    # +1 to start at 1 instead of 0
                    acq.idx.repetition = int(current_repetition + 1)
                    acq.idx.kspace_encode_step_1 = ind[phase_idx]+1 # phase
                    acq.idx.slice = slice_idx + 1
                    acq.idx.contrast = index_in_repetition + 1
                    acq.idx.average = scan + 1 # scan

                    acq.scan_counter = counter
                    acq.discard_pre = self.add_rd_points
                    acq.discard_post = self.add_rd_points
                    acq.sample_time_us = 1/bw
                    self.dfov = np.array(self.dfov)
                    acq.position = (ctypes.c_float * 3)(*self.dfov.flatten())


                    acq.read_dir = (ctypes.c_float * 3)(*read_dir)
                    acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                    acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)

                    dset.append_acquisition(acq) # Append the acquisition to the dataset


        image=self.mapVals['image3D']
        image_reshaped = np.reshape(image, (self.nPoints[::-1]))

        for slice_idx in range (n_sl): ## image3d does not have scan dimension

            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            img.transpose = False
            img.field_of_view = (ctypes.c_float * 3)(*(self.fov)*10) # mm

            img.position = (ctypes.c_float * 3)(*self.dfov)

            # img.data_type= 8 ## COMPLEX FLOAT
            img.image_type = 5 ## COMPLEX

            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)

            dset.append_image(f"image_raw", img) # Append the image to the dataset


        dset.close()


if __name__ == '__main__':
    seq = RarePyPulseqVD()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
