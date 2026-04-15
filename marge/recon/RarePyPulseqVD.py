import marge.configs.hw_config as hw
import numpy as np
import scipy as sp
from marge.marge_utils import utils


def RarePyPulseqVD(raw_data_path=None):
    """
    Reconstruction for the RarePyPulseqVD sequence.

    Handles variable-density k-space ordering and undersampling.
    Data lines are placed at their correct (ph, sl) positions using
    the vd_ordering list instead of the simple sweepOrder permutation.
    Unacquired k-space lines are zero-filled.

    Parameters:
        raw_data_path (str): Path to the .mat file.

    Returns:
        tuple: (output_dict, output, dicom_meta_data)
    """
    if raw_data_path is None:
        return None

    # Load .mat
    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}
    dicom_meta_data = {}

    # Print inputs
    try:
        keys = mat_data['input_keys']
        strings = mat_data['input_strings']
        string = ""
        print("****Inputs****")
        for ii, key in enumerate(keys):
            string = string + f"{str(strings[ii]).strip()}: {np.squeeze(mat_data[str(key).strip()])}, "
        print(string)
    except:
        pass
    print("****Outputs****")

    # Get data
    par_fourier_fraction = mat_data['parFourierFraction'].item()
    axes_orientation = np.squeeze(mat_data['axesOrientation'])
    fov = np.squeeze(mat_data['fov']) * 1e-2
    dfov = np.squeeze(mat_data['dfov']) * 1e-3
    fov = fov[axes_orientation]
    dfov = dfov[axes_orientation]
    etl = mat_data['etl'].item()
    n_scans = mat_data['nScans'].item()
    axes_enable = np.squeeze(mat_data['axes_enable'])
    data_decimated = np.squeeze(mat_data['data_decimated'])
    n_points = np.squeeze(mat_data['nPoints'])
    add_rd_points = mat_data['add_rd_points'].item()
    n_rd, n_ph, n_sl_full = n_points
    n_rd = n_rd + 2 * add_rd_points
    n_sl = (n_sl_full // 2 + mat_data['partialAcquisition'].item() * axes_enable[2] + (1 - axes_enable[2]))
    n_batches = mat_data['n_batches'].item()
    n_readouts = mat_data['n_readouts'][0]
    k_fill = mat_data['k_fill'].item()
    dummy_pulses = mat_data['dummyPulses'].item()
    rd_direction = mat_data['rd_direction'].item()
    n_noise = mat_data['nNoise'].item()

    # Get VD ordering and undersampling info
    try:
        vd_ordering = mat_data['vd_ordering']
        # vd_ordering is stored as Nx2 array of (ph_idx, sl_idx)
        if vd_ordering.ndim == 1:
            # Fallback: might be stored as flat list of tuples
            vd_ordering = np.reshape(vd_ordering, (-1, 2))
        vd_ordering = [(int(row[0]), int(row[1])) for row in vd_ordering]
        has_vd = True
        print(f"  VD ordering: {len(vd_ordering)} acquired lines")
    except (KeyError, Exception):
        has_vd = False
        vd_ordering = None
        print("  No VD ordering found, using sweepOrder fallback")

    try:
        undersampling_mask = np.squeeze(mat_data['undersampling_mask']).astype(bool)
        has_mask = True
    except (KeyError, Exception):
        has_mask = False
        undersampling_mask = np.ones((n_ph, n_sl), dtype=bool)

    # Fallback sweep order for non-VD data
    try:
        ind = np.squeeze(mat_data['sweepOrder'])
    except (KeyError, Exception):
        ind = np.arange(n_ph)

    # Number of acquired lines
    if has_vd:
        n_acquired = len(vd_ordering)
    else:
        n_acquired = n_ph * n_sl

    # Get noise data, dummy data and signal data
    data_noise = []
    data_dummy = []
    data_signal = []
    points_per_rd = n_rd
    points_per_train = points_per_rd * etl
    idx_0 = 0
    idx_1 = 0
    for batch in range(n_batches):
        n_rds = n_readouts[batch]
        for scan in range(n_scans):
            idx_1 += n_rds
            data_prov = data_decimated[idx_0:idx_1]
            if batch == 0:
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd * n_noise]), axis=0)
                if dummy_pulses > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[
                        points_per_rd * n_noise:points_per_rd * n_noise + points_per_train]), axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd * n_noise + points_per_train::]),
                                             axis=0)
            else:
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
                if dummy_pulses > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd + points_per_train]),
                                                axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd + points_per_train::]), axis=0)
            idx_0 = idx_1
        if batch == 0:
            n_readouts[batch] += -n_rd * n_noise - n_rd * etl
        else:
            n_readouts[batch] += -n_rd - n_rd * etl
    data_noise = np.reshape(data_noise, (-1, n_points[0] + add_rd_points * 2))
    data_noise = data_noise[:, add_rd_points: -add_rd_points]
    output_dict['data_noise'] = data_noise
    output_dict['data_dummy'] = data_dummy
    output_dict['data_signal'] = data_signal

    # Decimate data to get signal in desired bandwidth
    data_full = data_signal

    # Reorganize data_full across batches (same logic as RarePyPulseq)
    data_prov = np.zeros(shape=[n_scans, n_acquired * n_rd], dtype=complex)
    if n_batches == 2:
        data_full_a = data_full[0:sum(n_readouts[0:-1]) * n_scans]
        data_full_b = data_full[sum(n_readouts[0:-1]) * n_scans:]
        data_full_a = np.reshape(data_full_a, shape=(n_batches - 1, n_scans, -1, n_rd))
        data_full_b = np.reshape(data_full_b, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
            data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
            data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
    elif n_batches > 2:
        data_full_ini = data_full[0:n_readouts[0] * n_scans]
        data_full_a = data_full[
            n_readouts[0] * n_scans:n_readouts[0] * n_scans + n_readouts[1] * (n_batches - 2) * n_scans]
        data_full_b = data_full[n_readouts[0] * n_scans + n_readouts[1] * (n_batches - 2) * n_scans:]
        data_full_ini = np.reshape(data_full_ini, shape=(1, n_scans, -1, n_rd))
        data_full_a = np.reshape(data_full_a, shape=(n_batches - 2, n_scans, -1, n_rd))
        data_full_b = np.reshape(data_full_b, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_scan_ini = np.reshape(data_full_ini[:, scan, :, :], -1)
            data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
            data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
            data_prov[scan, :] = np.concatenate((data_scan_ini, data_scan_a, data_scan_b), axis=0)
    else:
        data_full = np.reshape(data_full, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
    data_full = np.reshape(data_prov, -1)

    # Place acquired data into full k-space grid using VD ordering
    if has_vd:
        # Data arrives as sequential readout lines following vd_ordering
        data_per_scan = np.reshape(data_full, (n_scans, n_acquired, n_rd))

        # Average scans
        data_avg = np.average(data_per_scan, axis=0)  # (n_acquired, n_rd)

        # Place into full k-space grid (zero-filled)
        kspace_full = np.zeros((n_sl, n_ph, n_rd), dtype=complex)
        for line_idx, (ph_idx, sl_idx) in enumerate(vd_ordering):
            if line_idx < data_avg.shape[0]:
                kspace_full[sl_idx, ph_idx, :] = data_avg[line_idx, :]

        # Find krd=0 index from center of k-space
        center_sl = int(n_points[2] / 2) if n_points[2] / 2 < n_sl else n_sl // 2
        center_ph = int(n_ph / 2)
        center_line = kspace_full[center_sl, center_ph, :]
        ind_krd_0 = np.argmax(np.abs(center_line))
        if ind_krd_0 < n_rd / 2 - add_rd_points or ind_krd_0 > n_rd / 2 + add_rd_points:
            ind_krd_0 = int(n_rd / 2)

        # Crop readout to nominal size
        data = kspace_full[:, :, ind_krd_0 - int(n_points[0] / 2):ind_krd_0 + int(n_points[0] / 2)]

        # Also store per-scan data for data_full output
        data_full_grid = np.zeros((n_scans, n_sl, n_ph, n_rd), dtype=complex)
        for scan in range(n_scans):
            for line_idx, (ph_idx, sl_idx) in enumerate(vd_ordering):
                if line_idx < data_per_scan.shape[1]:
                    data_full_grid[scan, sl_idx, ph_idx, :] = data_per_scan[scan, line_idx, :]
        data_full_cropped = data_full_grid[:, :, :,
                            ind_krd_0 - int(n_points[0] / 2):ind_krd_0 + int(n_points[0] / 2)]
        output_dict['data_full'] = data_full_cropped

    else:
        # Fallback: original sweepOrder-based reordering
        data_prov = np.reshape(data_full, shape=(n_scans, n_rd * n_ph * n_sl))
        data_avg = np.average(data_prov, axis=0)
        data_prov_3d = np.reshape(data_avg, shape=(n_sl, n_ph, n_rd))
        data_temp = np.zeros_like(data_prov_3d)
        for ii in range(n_ph):
            data_temp[:, ind[ii], :] = data_prov_3d[:, ii, :]
        data_prov_3d = data_temp
        center_line = data_prov_3d[int(n_points[2] / 2), int(n_ph / 2), :]
        ind_krd_0 = np.argmax(np.abs(center_line))
        if ind_krd_0 < n_rd / 2 - add_rd_points or ind_krd_0 > n_rd / 2 + add_rd_points:
            ind_krd_0 = int(n_rd / 2)

        data_full_4d = np.reshape(data_full, shape=(n_scans, n_sl, n_ph, n_rd))
        data_full_4d = data_full_4d[:, :, :, ind_krd_0 - int(n_points[0] / 2):ind_krd_0 + int(n_points[0] / 2)]
        data_temp = np.zeros_like(data_full_4d)
        for ii in range(n_ph):
            data_temp[:, :, ind[ii], :] = data_full_4d[:, :, ii, :]
        data_full_4d = data_temp
        output_dict['data_full'] = data_full_4d
        data = np.average(data_full_4d, axis=0)

    # Do zero padding (partial Fourier)
    data_temp = np.zeros(shape=(n_points[2], n_points[1], n_points[0]), dtype=complex)
    data_temp[0:n_sl, :, :] = data
    if k_fill == 'POCS':
        data_temp = utils.run_pocs_reconstruction(
            n_points=n_points[::-1], factors=[par_fourier_fraction, 1, 1], k_space_ref=data_temp)
    data = np.reshape(data_temp, shape=(1, n_points[0] * n_points[1] * n_points[2]))

    # Fix the position of the sample according to dfov
    bw = mat_data['bw_MHz'].item()
    time_vector = np.linspace(-n_points[0] / bw / 2 + 0.5 / bw,
                              n_points[0] / bw / 2 - 0.5 / bw, n_points[0]) * 1e-6  # s
    kMax = np.squeeze(np.array(n_points) / (2 * np.array(fov)) * np.array(mat_data['axes_enable']))
    kRD = time_vector * hw.gammaB * mat_data['rd_grad_amplitude'].item()
    kPH = np.linspace(-kMax[1], kMax[1], num=n_points[1], endpoint=False)
    kSL = np.linspace(-kMax[2], kMax[2], num=n_points[2], endpoint=False)
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kPH = np.reshape(kPH, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kSL = np.reshape(kSL, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    dPhase = np.exp(2 * np.pi * 1j * (dfov[0] * kRD + dfov[1] * kPH + dfov[2] * kSL))
    data = np.reshape(data * dPhase, shape=(n_points[2], n_points[1], n_points[0]))
    output_dict['kSpace3D'] = data
    output_dict['image3D'] = utils.run_ifft(data)
    data = np.reshape(data, shape=(1, n_points[0] * n_points[1] * n_points[2]))

    # Store undersampling metadata
    if has_mask:
        output_dict['undersampling_mask'] = undersampling_mask
    if has_vd:
        output_dict['vd_ordering'] = vd_ordering

    # Create sampled data
    kRD = np.reshape(kRD, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kPH = np.reshape(kPH, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kSL = np.reshape(kSL, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    data = np.reshape(data, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    output_dict['kMax_1/m'] = kMax
    output_dict['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    output_dict['sampledCartesian'] = output_dict['sampled']
    data = np.reshape(data, shape=(n_points[2], n_points[1], n_points[0]))

    # Get axes in strings
    axesDict = {'x': 0, 'y': 1, 'z': 2}
    axesKeys = list(axesDict.keys())
    axesVals = list(axesDict.values())
    axesStr = ['', '', '']
    n = 0
    for val in axes_orientation:
        index = axesVals.index(val)
        axesStr[n] = axesKeys[index]
        n += 1

    if axes_enable[1] == 0 and axes_enable[2] == 0:
        bw = mat_data['bw_MHz'] * 1e-3  # kHz
        acqTime = mat_data['acqTime']  # ms
        tVector = np.linspace(-acqTime / 2, acqTime / 2, n_points[0])
        sVector = mat_data['sampled'][:, 3]
        fVector = np.linspace(-bw / 2, bw / 2, n_points[0])
        iVector = utils.run_ifft(sVector)

        result_1 = {}
        result_1['widget'] = 'curve'
        result_1['xData'] = tVector
        result_1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
        result_1['xLabel'] = 'Time (ms)'
        result_1['yLabel'] = 'Signal amplitude (mV)'
        result_1['title'] = "Signal"
        result_1['legend'] = ['Magnitude', 'Real', 'Imaginary']
        result_1['row'] = 0
        result_1['col'] = 0

        result_2 = {}
        result_2['widget'] = 'curve'
        result_2['xData'] = fVector
        result_2['yData'] = [np.abs(iVector)]
        result_2['xLabel'] = 'Frequency (kHz)'
        result_2['yLabel'] = "Amplitude (a.u.)"
        result_2['title'] = "Spectrum"
        result_2['legend'] = ['Spectrum magnitude']
        result_2['row'] = 1
        result_2['col'] = 0

        output = [result_1, result_2]

    else:
        # Plot image
        image = np.abs(output_dict['image3D'])

        if mat_data['unlock_orientation'] == 0:
            result_1, _, _ = utils.fix_image_orientation(image, axes=axes_orientation, rd_direction=rd_direction)
            result_1['row'] = 0
            result_1['col'] = 0
        else:
            result_1 = {'widget': 'image',
                        'data': image,
                        'xLabel': "%s" % axesStr[1],
                        'yLabel': "%s" % axesStr[0],
                        'title': "i-Space",
                        'row': 0,
                        'col': 0}

        # k-space plot
        if par_fourier_fraction == 1:
            data = np.log10(np.abs(output_dict['kSpace3D']))
        else:
            if k_fill == 'ZP':
                data = np.zeros_like(output_dict['kSpace3D'], dtype=float)
                data[0:n_sl, :, :] = np.log10(np.abs(output_dict['kSpace3D'][0:n_sl, :, :]))
            elif k_fill == 'POCS':
                data = np.log10(np.abs(output_dict['kSpace3D']))

        if mat_data['unlock_orientation'] == 0:
            result_2, _, _ = utils.fix_image_orientation(data, axes=axes_orientation, rd_direction=rd_direction)
            result_2['row'] = 0
            result_2['col'] = 1
            result_2['title'] = "k-Space"
        elif mat_data['unlock_orientation'] == 1:
            result_2 = {'widget': 'image',
                        'data': data,
                        'xLabel': "%s" % axesStr[1],
                        'yLabel': "%s" % axesStr[0],
                        'title': "k-Space",
                        'row': 0,
                        'col': 1}

        # Dicom parameters
        dicom_meta_data["RepetitionTime"] = mat_data['repetitionTime']
        dicom_meta_data["EchoTime"] = mat_data['echoSpacing']
        dicom_meta_data["EchoTrainLength"] = mat_data['etl']

        output = [result_1, result_2]

    return output_dict, output, dicom_meta_data
