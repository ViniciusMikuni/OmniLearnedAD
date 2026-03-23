import numpy as np
import uproot
import os

SHAPE_NUISANCES = ['jes', 'jer', 'jms', 'jmr']
WEIGHT_NUISANCES_CORR = ['pu', 'pref']
WEIGHT_NUISANCES_INDEP = ['isr', 'fsr']
TOPPT_PROCS = ['top_dijet']
NO_ISR_FSR_PROCS = {'ww_dijet', 'wz_dijet', 'zz_dijet'}
NO_SCALE_PROCS = {'ww_dijet', 'wz_dijet', 'zz_dijet'}
NO_JES_JER_PROCS = {'ww_dijet', 'wz_dijet', 'zz_dijet'}
SIGNAL_NAMES = ['dihiggs_dijet']


def _obs1d(sample, mask):
    return sample.obs[mask][:, 0].ravel()


def _w1d(sample, mask, name='nom'):
    return sample.weight(name=name)[mask][:, 0].ravel()


def _hist_and_var(x, w, bin_edges):
    vals, _ = np.histogram(x, bins=bin_edges, weights=w)
    sw2, _ = np.histogram(x, bins=bin_edges, weights=w**2)
    return vals, sw2


def _hist_var_count(x, w, bin_edges):
    vals, sw2 = _hist_and_var(x, w, bin_edges)
    counts, _ = np.histogram(x, bins=bin_edges)
    return vals, sw2, counts


def _is_present(hist):
    return np.sum(hist.values(flow=False)) > 0


def _qcd_bin_entries(bin_line, process_line, qcd_name, channel):
    return [
        '1' if proc.startswith(f'{qcd_name}_bin_') and ch == channel else '-'
        for ch, proc in zip(bin_line, process_line)
    ]


def _qcd_bin_unc_entries(bin_line, process_line, qcd_name, bin_idx, unc, channel):
    target = f'{qcd_name}_bin_{bin_idx}'
    value = f'{1 + unc:.4f}'
    return [value if proc == target and ch == channel else '-' for ch, proc in zip(bin_line, process_line)]


def _append_abcd_rateparams(lines, qcd_name, tag, year, n_bins):
    for b in range(1, n_bins + 1):
        cr1_name = f'yield_{qcd_name}_CR1_bin_{b}_{year}_{tag}'
        cr2_name = f'yield_{qcd_name}_CR2_bin_{b}_{year}_{tag}'
        cr3_name = f'yield_{qcd_name}_CR3_bin_{b}_{year}_{tag}'
        sr_name = f'yield_{qcd_name}_SR_bin_{b}_{year}_{tag}'
        qcd_proc = f'{qcd_name}_bin_{b}'
        lines += [
            f'{cr1_name} rateParam {tag}_CR1 {qcd_proc} 1.0 [0,5]',
            f'{cr2_name} rateParam {tag}_CR2 {qcd_proc} 1.0 [0,5]',
            f'{cr3_name} rateParam {tag}_CR3 {qcd_proc} 1.0 [0,5]',
            f'{sr_name}  rateParam {tag}_SR  {qcd_proc} (@0*@1/@2) {cr1_name},{cr3_name},{cr2_name}',
            '',
        ]


def SetStyle():
    from matplotlib import rc
    import matplotlib as mpl

    rc('font', family='serif', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)
    mpl.rcParams.update({
        'font.size': 19,
        'text.usetex': False,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.labelsize': 18,
        'legend.frameon': False,
        'lines.linewidth': 2,
        'figure.figsize': (9, 9),
    })



def make_th1(name, values, sumw2, edges, counts=None):
    edges = np.array(edges, dtype=np.float64)
    values = np.array(values, dtype=np.float64).clip(0.0)
    sumw2 = np.array(sumw2, dtype=np.float64)
    n_bins = len(edges) - 1

    data = np.zeros(n_bins + 2, dtype=np.float64)
    data[1:-1] = values

    fSumw2 = np.zeros(n_bins + 2, dtype=np.float64)
    fSumw2[1:-1] = sumw2

    fEntries = float(np.sum(counts)) if counts is not None else float(np.sum(values))
    fTsumw = float(np.sum(values))
    fTsumw2 = float(np.sum(sumw2))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    fTsumwx = float(np.sum(values * bin_centers))
    fTsumwx2 = float(np.sum(values * bin_centers**2))

    fXaxis = uproot.writing.identify.to_TAxis(
        fName='xaxis',
        fTitle='',
        fNbins=n_bins,
        fXmin=float(edges[0]),
        fXmax=float(edges[-1]),
        fXbins=edges,
    )

    return uproot.writing.identify.to_TH1x(
        fName=name,
        fTitle=name,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
    )




def _shape_row(nuisance, affected_procs, process_line, width=50, bin_line=None, bin_pattern=None):
    if bin_line is None or bin_pattern is None:
        vals = ['1' if p in affected_procs else '-' for p in process_line]
    else:
        vals = [
            '1' if (p in affected_procs and b.startswith(bin_pattern)) else '-'
            for b, p in zip(bin_line, process_line)
        ]
    return f"{nuisance:<{width}} shape  " + '  '.join(vals)



def _lnN_row(nuisance, affected, process_line, width=50, bin_line=None, bin_pattern=None):
    if bin_line is None or bin_pattern is None:
        vals = [affected.get(p, '-') for p in process_line]
        return f"{nuisance:<{width}} lnN  " + '  '.join(vals)

    # Filter by both process and bin pattern
    vals = [
        affected[p] if (p in affected and bin_pattern in b) else '-'
        for b, p in zip(bin_line, process_line)
    ]
    return f"{nuisance:<{width}} lnN  " + '  '.join(vals)


def _write_systematics(lines, process_line, bkg_procs, mc_names, do_sys, bin_line, width=50):
    all_mc = set(bkg_procs) | set(SIGNAL_NAMES)

    lines.append(
        _lnN_row(
            'lumi_13TeV',
            {
                **{s: '1.025' for s in SIGNAL_NAMES},
                **{p: '1.025' for p in bkg_procs},
            },
            process_line,
            width,
        )
    )

    sf_v = ['wjets_dijet','zjets_dijet']

    lines.append(
        _lnN_row(
            f'sf_v',
            {
                **{g: '1.5' for g in sf_v}
            },
            process_line,
            width,
            bin_line=bin_line,
            bin_pattern='_SR',
        )
    )

    sf_vv = ['wz_dijet','ww_dijet','zz_dijet']

    lines.append(
        _lnN_row(
            f'sf_vv',
            {
                **{g: '1.5' for g in sf_vv}
            },
            process_line,
            width,
            bin_line=bin_line,
            bin_pattern='_SR',
        )
    )


    sf_stop = ['stopw_dijet','stop_dijet']

    lines.append(
        _lnN_row(
            f'sf_stop',
            {
                **{g: '1.5' for g in sf_stop}
            },
            process_line,
            width,
            bin_line=bin_line,
            bin_pattern='_SR',
        )
    )

    
    if not do_sys:
        return

    lines.append('')

    corr_samples = [s for s in all_mc if s not in NO_JES_JER_PROCS]
    for nuisance in SHAPE_NUISANCES + WEIGHT_NUISANCES_CORR:
        lines.append(_shape_row(nuisance, corr_samples, process_line, width))

    scale_samples = [s for s in mc_names if s not in NO_SCALE_PROCS]
    for sample in scale_samples:
        lines.append(_shape_row(f'scale_{sample}', {sample}, process_line, width))

    isr_fsr_samples = [s for s in mc_names if s not in NO_ISR_FSR_PROCS]
    for sample in isr_fsr_samples:
        for nuisance in WEIGHT_NUISANCES_INDEP:
            lines.append(_shape_row(f'{nuisance}_{sample}', {sample}, process_line, width))

    lines.append(_shape_row('toppt', set(TOPPT_PROCS) & all_mc, process_line, width))



def _print_yield_summary(regions, data_per_region, qcd_per_region, mc_hists_per_region, qcd_name, bin_edges):
    print(f"\n{'Region':<8} {'Process':<25} {'Yield':>12}")
    print('-' * 47)
    for region in regions:
        d_vals, _ = data_per_region[region]
        qcd_v, _ = qcd_per_region[region]
        print(f"  {region:<6} {'data_obs':<25} {d_vals.sum():>12.2f}")
        print(f"{'':>8} {qcd_name:<25} {qcd_v.sum():>12.2f}")
        for proc_name, (mass_arr, w_arr) in mc_hists_per_region[region].items():
            vals, _ = np.histogram(mass_arr, bins=bin_edges, weights=w_arr)
            print(f"{'':>8} {proc_name:<25} {vals.sum():>12.2f}")



def hist_from_arrays(mass_arr, w_arr, bin_edges):
    return _hist_var_count(mass_arr, w_arr, bin_edges)



def data_hist(data, mask, bin_edges):
    return _hist_and_var(_obs1d(data, mask), _w1d(data, mask), bin_edges)



def get_abcd_prediction_per_region(data, mcs, cut, bin_edges):
    n_bins = len(bin_edges) - 1
    _, mask_b, mask_c, mask_d = abcd_pred(
        data.prediction[:, 0], data.prediction[:, 1], cut, cut, data.mask
    )

    data_b, data_b_sw2 = data_hist(data, mask_b, bin_edges)
    data_c, data_c_sw2 = data_hist(data, mask_c, bin_edges)
    data_d, data_d_sw2 = data_hist(data, mask_d, bin_edges)

    mc_b, mc_b_sw2 = np.zeros(n_bins), np.zeros(n_bins)
    mc_c, mc_c_sw2 = np.zeros(n_bins), np.zeros(n_bins)
    mc_d, mc_d_sw2 = np.zeros(n_bins), np.zeros(n_bins)

    for proc_name, mc in mcs.items():
        if proc_name in SIGNAL_NAMES:
            continue

        _, mb, mc_, md = abcd_pred(mc.prediction[:, 0], mc.prediction[:, 1], cut, cut, mc.mask)
        for region_mask, acc, acc_sw2 in ((mb, mc_b, mc_b_sw2), (mc_, mc_c, mc_c_sw2), (md, mc_d, mc_d_sw2)):
            vals, sw2 = _hist_and_var(_obs1d(mc, region_mask), _w1d(mc, region_mask), bin_edges)
            acc += vals
            acc_sw2 += sw2

    B, C, D = data_b - mc_b, data_c - mc_c, data_d - mc_d
    B_sw2, C_sw2, D_sw2 = data_b_sw2 + mc_b_sw2, data_c_sw2 + mc_c_sw2, data_d_sw2 + mc_d_sw2
    abcd_vals = np.divide(D, C, out=np.zeros_like(D, dtype=float), where=C != 0) * B

    valid = (B != 0) & (C != 0) & (D != 0)
    abcd_sw2 = np.zeros_like(abcd_vals)
    abcd_sw2[valid] = abcd_vals[valid]**2 * (
        D_sw2[valid] / D[valid]**2 + C_sw2[valid] / C[valid]**2 + B_sw2[valid] / B[valid]**2
    )
    return abcd_vals, abcd_sw2, B, B_sw2, C, C_sw2, D, D_sw2

def _qcd_shape_syst_name(pred_type, observable, size, year, tag):
    return f"qcd_shape_{pred_type}_{observable}_{size}_{year}"


def _read_hist_values_edges(obj):
    vals, edges = obj.to_numpy(flow=False)
    return np.asarray(vals, dtype=np.float64), np.asarray(edges, dtype=np.float64)

def add_qcd_shape_variations_to_root(
    output_path,
    qcd_shape_root,
    tag,
    pred_type,
    observable,
    size,
    year,
):
    """
    Read relative shape histograms from qcd_shape_root:
      {tag}_SR/qcd_shape_<...>Up
      {tag}_SR/qcd_shape_<...>Down

    and expand them into per-bin process systematics in output_path:
      SR/QCD_{tag}_bin_i_<nuisance>Up
      SR/QCD_{tag}_bin_i_<nuisance>Down
    """
    nuisance = _qcd_shape_syst_name(pred_type, observable, size, year, tag)
    src_dir = f"{tag}_SR"
    qcd_name = f"QCD_{tag}"

    # Read external relative variations
    with uproot.open(qcd_shape_root) as fin:
        up_key = f"{src_dir}/{nuisance}Up"
        down_key = f"{src_dir}/{nuisance}Down"

        if up_key not in fin:
            raise KeyError(f"Missing histogram: {up_key} in {qcd_shape_root}")
        if down_key not in fin:
            raise KeyError(f"Missing histogram: {down_key} in {qcd_shape_root}")

        up_vals, _ = _read_hist_values_edges(fin[up_key])
        down_vals, _ = _read_hist_values_edges(fin[down_key])

    n_bins = len(up_vals)

    # Read nominal per-bin QCD templates from the output file
    nominal_hists = {}
    with uproot.open(output_path) as f_nom:
        for i in range(n_bins):
            proc = f"{qcd_name}_bin_{i+1}"
            nominal_key = f"SR/{proc}"

            if nominal_key not in f_nom:
                raise KeyError(f"Missing nominal histogram in output file: {nominal_key}")

            obj = f_nom[nominal_key]
            nominal_vals, edges = _read_hist_values_edges(obj)

            try:
                nominal_var = obj.variances(flow=False)
                if nominal_var is None:
                    nominal_var = np.zeros_like(nominal_vals, dtype=np.float64)
            except Exception:
                nominal_var = np.zeros_like(nominal_vals, dtype=np.float64)

            nominal_hists[proc] = (
                nominal_vals,
                np.asarray(nominal_var, dtype=np.float64),
                edges,
            )

    # Write shape-varied per-bin templates
    with uproot.update(output_path) as fout:
        for i in range(n_bins):
            proc = f"{qcd_name}_bin_{i+1}"
            nominal_vals, nominal_var, edges = nominal_hists[proc]

            up_factor = up_vals[i]
            down_factor = down_vals[i]

            up_hist_vals = nominal_vals * up_factor
            down_hist_vals = nominal_vals * down_factor

            up_hist_sw2 = nominal_var * (up_factor ** 2)
            down_hist_sw2 = nominal_var * (down_factor ** 2)

            fout[f"SR/{proc}_{nuisance}Up"] = make_th1(
                f"{proc}_{nuisance}Up",
                up_hist_vals,
                up_hist_sw2,
                edges,
            )

            fout[f"SR/{proc}_{nuisance}Down"] = make_th1(
                f"{proc}_{nuisance}Down",
                down_hist_vals,
                down_hist_sw2,
                edges,
            )
            
def save_combine_histograms(
    output_path,
    bin_edges,
    data,
    mcs,
    cut,
    year,
    tag,
    mcs_sys=None,
    do_sys=False,
    qcd_shape_root=None,
    pred_type=None,
    observable=None,
    size=None,
):
    mcs_sys = mcs_sys or {}
    n_bins = len(bin_edges) - 1
    qcd_name = f'QCD_{tag}'
    regions = ['SR', 'CR1', 'CR2', 'CR3']
    region_order = ['SR', 'CR1', 'CR2', 'CR3']

    mask_sr, mask_cr1, mask_cr2, mask_cr3 = abcd_pred(
        data.prediction[:, 0], data.prediction[:, 1], cut, cut, data.mask
    )
    data_per_region = {
        'SR': data_hist(data, mask_sr, bin_edges),
        'CR1': data_hist(data, mask_cr1, bin_edges),
        'CR2': data_hist(data, mask_cr2, bin_edges),
        'CR3': data_hist(data, mask_cr3, bin_edges),
    }

    mc_hists_per_region = {r: {} for r in regions}
    for proc_name, mc in mcs.items():
        masks = get_region_masks(mc, cut)
        for region_idx, region_name in enumerate(region_order):
            mask = masks[region_idx]
            mc_hists_per_region[region_name][proc_name] = (_obs1d(mc, mask), _w1d(mc, mask))

    def _clip_negative(h):
        return np.maximum(h, 1e-5)

    sys_hists_per_region = {r: {p: {} for p in mcs} for r in regions}

    for proc_name, mc in mcs.items():
        nom_masks = get_region_masks(mc, cut)

        def _weight_sys(mask, var):
            masses = _obs1d(mc, mask)
            h_up, sw2_up = _hist_and_var(masses, _w1d(mc, mask, f'{var}_up'), bin_edges)
            h_down, sw2_down = _hist_and_var(masses, _w1d(mc, mask, f'{var}_down'), bin_edges)
            return (_clip_negative(h_up), sw2_up), (_clip_negative(h_down), sw2_down)

        for region_idx, region_name in enumerate(region_order):
            mask = nom_masks[region_idx]

            for wvar in WEIGHT_NUISANCES_CORR:
                sys_hists_per_region[region_name][proc_name][wvar] = _weight_sys(mask, wvar)

            if proc_name not in NO_ISR_FSR_PROCS:
                for wvar in WEIGHT_NUISANCES_INDEP:
                    sys_hists_per_region[region_name][proc_name][f'{wvar}_{proc_name}'] = _weight_sys(mask, wvar)

            if proc_name in TOPPT_PROCS:
                sys_hists_per_region[region_name][proc_name]['toppt'] = _weight_sys(mask, 'toppt')

            masses = _obs1d(mc, mask)
            w_nom = _w1d(mc, mask)
            w_scales = mc.weight(name='scale')[mask]
            h_nom, _ = np.histogram(masses, bins=bin_edges, weights=w_nom)
            nom_sw2, _ = np.histogram(masses, bins=bin_edges, weights=w_nom**2)
            h_scales = np.stack([
                np.histogram(masses, bins=bin_edges, weights=w_scales[:, :, i][:,0].flatten())[0]
                for i in range(6)
            ], axis=0)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(h_nom > 0, h_scales / h_nom, 1.0)

            if not np.allclose(ratios, 1.0):
                ratio_up = np.max(ratios, axis=0)
                ratio_down = np.min(ratios, axis=0)
                sys_hists_per_region[region_name][proc_name][f'scale_{proc_name}'] = (
                    (_clip_negative(h_nom * ratio_up), nom_sw2 * ratio_up**2),
                    (_clip_negative(h_nom * ratio_down), nom_sw2 * ratio_down**2),
                )
            else:
                print(f'[!] Skipping scale_{proc_name} in {region_name}: envelope is trivial')

        for base in SHAPE_NUISANCES:
            for region_name in regions:
                sys_hists_per_region[region_name][proc_name].setdefault(base, [None, None])

            for direction in ('up', 'down'):
                sys_key = f'{proc_name}_{base}_{direction}'
                if sys_key not in mcs_sys:
                    continue

                mc_var = mcs_sys[sys_key]
                var_masks = get_region_masks(mc_var, cut)
                for region_idx, region_name in enumerate(region_order):
                    var_mask = var_masks[region_idx]
                    masses = _obs1d(mc_var, var_mask)
                    h, sw2 = _hist_and_var(masses, _w1d(mc_var, var_mask), bin_edges)
                    sys_hists_per_region[region_name][proc_name][base][0 if direction == 'up' else 1] = (_clip_negative(h), sw2)

    abcd_vals, abcd_sw2, _, _, _, _, _, _ = get_abcd_prediction_per_region(data, mcs, cut, bin_edges)

    def _qcd_in_region(region, d_vals, d_sw2):
        mc_sum, mc_sum_sw2 = np.zeros(n_bins), np.zeros(n_bins)
        for proc_name, (mass_arr, w_arr) in mc_hists_per_region[region].items():
            if proc_name in SIGNAL_NAMES:
                continue
            v, s, _ = hist_from_arrays(mass_arr, w_arr, bin_edges)
            mc_sum += v
            mc_sum_sw2 += s
        return np.maximum(d_vals - mc_sum, 0.0), np.zeros_like(d_sw2 + mc_sum_sw2)

    qcd_per_region = {
        'SR': (abcd_vals, np.zeros_like(abcd_sw2)),
        'CR1': _qcd_in_region('CR1', *data_per_region['CR1']),
        'CR2': _qcd_in_region('CR2', *data_per_region['CR2']),
        'CR3': _qcd_in_region('CR3', *data_per_region['CR3']),
    }

    with uproot.recreate(output_path) as f:
        for region in regions:
            d_vals, d_sw2 = data_per_region[region]
            qcd_vals, qcd_sw2 = qcd_per_region[region]

            f[f'{region}/data_obs'] = make_th1('data_obs', d_vals, d_sw2, bin_edges)
            f[f'{region}/{qcd_name}'] = make_th1(qcd_name, qcd_vals, qcd_sw2, bin_edges)

            for hist_name, th1 in make_per_bin_qcd(qcd_vals, qcd_sw2, bin_edges, process_name=qcd_name).items():
                f[f'{region}/{hist_name}'] = th1

            for proc_name, (mass_arr, w_arr) in mc_hists_per_region[region].items():
                vals, sw2, counts = hist_from_arrays(mass_arr, w_arr, bin_edges)
                f[f'{region}/{proc_name}'] = make_th1(proc_name, vals, sw2, bin_edges, counts)

                if not do_sys:
                    continue

                for nuisance, h_pair in sys_hists_per_region[region][proc_name].items():
                    up_pair, down_pair = h_pair
                    if up_pair is None or down_pair is None:
                        continue
                    h_up, sw2_up = up_pair
                    h_down, sw2_down = down_pair

                    out_nuisance = nuisance
                    
                    f[f'{region}/{proc_name}_{out_nuisance}Up'] = make_th1(
                        f'{proc_name}_{out_nuisance}Up', h_up, sw2_up, bin_edges
                    )
                    f[f'{region}/{proc_name}_{out_nuisance}Down'] = make_th1(
                        f'{proc_name}_{out_nuisance}Down', h_down, sw2_down, bin_edges
                    )

    # only new addition
    if qcd_shape_root is not None:
        if pred_type is None or observable is None or size is None:
            raise ValueError(
                'pred_type, observable, and size are required when qcd_shape_root is provided'
            )

        add_qcd_shape_variations_to_root(
            output_path=output_path,
            qcd_shape_root=qcd_shape_root,
            tag=tag,
            pred_type=pred_type,
            observable=observable,
            size=size,
            year=year,
        )

    print(f'[✓] Saved histograms to: {output_path}')
    _print_yield_summary(regions, data_per_region, qcd_per_region, mc_hists_per_region, qcd_name, bin_edges)

def write_datacard(
    output_path,
    root_file,
    bin_edges,
    mc_names,
    abcd_vals,
    data_per_cr,
    year='2016',
    tag='HHdijet',
    do_sys=False,
    qcd_sr_bin_unc=None,
    pred_type=None,
    observable=None,
    size=None,
):
    n_bins = len(bin_edges) - 1
    regions = ['SR', 'CR1', 'CR2', 'CR3']
    qcd_name = f'QCD_{tag}'

    # ---------------------------------------------------
    # Decide which background processes are present in each region
    # ---------------------------------------------------
    region_bkg_procs = {r: [] for r in regions}

    with uproot.open(root_file) as f:
        for region in regions:
            for proc in mc_names:
                if proc in SIGNAL_NAMES:
                    continue
                key = f'{region}/{proc}'
                if key not in f:
                    continue
                if _is_present(f[key]):
                    region_bkg_procs[region].append(proc)

    # For the standard systematic rows, use the union of actually present bkgs
    bkg_procs = sorted(set(p for plist in region_bkg_procs.values() for p in plist))

    lines = [
        f'# Datacard for {tag} ABCD analysis — {year}',
        f'imax {len(regions)}  number of channels',
        'jmax *  number of backgrounds',
        'kmax *  number of nuisance parameters',
        '', '-' * 80,
        *[
            f'shapes * {tag}_{r} {root_file} {r}/$PROCESS {r}/$PROCESS_$SYSTEMATIC'
            for r in regions
        ],
        '', '-' * 80, '',
        'bin          ' + '  '.join(f'{tag}_{r}' for r in regions),
        'observation  ' + '  '.join(['-1'] * len(regions)),
        '', '-' * 80, '',
    ]

    bin_line, process_line, proc_idx_line, rate_line = [], [], [], []

    for region in regions:
        # signals
        for sig_idx, sig_name in enumerate(SIGNAL_NAMES):
            bin_line.append(f'{tag}_{region}')
            process_line.append(sig_name)
            proc_idx_line.append(str(sig_idx - len(SIGNAL_NAMES) + 1))
            rate_line.append('-1')

        # QCD per-bin processes
        for i in range(n_bins):
            bin_line.append(f'{tag}_{region}')
            process_line.append(f'{qcd_name}_bin_{i+1}')
            proc_idx_line.append(str(i + 1))
            rate_line.append('-1')

        # only backgrounds that actually exist in this region
        for j, proc in enumerate(region_bkg_procs[region]):
            bin_line.append(f'{tag}_{region}')
            process_line.append(proc)
            proc_idx_line.append(str(n_bins + 1 + j))
            rate_line.append('-1')

    lines += [
        'bin      ' + '  '.join(bin_line),
        'process  ' + '  '.join(process_line),
        'process  ' + '  '.join(proc_idx_line),
        'rate     ' + '  '.join(rate_line),
        '', '-' * 80, '',
    ]

    _write_systematics(lines, process_line, bkg_procs, mc_names, do_sys,bin_line)

    # Optional QCD shape nuisance
    if pred_type is not None and observable is not None and size is not None:
        nuisance = f'qcd_shape_{pred_type}_{observable}_{size}_{year}_{tag}'
        entries = _qcd_bin_entries(bin_line, process_line, qcd_name, f'{tag}_SR')
        lines.append(f"{nuisance:<50} shape  " + '  '.join(entries))

    # Optional lnN per QCD bin
    if qcd_sr_bin_unc is not None:
        for i, unc in enumerate(qcd_sr_bin_unc, start=1):
            entries = _qcd_bin_unc_entries(bin_line, process_line, qcd_name, i, unc, f'{tag}_SR')
            lines.append(f"{('qcdnorm_sr_bin' + str(i - 1)):<50} lnN  " + '  '.join(entries))

    lines += ['', '* autoMCStats 0 0 1', '', '-' * 80, '']

    cr1_vals, _ = data_per_cr['CR1']
    cr2_vals, _ = data_per_cr['CR2']
    cr3_vals, _ = data_per_cr['CR3']

    _append_abcd_rateparams(lines, qcd_name, tag, year, n_bins)

    with open(output_path, 'w') as fout:
        fout.write('\n'.join(lines))
    print(f'[✓] Datacard written to: {output_path}')

    print(f"\n{'Bin':<6} {'CR1 (B)':>10} {'CR2 (C)':>10} {'CR3 (D)':>10} {'SR pred':>10}")
    print('-' * 46)
    for i in range(n_bins):
        print(f'  {i+1:<4} {cr1_vals[i]:>10.2f} {cr2_vals[i]:>10.2f} {cr3_vals[i]:>10.2f} {abcd_vals[i]:>10.2f}')
        
def write_combined_datacard_from_existing(
    output_path,
    input_datacards,
    root_files,
    bin_edges,
    mc_names,
    data_per_cr,
    abcd_vals_dict,
    year='2017',
    do_sys=False,
    qcd_sr_bin_unc=None,
    pred_type=None,
    observable=None,
    size=None
):

    regions = ['SR', 'CR1', 'CR2', 'CR3']
    tags = list(input_datacards.keys())

    if isinstance(bin_edges, dict):
        bin_edges_dict = bin_edges
    else:
        bin_edges_dict = {tag: bin_edges for tag in tags}

    n_bins_dict = {tag: len(bin_edges_dict[tag]) - 1 for tag in tags}

    # Optional per-tag qcd_sr_bin_unc
    if qcd_sr_bin_unc is None:
        qcd_sr_bin_unc_dict = {tag: None for tag in tags}
    elif isinstance(qcd_sr_bin_unc, dict):
        qcd_sr_bin_unc_dict = qcd_sr_bin_unc
    else:
        qcd_sr_bin_unc_dict = {tag: qcd_sr_bin_unc for tag in tags}

    # Determine which background processes are actually present in each tag/region
    region_bkg_procs = {tag: {r: [] for r in regions} for tag in tags}

    for tag in tags:
        with uproot.open(root_files[tag]) as f:
            for region in regions:
                for proc in mc_names:
                    if proc in SIGNAL_NAMES:
                        continue
                    key = f'{region}/{proc}'
                    if key not in f:
                        continue
                    if _is_present(f[key]):
                        region_bkg_procs[tag][region].append(proc)

    # Union of backgrounds that are present somewhere
    bkg_procs = sorted(
        set(
            p
            for tag in tags
            for region in regions
            for p in region_bkg_procs[tag][region]
        )
    )

    sf_rateparams = ['top_dijet', 'wjets_dijet', 'zjets_dijet', 'stopw_dijet']
    if 'dihiggs_dijet' in SIGNAL_NAMES:
        sf_rateparams.append('dihiggs_dijet')

    all_channels = [f'{tag}_{r}' for tag in tags for r in regions]

    lines = [
        f"# Combined datacard from: {', '.join(input_datacards.values())}",
        f'# year: {year}',
        f'imax {len(all_channels)}  number of channels',
        'jmax *  number of backgrounds',
        'kmax *  number of nuisance parameters',
        '', '-' * 80,
        *[
            f'shapes * {tag}_{r} {root_files[tag]} {r}/$PROCESS {r}/$PROCESS_$SYSTEMATIC'
            for tag in tags for r in regions
        ],
        '', '-' * 80, '',
        'bin          ' + '  '.join(all_channels),
        'observation  ' + '  '.join(['-1'] * len(all_channels)),
        '', '-' * 80, '',
    ]

    bin_line, process_line, proc_idx_line, rate_line = [], [], [], []

    for tag in tags:
        qcd_name = f'QCD_{tag}'
        n_bins = n_bins_dict[tag]

        for region in regions:
            ch = f'{tag}_{region}'

            # signals
            for sig_idx, sig_name in enumerate(SIGNAL_NAMES):
                bin_line.append(ch)
                process_line.append(sig_name)
                proc_idx_line.append(str(sig_idx - len(SIGNAL_NAMES) + 1))
                rate_line.append('-1')

            # per-bin QCD
            for i in range(n_bins):
                bin_line.append(ch)
                process_line.append(f'{qcd_name}_bin_{i+1}')
                proc_idx_line.append(str(i + 1))
                rate_line.append('-1')

            # only backgrounds that actually exist in this tag/region
            for j, proc in enumerate(region_bkg_procs[tag][region]):
                bin_line.append(ch)
                process_line.append(proc)
                proc_idx_line.append(str(n_bins + 1 + j))
                rate_line.append('-1')

    lines += [
        'bin      ' + '  '.join(bin_line),
        'process  ' + '  '.join(process_line),
        'process  ' + '  '.join(proc_idx_line),
        'rate     ' + '  '.join(rate_line),
        '', '-' * 80, '',
    ]

    _write_systematics(lines, process_line, bkg_procs, mc_names, do_sys, bin_line)

    # QCD shape systematics
    if pred_type is not None and observable is not None and size is not None:
        for tag in tags:
            nuisance = f'qcd_shape_{pred_type}_{observable}_{size}_{year}'
            qcd_name = f'QCD_{tag}'

            entries = _qcd_bin_entries(bin_line, process_line, qcd_name, f'{tag}_SR')
            lines.append(f"{nuisance:<50} shape  " + '  '.join(entries))

    # Optional lnN per QCD bin
    for tag in tags:
        tag_unc = qcd_sr_bin_unc_dict.get(tag, None)
        if tag_unc is None:
            continue

        qcd_name = f'QCD_{tag}'
        n_bins = n_bins_dict[tag]

        if len(tag_unc) != n_bins:
            raise ValueError(
                f"qcd_sr_bin_unc for tag '{tag}' has length {len(tag_unc)}, "
                f"but this tag has {n_bins} bins."
            )

        for i, unc in enumerate(tag_unc, start=1):
            entries = _qcd_bin_unc_entries(bin_line, process_line, qcd_name, i, unc, f'{tag}_SR')
            lines.append(f"{('qcdnorm_sr_' + tag + '_bin' + str(i - 1)):<50} lnN  " + '  '.join(entries))

    lines += ['', '* autoMCStats 0 0 1', '', '-' * 80, '']

    # Existing floating normalizations
    for proc in sf_rateparams:
        lines.append(f'sf_{year}_SR rateParam *_SR {proc} 0.5 [0,1]')

    lines += ['', '-' * 80, '']

    # ABCD rateParams for each tag
    for tag in tags:
        qcd_name = f'QCD_{tag}'
        n_bins = n_bins_dict[tag]

        lines.append(f'# ABCD rateParams for {tag} (uncorrelated)')

        _append_abcd_rateparams(lines, qcd_name, tag, year, n_bins)
        lines.append('')

    with open(output_path, 'w') as fout:
        fout.write('\n'.join(lines))

    print(f'[✓] Combined datacard written to: {output_path}')
    print(f'    Channels        : {len(all_channels)} ({len(tags)} tags × {len(regions)} regions)')
    print("    Free-floating   : ['top_dijet']")
    print(f"    Uncorrelated QCD: {[f'QCD_{t} ({n_bins_dict[t]} bins)' for t in tags]}")

    
def make_per_bin_qcd(vals, sw2, edges, process_name='QCD'):
    n_bins = len(vals)
    hists = {}
    for i in range(n_bins):
        bin_vals = np.zeros(n_bins)
        bin_sw2 = np.zeros(n_bins)
        bin_vals[i] = vals[i]
        bin_sw2[i] = sw2[i]
        hists[f'{process_name}_bin_{i+1}'] = make_th1(
            f'{process_name}_bin_{i+1}', bin_vals, bin_sw2, edges
        )
    return hists



def get_region_masks(mc, cut):
    return abcd_pred(mc.prediction[:, 0], mc.prediction[:, 1], cut, cut, mc.mask)



def abcd_pred(p1, p2, cut1, cut2, mask=None):
    mask_a = (p1 > cut1) & (p2 > cut2)
    mask_b = (p1 < cut1) & (p2 > cut2)
    mask_c = (p1 < cut1) & (p2 < cut2)
    mask_d = (p1 > cut1) & (p2 < cut2)
    if mask is not None:
        return mask_a & mask, mask_b & mask, mask_c & mask, mask_d & mask
    return mask_a, mask_b, mask_c, mask_d



class EventData:
    SYS_INDEX = {
        "jes_up":   (18, 19),
        "jes_down": (20, 21),
        "jer_up":   (22, 23),
        "jer_down": (24, 25),
        "jms_up":   (26, 27),
        "jms_down": (28, 29),
        "jmr_up":   (30, 31),
        "jmr_down": (32, 33),
    }

    WEIGHT_INDEX = {
        "nom": 0,
        "pu_up": 1,
        "pu_down": 2,
        "pref_up": 3,
        "pref_down": 4,
        "toppt_up": 5,
        "toppt_down": 6,
        "isr_up": 7,
        "isr_down": 8,
        "fsr_up": 9,
        "fsr_down": 10,
    }
    SCALE_INDICES = list(range(11, 17))

    def __init__(
        self,
        name,
        pred_type,
        sort_type,
        region_type,
        predictions,
        dimass,
        mass,
        pt,
        eta,
        btag,
        htag,
        tau21,
        obs,
        event_mask,
        weight,
    ):
        self.name = name
        self.pred_type = pred_type
        self.sort_type = sort_type
        self.region_type = region_type

        self.predictions = predictions
        self.dimass = dimass
        self.mass = mass
        self.pt = pt
        self.eta = eta
        self.btag = btag
        self.htag = htag
        self.tau21 = tau21
        self.obs = obs
        self.event_mask = event_mask
        self.weights = weight

        if "dihiggs" in self.name:
            self.weights *= 1000

        self.prediction = self._get_prediction()
        self._sort()
        self.mask = self._get_mask()


    @classmethod
    def from_npz_folder(
        cls,
        name,
        pred_type,
        sort_type,
        region_type,
        folder,
        pattern,
        sys="",
        observable="mass",
    ):
        keys = [
            "predictions", "dimasses", "masses", "pts", "etas",
            "btags", "htags", "tau21s", "obs", "weights", "masks"
        ]
        acc = {k: [] for k in keys}

        files = [
            f for f in os.listdir(folder)
            if pattern in f and f.endswith(".npz") and "ad" not in f
        ]
        if not files:
            raise RuntimeError(f"No matching .npz files found for pattern {pattern!r} in {folder!r}")

        for file_name in files:
            data = np.load(os.path.join(folder, file_name))
            cond = data["cond"]

            acc["predictions"].append(data["prediction"])
            acc["etas"].append(cond[:, 1])

            weight = cond[:, 34:].copy()
            btag = cond[:, 10].copy()
            btag[btag < 0] = 0
            htag = cond[:, 11].copy()

            acc["weights"].append(weight)
            acc["btags"].append(btag)
            acc["htags"].append(htag)
            acc["masks"].append((cond[:, 14] == 1.0) & (cond[:, 16] == 0.0) & (cond[:, 17] == 0.0))
            acc["tau21s"].append(cond[:, 7] / (cond[:, 6] + 1e-6))

            pt, mass = cls._extract_pt_mass(cond, sys)
            acc["pts"].append(pt)
            acc["masses"].append(mass)

            dimass = cls._compute_dijet_mass(cond[:, :4], pt, mass)
            acc["dimasses"].append(dimass)

            if observable == "dimass":
                acc["obs"].append(dimass)
            elif observable == "htag":
                acc["obs"].append(htag)
            elif observable == "mass":
                acc["obs"].append(mass)
            else:
                raise ValueError(f"Unknown observable: {observable!r}.")

        prediction = np.concatenate(acc["predictions"], axis=0)
        dimass = np.concatenate(acc["dimasses"], axis=0)
        mass = np.concatenate(acc["masses"], axis=0)
        pt = np.concatenate(acc["pts"], axis=0)
        eta = np.concatenate(acc["etas"], axis=0)
        btag = np.concatenate(acc["btags"], axis=0)
        htag = np.concatenate(acc["htags"], axis=0)
        tau21 = np.concatenate(acc["tau21s"], axis=0)
        obs = np.concatenate(acc["obs"], axis=0)
        event_mask = np.concatenate(acc["masks"], axis=0)
        weight = np.concatenate(acc["weights"], axis=0)

        def reshape(arr, extra_dim=False):
            return arr.reshape(-1, 2, arr.shape[-1]) if extra_dim else arr.reshape(-1, 2)

        return cls(
            name,
            pred_type,
            sort_type,
            region_type,
            reshape(prediction, extra_dim=True),
            reshape(dimass),
            reshape(mass),
            reshape(pt),
            reshape(eta),
            reshape(btag),
            reshape(htag),
            reshape(tau21),
            reshape(obs),
            reshape(event_mask),
            reshape(weight, extra_dim=True),
        )

    @staticmethod
    def _extract_pt_mass(cond, sys):
        if sys == "":
            return np.exp(cond[:, 0]), np.exp(cond[:, 3])

        if sys not in EventData.SYS_INDEX:
            raise ValueError(f"Unknown sys: {sys!r}. Valid options: {list(EventData.SYS_INDEX)}")

        pt_col, mass_col = EventData.SYS_INDEX[sys]
        return cond[:, pt_col], cond[:, mass_col]

    @staticmethod
    def _compute_dijet_mass(jets, pts, masses):
        dijets = jets.reshape(-1, 2, jets.shape[-1])
        pt = pts.reshape(-1, 2)
        mass = masses.reshape(-1, 2)

        px = np.sum(pt * np.cos(dijets[:, :, 2]), axis=1)
        py = np.sum(pt * np.sin(dijets[:, :, 2]), axis=1)
        pz = np.sum(pt * np.sinh(dijets[:, :, 1]), axis=1)

        e1 = np.sqrt(mass[:, 0] ** 2 + (pt[:, 0] * np.cosh(dijets[:, 0, 1])) ** 2)
        e2 = np.sqrt(mass[:, 1] ** 2 + (pt[:, 1] * np.cosh(dijets[:, 1, 1])) ** 2)

        m2 = (e1 + e2) ** 2 - (px ** 2 + py ** 2 + pz ** 2)
        m2 = np.maximum(m2, 0.0)
        m = np.repeat(np.sqrt(m2)[:, None], 2, axis=0)
        return m[:, 0]

    def weight(self, name="nom"):
        if name == "scale":
            return self.weights[:, :, self.SCALE_INDICES]
        if name not in self.WEIGHT_INDEX:
            raise ValueError(
                f"Unknown weight name: {name!r}. Valid options: {list(self.WEIGHT_INDEX)}"
            )
        return self.weights[:, :, self.WEIGHT_INDEX[name]]

    def _get_prediction(self):
        match self.pred_type:
            case "NP":
                signal = np.sum(self.predictions[:, :, 10:], axis=-1)
                return signal / (self.predictions[:, :, 0] + signal + 1e-6)
            case "htag":
                return self.htag
            case _:
                raise ValueError(f"Unknown pred_type: {self.pred_type!r}")

    def _sort(self):
        match self.sort_type:
            case "pt":
                idx = np.argsort(-self.pt, axis=1)
            case "mass":
                idx = np.argsort(-self.mass, axis=1)
            case "random":
                flip = (np.random.rand(self.pt.shape[0]) < 0.5).astype(np.int64)
                idx = np.stack([flip, 1 - flip], axis=1)
            case _:
                raise ValueError(f"Unknown sort_type: {self.sort_type!r}")

        for attr in ("prediction", "mass", "btag", "htag", "pt", "eta", "tau21", "obs"):
            arr = getattr(self, attr)
            setattr(self, attr, np.take_along_axis(arr, idx, axis=1))

    def _get_mask(self, tau_cut=0.45, pt_cut=450):
        pt_mask = np.min(self.pt, axis=1) > pt_cut
        tau21_mask = np.max(self.tau21, axis=1) < tau_cut
        dimass_mask = self.dimass[:, 0] > 1000
        event_mask = self.event_mask[:, 0]
        mass_mask = np.min(self.mass, axis=1) > 60

        self.base_mask = pt_mask & event_mask & mass_mask

        match self.region_type:
            case "SR":
                return self.base_mask & tau21_mask
            case "VR1":
                return self.base_mask & ~tau21_mask
            case "SR1":
                return self.base_mask & tau21_mask & (self.mass[:, 1] > 100)
            case "SR2":
                return self.base_mask & tau21_mask & (self.mass[:, 1] < 100)
            case "SR3":
                return self.base_mask & tau21_mask & (self.mass[:, 1] > 100) & (np.max(self.btag, axis=1) > 0.5847)
            case "SR4":
                return self.base_mask & tau21_mask & (np.max(self.btag, axis=1) > 0.5847)
            case _:
                raise ValueError(f"Unknown region_type: {self.region_type!r}")

    def obs1d(self, mask):
        return self.obs[mask][:, 0].ravel()

    def weight1d(self, mask, name="nom"):
        return self.weight(name)[mask][:, 0].ravel()

    def __repr__(self):
        return f"EventData(name={self.name!r})"
