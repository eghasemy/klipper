# Definitions of the supported input shapers
#
# Copyright (C) 2020-2021  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections, math

SHAPER_VIBRATION_REDUCTION=20.
DEFAULT_DAMPING_RATIO = 0.1

InputShaperCfg = collections.namedtuple(
        'InputShaperCfg', ('name', 'init_func', 'min_freq'))

def get_none_shaper():
    return ([], [])

def get_zv_shaper(shaper_freq, damping_ratio):
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)
    A = [1., K]
    T = [0., .5*t_d]
    return (A, T)

def get_zvd_shaper(shaper_freq, damping_ratio):
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)
    A = [1., 2.*K, K**2]
    T = [0., .5*t_d, t_d]
    return (A, T)

def get_mzv_shaper(shaper_freq, damping_ratio):
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-.75 * damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    a1 = 1. - 1. / math.sqrt(2.)
    a2 = (math.sqrt(2.) - 1.) * K
    a3 = a1 * K * K

    A = [a1, a2, a3]
    T = [0., .375*t_d, .75*t_d]
    return (A, T)

def get_ei_shaper(shaper_freq, damping_ratio):
    v_tol = 1. / SHAPER_VIBRATION_REDUCTION # vibration tolerance
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    a1 = .25 * (1. + v_tol)
    a2 = .5 * (1. - v_tol) * K
    a3 = a1 * K * K

    A = [a1, a2, a3]
    T = [0., .5*t_d, t_d]
    return (A, T)

def get_2hump_ei_shaper(shaper_freq, damping_ratio):
    v_tol = 1. / SHAPER_VIBRATION_REDUCTION # vibration tolerance
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    V2 = v_tol**2
    X = pow(V2 * (math.sqrt(1. - V2) + 1.), 1./3.)
    a1 = (3.*X*X + 2.*X + 3.*V2) / (16.*X)
    a2 = (.5 - a1) * K
    a3 = a2 * K
    a4 = a1 * K * K * K

    A = [a1, a2, a3, a4]
    T = [0., .5*t_d, t_d, 1.5*t_d]
    return (A, T)

def get_3hump_ei_shaper(shaper_freq, damping_ratio):
    v_tol = 1. / SHAPER_VIBRATION_REDUCTION # vibration tolerance
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    K2 = K*K
    a1 = 0.0625 * (1. + 3. * v_tol + 2. * math.sqrt(2. * (v_tol + 1.) * v_tol))
    a2 = 0.25 * (1. - v_tol) * K
    a3 = (0.5 * (1. + v_tol) - 2. * a1) * K2
    a4 = a2 * K2
    a5 = a1 * K2 * K2

    A = [a1, a2, a3, a4, a5]
    T = [0., .5*t_d, t_d, 1.5*t_d, 2.*t_d]
    return (A, T)

def get_smooth_shaper(shaper_freq, damping_ratio):
    # Optimized shaper for minimal smoothing with good vibration reduction
    v_tol = 1. / (SHAPER_VIBRATION_REDUCTION * 1.5) # tighter tolerance for smoothness
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    # Optimized coefficients for smooth motion
    a1 = 0.2 * (1. + v_tol)
    a2 = 0.6 * (1. - v_tol) * K
    a3 = 0.2 * K * K

    # Normalize to sum to 1.0
    total = a1 + a2 + a3
    A = [a1/total, a2/total, a3/total]
    T = [0., .4*t_d, 0.8*t_d]
    return (A, T)

def get_adaptive_ei_shaper(shaper_freq, damping_ratio):
    # Adaptive EI shaper that adjusts based on damping ratio
    v_tol = 1. / SHAPER_VIBRATION_REDUCTION
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    # Adjust parameters based on damping ratio
    if damping_ratio < 0.05:
        # Low damping - more aggressive compensation
        a1 = .2 * (1. + v_tol * 1.5)
        a2 = .6 * (1. - v_tol * 0.5) * K
        a3 = .2 * K * K
    elif damping_ratio > 0.2:
        # High damping - gentler compensation
        a1 = .3 * (1. + v_tol * 0.5)
        a2 = .4 * (1. - v_tol * 1.5) * K
        a3 = .3 * K * K
    else:
        # Standard EI
        a1 = .25 * (1. + v_tol)
        a2 = .5 * (1. - v_tol) * K
        a3 = .25 * K * K

    # Normalize to sum to 1.0
    total = a1 + a2 + a3
    A = [a1/total, a2/total, a3/total]
    T = [0., .5*t_d, t_d]
    return (A, T)

def get_multi_freq_shaper(shaper_freq, damping_ratio):
    # Multi-frequency shaper for complex resonance patterns
    # Primary frequency at shaper_freq, secondary at 1.5x frequency
    v_tol = 1. / SHAPER_VIBRATION_REDUCTION
    df = math.sqrt(1. - damping_ratio**2)
    K1 = math.exp(-damping_ratio * math.pi / df)
    t_d1 = 1. / (shaper_freq * df)
    
    # Secondary frequency compensation
    freq_ratio = 1.4  # Slightly less than 1.5 for better stability
    t_d2 = t_d1 / freq_ratio
    K2 = math.exp(-damping_ratio * math.pi / (df * freq_ratio))

    # Combined shaper coefficients
    a1 = 0.15 * (1. + v_tol)
    a2 = 0.3 * (1. - v_tol * 0.5) * K1
    a3 = 0.35 * (1. - v_tol) * K2
    a4 = 0.2 * K1 * K2

    # Normalize to sum to 1.0
    total = a1 + a2 + a3 + a4
    A = [a1/total, a2/total, a3/total, a4/total]
    T = [0., .3*t_d1, .6*t_d2, t_d1]
    return (A, T)

def get_ultra_low_vibration_shaper(shaper_freq, damping_ratio):
    # Ultra low vibration shaper with very tight tolerance
    v_tol = 1. / (SHAPER_VIBRATION_REDUCTION * 2.) # Even tighter tolerance
    df = math.sqrt(1. - damping_ratio**2)
    K = math.exp(-damping_ratio * math.pi / df)
    t_d = 1. / (shaper_freq * df)

    K2 = K*K
    K3 = K2*K
    K4 = K3*K
    
    # Extended 6-impulse shaper for maximum vibration reduction
    a1 = 0.05 * (1. + 4. * v_tol + 3. * math.sqrt(3. * (v_tol + 1.) * v_tol))
    a2 = 0.15 * (1. - v_tol) * K
    a3 = 0.3 * (1. + 0.5 * v_tol) * K2
    a4 = 0.3 * (1. - 0.5 * v_tol) * K3
    a5 = 0.15 * (1. - v_tol) * K4
    a6 = 0.05 * K4 * K

    # Normalize to sum to 1.0
    total = a1 + a2 + a3 + a4 + a5 + a6
    A = [a1/total, a2/total, a3/total, a4/total, a5/total, a6/total]
    T = [0., .3*t_d, .6*t_d, t_d, 1.3*t_d, 1.6*t_d]
    return (A, T)

# min_freq for each shaper is chosen to have projected max_accel ~= 1500
INPUT_SHAPERS = [
    InputShaperCfg('zv', get_zv_shaper, min_freq=21.),
    InputShaperCfg('mzv', get_mzv_shaper, min_freq=23.),
    InputShaperCfg('zvd', get_zvd_shaper, min_freq=29.),
    InputShaperCfg('ei', get_ei_shaper, min_freq=29.),
    InputShaperCfg('2hump_ei', get_2hump_ei_shaper, min_freq=39.),
    InputShaperCfg('3hump_ei', get_3hump_ei_shaper, min_freq=48.),
    # Advanced shapers for comprehensive resonance compensation
    InputShaperCfg('smooth', get_smooth_shaper, min_freq=25.),
    InputShaperCfg('adaptive_ei', get_adaptive_ei_shaper, min_freq=32.),
    InputShaperCfg('multi_freq', get_multi_freq_shaper, min_freq=35.),
    InputShaperCfg('ulv', get_ultra_low_vibration_shaper, min_freq=55.),
]
