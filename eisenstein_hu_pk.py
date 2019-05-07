from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import CubicSpline

def compute_pknowiggle(k_ov_h, A_s, n_s, h0, om0, ob0):
    trans = eisensteinhu_nowiggle(k_ov_h*h0, om0*h0**2, ob0/om0)
    pk=A_s*(2.0*k_ov_h**2*2998.0**2/5.0/om0)**2*trans**2*(k_ov_h*h0/0.05)**(n_s-1.0)*2.0*3.14159265359**2/k_ov_h**3

    return pk

def eisensteinhu_nowiggle(ak, omegamh2, fb):
    alpha= 1.0-0.328*np.log(431.0*omegamh2)*fb +0.38*np.log(22.3*omegamh2)*fb**2
    sound= 44.5*np.log(9.83/omegamh2)/np.sqrt(1.0+10.0*(fb*omegamh2)**(0.75))
    shape= omegamh2*(alpha+(1.0-alpha)/(1.0+(0.43*ak*sound)**4))
    aq= ak*(2.725/2.7)**2/shape
    T = np.log(2.0*np.exp(1)+1.8*aq)/(np.log(2.0*np.exp(1)+1.8*aq)+(14.2+731/(1+62.5*aq))*aq*aq)

    return T

def compute_pkwithwiggle(k_ov_h, A_s, n_s, h0, om0, ob0):
    trans = eisensteinhu_withwiggle(k_ov_h*h0, om0*h0**2, ob0*h0**2, ob0/om0)
    pk=A_s*(2.0*k_ov_h**2*2998.0**2/5.0/om0)**2*trans**2*(k_ov_h*h0/0.05)**(n_s-1.0)*2.0*3.14159265359**2/k_ov_h**3
    return pk

def eisensteinhu_withwiggle(ak, Omh2, Obh2, f_baryon):
    # redshift and wavenumber equality
    z_eq = 2.5e4 * Omh2 * (2.725/2.7)**(-4)
    k_eq = 0.0746 * Omh2 * (2.725/2.7) ** (-2)

    #sound horizon and k_silk
    z_drag_b1 = 0.313 * Omh2 ** -0.419 * (1 + 0.607 * Omh2 ** 0.674)
    z_drag_b2 = 0.238 * Omh2 ** 0.223
    z_drag = 1291 * Omh2 ** 0.251 / (1. + 0.659 * Omh2 ** 0.828) * (1. + z_drag_b1 * Obh2 ** z_drag_b2)

    r_drag = 31.5 * Obh2 * (2.725/2.7) ** -4 * (1000. / (1+z_drag))
    r_eq   = 31.5 * Obh2 * (2.725/2.7) ** -4 * (1000. / z_eq)

    sound_horizon = 2. / (3.*k_eq) * np.sqrt(6. / r_eq) * np.log((np.sqrt(1 + r_drag) + np.sqrt(r_drag + r_eq)) / (1 + np.sqrt(r_eq)) )
    k_silk = 1.6 * Obh2 ** 0.52 * Omh2 ** 0.73 * (1 + (10.4*Omh2) ** -0.95)

    # alpha c
    alpha_c_a1 = (46.9*Omh2) ** 0.670 * (1 + (32.1*Omh2) ** -0.532)
    alpha_c_a2 = (12.0*Omh2) ** 0.424 * (1 + (45.0*Omh2) ** -0.582)
    alpha_c = alpha_c_a1 ** -f_baryon * alpha_c_a2 ** (f_baryon**3)

    # beta_c
    beta_c_b1 = 0.944 / (1 + (458*Omh2) ** -0.708)
    beta_c_b2 = 0.395 * Omh2 ** -0.0266
    beta_c = 1. / (1 + beta_c_b1 * ((1-f_baryon) ** beta_c_b2) - 1)

    y = z_eq / (1 + z_drag)
    alpha_b_G = y * (-6.*np.sqrt(1+y) + (2. + 3.*y) * np.log((np.sqrt(1+y)+1) / (np.sqrt(1+y)-1)))
    alpha_b = 2.07 *  k_eq * sound_horizon * (1+r_drag)**-0.75 * alpha_b_G

    beta_node = 8.41 * Omh2 ** 0.435
    beta_b    = 0.5 + f_baryon + (3. - 2.*f_baryon) * np.sqrt( (17.2*Omh2) ** 2 + 1 )

    # Now we compute T

    q = ak / (13.41 * k_eq)
    ks = ak * sound_horizon

    T_c_ln_beta   = np.log(np.e + 1.8 * beta_c * q)
    T_c_ln_nobeta = np.log(np.e + 1.8 * q);
    T_c_C_alpha   = 14.2 / alpha_c + 386. / (1 + 69.9 * q ** 1.08)
    T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

    T_c_f = 1. / (1. + (ks/5.4) ** 4)
    f = lambda a, b : a / (a + b*q**2)
    T_c = T_c_f * f(T_c_ln_beta, T_c_C_noalpha) + (1-T_c_f) * f(T_c_ln_beta, T_c_C_alpha)

    s_tilde = sound_horizon * (1. + (beta_node/ks)**3) ** (-1./3.)
    ks_tilde = ak*s_tilde

    T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha)
    T_b_1 = T_b_T0 / (1. + (ks/5.2)**2 )
    T_b_2 = alpha_b / (1 + (beta_b/ks)**3 ) * np.exp(-(ak/k_silk) ** 1.4)
    T_b = np.sinc(ks_tilde/np.pi) * (T_b_1 + T_b_2)

    T = f_baryon*T_b + (1.-f_baryon)*T_c;
    return T

def setup(options):

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz_steps = int(options[option_section, 'nz_steps'])
    kmin = options[option_section, 'kmin']
    kmax = options[option_section, 'kmax']
    nk_steps = int(options[option_section, 'nk_steps'])
    verbose = options.get_bool(option_section, 'verbose', False)
    wiggle = options.get_bool(option_section, "BAO_wiggle", False)

    dz = (zmax - zmin)/(nz_steps-1)
    dk = (np.log(kmax) - np.log(kmin))/(nk_steps-1)

    if verbose:
        print('''Eisenstein & Hu power spectrum options:
        zmin: {0}
        zmax: {1}
        nz_steps: {2}
        kmin: {3}
        kmax: {4}
        nk_steps: {5}'''.format(zmin, zmax, nz_steps, kmin, kmax, nk_steps))

    config = {}
    config['zmin'] = zmin
    config['zmax'] = zmax
    config['nz_steps'] = nz_steps
    config['kmin'] = kmin
    config['kmax'] = kmax
    config['nk_steps'] = nk_steps
    config['dz'] = dz
    config['dk'] = dk
    config['wiggle'] = wiggle

    return config

def execute(block, config):
    omega_b = block['cosmological_parameters','omega_b']
    omega_m = block['cosmological_parameters','omega_m']
    w = block['cosmological_parameters','w']
    h0 = block['cosmological_parameters','h0']
    n_s = block['cosmological_parameters','n_s']
    n_run = block['cosmological_parameters','n_run']
    A_s = block['cosmological_parameters','A_s']

    wiggle = config['wiggle']

    d_z = block['growth_parameters','d_z']
    z = block['growth_parameters','z']
    n_growth = len(z)

    nz_steps = config['nz_steps']
    nk_steps = config['nk_steps']

    dz = config['dz']
    dk = config['dk']

    zmin = config['zmin']
    zmax = config['zmax']
    kmin = config['kmin']
    kmax = config['kmax']

    interpolation = CubicSpline(z, d_z, bc_type = 'natural')

    if (abs(z[0]-zmin) > 0.1 or abs(z[-1] - zmax) > 0.1):
        print('=====================')
        print('The chosen bounds of the growth module does not cover the entire redshift range requested for the power spectrum')
        print('zmin, zmax from growth: {0}, {1}'.format(z[0], z[-1]))
        print('zmin, zmax from Pk: {0}, {1}'.format(zmin, zmax))
        print('=====================')
        raise RuntimeError('')

    z_eval = np.arange(zmin, zmax+dz, dz)
    log_k_eval = np.arange(np.log(kmin), np.log(kmax), dk)
    k_eval = np.exp(log_k_eval)

    PK = np.zeros((nz_steps, nk_steps))

    if wiggle:
        for i, k_h in enumerate(k_eval):
            PK[0,i] = compute_pkwithwiggle(k_h, A_s, n_s, h0, omega_m, omega_b)
    else:
        for i, k_h in enumerate(k_eval):
            PK[0,i] = compute_pknowiggle(k_h, A_s, n_s, h0, omega_m, omega_b)


    D_0 = interpolation(0)
    PK[0,:] = PK[0,:]*D_0**2

    print(PK[0,:])

    for i, z_h in enumerate(z_eval[1:]):
        PK[i+1,:] = PK[0,:]*(interpolation(z_h) / D_0)**2

    block['matter_power_lin','z'] = z_eval
    block['matter_power_lin','k_h'] = k_eval
    block['matter_power_lin','p_k'] = PK

    return 0

def cleanup(config):
    return 0
