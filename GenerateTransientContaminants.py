# Imports
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import LambdaCDM
import redback
from redback.simulate_transients import SimulateGenericTransient
import pandas as pd
from bilby.core.prior import Uniform
from bilby.core.prior.dict import PriorDict
from scipy import integrate
from scipy.interpolate import interp1d
from spherical_geometry.polygon import SphericalPolygon
import os

## Cosmology
cosmo = LambdaCDM(H0=70. *u.km / u.s / u.Mpc, Om0=0.3, Ode0=0.70)

## Differential Comoving Volume
def diff_cm_volume(z):
    dvol = cosmo.differential_comoving_volume(z)
    return dvol.value ## In Mpc^3


## -------------------------------------------
## Rate Functions
## -------------------------------------------

def rate_CCSN(z):
    R = (1.365e-4)*((1+z)**5)/(1+((1+z)/1.5)**6.1)
    return R

def rate_SNII(z):       ## Assuming 70% of CCSN are SN II
    R = rate_CCSN(z)*0.70
    return R

def rate_SNIc(z):       ## Assumining remaining 30% of CCSN are SNIbc
    R = rate_CCSN(z)*0.15
    return R

def rate_SNIb(z):
    R = rate_CCSN(z)*0.15
    return R

def rate_SNIax(z):
    R = (6e-6)*((1+z)**2.7)/(1+((1+z)/2.9)**5.6)
    return R

def rate_SNIa(z):
    r_smallz = (2.5e-5)*(1+z)**1.5
    r_bigz = (9.7e-5)*(1+z)**(-0.5)
    R = np.where(z<=1, r_smallz,
        np.where(z>1, r_bigz, 0))
    return R

def rate_TDE(z):
    R = (1e-6)*10**(-0.009*(z**3) + 0.121*(z**2) - 0.850*z)
    return R

def rate_GRB_L(z):
    z1 = 3.11
    R_0 = 1.25e-9
    r_smallz = (1+z)**2.07
    r_bigz = ((1+z1)**(3.43))*(1+z)**(-1.36)
    R = np.where(z<z1, R_0*r_smallz,
        np.where(z>=z1, R_0*r_bigz, 0))
    return R

def rate_GRB_S(z):
    R = 1e-7*(z**0)
    return R

## -------------------------------------------
## Transient Number Functions, integrands
## -------------------------------------------

def N_CCSN(z):
    N = rate_CCSN(z)*diff_cm_volume(z)
    return N

def N_SNII(z):
    N = rate_SNII(z)*diff_cm_volume(z)
    return N

def N_SNIc(z):
    N = rate_SNIc(z)*diff_cm_volume(z)
    return N

def N_SNIb(z):
    N = rate_SNIb(z)*diff_cm_volume(z)
    return N

def N_TDE(z):
    N = rate_TDE(z)*diff_cm_volume(z)
    return N

def N_SNIa(z):
    N = rate_SNIa(z)*diff_cm_volume(z)
    return N

def N_SNIax(z):
    N = rate_SNIax(z)*diff_cm_volume(z)
    return N

def N_GRB_L(z):
    N = rate_GRB_L(z)*diff_cm_volume(z)
    return N

def N_GRB_S(z):
    N = rate_GRB_S(z)*diff_cm_volume(z)
    return N


## -------------------------------------------
## Grouping Transient Properties
## -------------------------------------------


TDE_params = {'Model': 'tde_analytical', 'Rate Function': rate_TDE, 'Number Function': N_TDE, 'deltaT': 56.0/365.2425}
SNII_params = {'Model': 'typeII_surrogate_sarin25', 'Rate Function': rate_SNII, 'Number Function': N_SNII, 'deltaT': 55.0/365.2425}
SNIa_params = {'Model': 'salt2', 'Rate Function': rate_SNIa, 'Number Function': N_SNIa, 'deltaT': 25.0/365.2425}
SNIb_params = {'Model': 'arnett', 'Rate Function': rate_SNIb, 'Number Function': N_SNIb, 'deltaT': 55.0/365.2425}
SNIc_params = {'Model': 'type_1c', 'Rate Function': rate_SNIc, 'Number Function': N_SNIc, 'deltaT': 55.0/365.2425}
GRB_L_params = {'Model': 'gaussian_redback', 'Rate Function': rate_GRB_S, 'Number Function': N_GRB_S, 'deltaT': 1.0/365.2425}
GRB_S_params = {'Model': 'gaussian_redback', 'Rate Function': rate_GRB_L, 'Number Function': N_GRB_L, 'deltaT': 20.0/365.2425}


transients = {'TDE': TDE_params,
              'SNII': SNII_params,
              'SNIa': SNIa_params,
              'SNIb': SNIb_params,
              'SNIc': SNIc_params,
              'sGRB': GRB_L_params,
              'lGRB': GRB_S_params}


transients_GroupA = {'TDE': TDE_params,
                     'SNIb': SNIb_params,
                     'SNIc': SNIc_params,
                     'sGRB': GRB_S_params,
                     'lGRB': GRB_L_params}

transients_GroupB = {'SNIa': SNIa_params,
                    'SNII': SNII_params}



# ---------------------------------------
# Model Priors
# ---------------------------------------

# Type Ib Priors
TypeIb_priors = PriorDict({'mej': Uniform(minimum=1, maximum=10, name='mej'),
                            'Ek': Uniform(minimum=1, maximum=10, name='Ek'),
                            'f_nickel': Uniform(minimum=0.01, maximum=0.15, name='f_nickel')})

def update_parameters(params, trans_type):
    if trans_type == 'SNIb':            ## For SNIb
        model = transients[trans_type]['Model']
        params = redback.priors.get_priors(model=model).sample()
        
        mej = TypeIb_priors['mej'].sample()
        f_nickel = TypeIb_priors['f_nickel'].sample()
        Ek = TypeIb_priors['Ek'].sample()
        const = np.sqrt(2./1.989e33)*1e-5
        vej = const*np.sqrt(Ek*1e51/mej)
        params['mej'] = mej
        params['f_nickel'] = f_nickel
        params['vej'] = vej
        
    else:               ## For SNIa, SNII, SNIc, and TDE
        model = transients[trans_type]['Model']
        params = redback.priors.get_priors(model=model).sample()
    return params



## Miscellaneous Functions
def flux_to_mag(flux):
    return -2.5*np.log10(flux) + 16.4

def flux_error_to_mag(flux_err, flux_obs):
    return 1.0857 * flux_err/flux_obs

def mag_to_flux(mag):
    return 10**((mag-16.4)/-2.5)

def add_noise(flux, sigma):
    noise = np.random.normal(0, sigma*flux, len(flux))
    return flux + noise

# ---------------------------------------
# Class Setup Below
# ---------------------------------------

class TransientContaminants:
    def __init__(self, coordinates, z_max=2, z_min=0):
        """
        Coordinates must be in the form of astropy SkyCoord object.


        """
        self.coordinates = coordinates
        self.z_max = z_max
        self.z_min = z_min

    ## Return the sky localization area bound by the input coordinates
    def compute_sky_area(self, unit=u.sr):
        """
        Computes the total area on the sky bound by the input coordinates. 
        unit: astropy unit for surface area of a sphere (default is steradian)
        """
        coords = self.coordinates

        coords_cartesian = coords.cartesian
        verts = np.column_stack((coords_cartesian.x.value, coords_cartesian.y.value, coords_cartesian.z.value))
        sky_polygon = SphericalPolygon(verts)



        sky_area = sky_polygon.area() * u.steradian
        output = sky_area.to(unit)
        return output
    
    ## Return the projected integrated localization volume 
    def compute_projected_volume(self):
        """
        Computes the total localization volume projected along the line of sight.
        """
        area = self.compute_sky_area().value
        z_min = self.z_min
        z_max = self.z_max
        integrand = integrate.quad(diff_cm_volume, z_min, z_max)[0]
        vol = (area/(4*np.pi)) * integrand * u.Mpc**3
        return vol 
    
    ## Return the total number of expected transient contaminants in the localization volume
    def compute_transients(self, trans_type=list(transients.keys())):
        """
        Computes the total number of expected transients in the localization volume over a ten year period.
        trans_type: list of types of transients you wish to compute the number of. Default is full set of transients with rates: SNIa, SNIb, SNIc, SNII, SNIax, sGRB, lGRB, TDE
        """
        N_tot = 0
        z_min = self.z_min
        z_max = self.z_max
        try:
            for transient in trans_type:
                area = self.compute_sky_area().value
                N_trans = transients[transient]['Number Function']
                integrand = integrate.quad(N_trans, z_min, z_max)[0]
                N = (area/(4*np.pi))*integrand*4.5               # Multiply by 4.5 instead of deltaT for a 4.5 year period.
                N_tot += N
            return N_tot
        except KeyError:
            print('ERROR: Transient type not recognized. Must be set of the following:\nSNII, SNIa, SNIb, SNIc, TDE, sGRB, lGRB, SNIax')
    

    def generate_transients(self):              
        """
        Generates lightcurves for each transient expected in the localization volume.
        """
        area = self.compute_sky_area().value
        z_min = self.z_min
        z_max = self.z_max
        z_list = np.linspace(z_min, z_max, 100)

        os.makedirs('TransientContaminantsOutput', exist_ok=True)

        ## TDEs and CCSN, output is multiband lightcurves in units of AB Magnitude
        for transient in transients_GroupA:
            N_trans = transients_GroupA[transient]['Number Function']
            integrand = integrate.quad(N_trans, z_min, z_max)[0]
            N_type = (area/(4*np.pi))*integrand*4.5 # number per year * 4.5 years of LISA

            t_lisa = np.linspace(0.2, 1644, 2000)  # Time of LISA mission

            times = np.linspace(0.2,1644,2000)
            bands = 'lssti'
            freq = redback.utils.bands_to_frequency([bands])

            n_datapoints = 2000
            kwargs = {'output_format': 'flux_density', 'frequency': freq}
            model = transients_GroupA[transient]['Model']


            z_pdf = N_trans(z_list)/np.trapezoid(N_trans(z_list), z_list)
            delta_z = z_list[1] - z_list[0]
            z_cdf = np.cumsum(z_pdf) * delta_z
            z_interp = interp1d(z_cdf, z_list, kind='linear')

            print(f'Producing {int(N_type)} {transient} lightcurves.')
            for i in range(int(N_type)):
                try:
                    parameters = {}
                    parameters = update_parameters(parameters, transient)
                    parameters['redshift'] = z_interp(np.random.rand())
                    sim_obs = SimulateGenericTransient(model=model, parameters=parameters, times=times, data_points=n_datapoints,
                                                        model_kwargs=kwargs, noise_term=0.02)

                    sigma_bg = 2e-9
                    randindex = np.random.randint(0,2000)
                    t0 = t_lisa[randindex]

                    background_df = pd.DataFrame()
                    background_df['time'] = t_lisa
                    background_df['true_output'] = 3.6e-8           ## milliJanskys for ~35 mag 
                    background_df['output'] = add_noise(background_df['true_output'], sigma_bg)
                    background_df['output_error'] = sigma_bg

                    source_df = sim_obs.data
                    source_df['time'] = source_df['time'] + t0 - 0.2
                    source_df = source_df[source_df['time'] < 1644]
                    source_df['band'] = bands

                    combined_df = background_df.merge(source_df, on=['time'], how='left', suffixes=('_bg', '_src'))

                    combined_df['true_output_src'] = combined_df['true_output_src'].fillna(0)
                    combined_df['output_src'] = combined_df['output_src'].fillna(0)
                    combined_df['output_error_src'] = combined_df['output_error_src'].fillna(0)

                    combined_df['true_output'] = (combined_df['true_output_bg'] + combined_df['true_output_src'])
                    combined_df['output'] = (combined_df['output_bg'] + combined_df['output_src'])
                    combined_df['output_error'] = np.sqrt(combined_df['output_error_src']**2 + combined_df['output_error_bg']**2)

                    combined_df['magnitude'] = flux_to_mag(combined_df['output'])
                    combined_df['true_magnitude'] = flux_to_mag(combined_df['true_output'])
                    combined_df['magnitude_error'] = flux_error_to_mag(combined_df['output_error'], combined_df['output'])

                    master_data = combined_df[['time','band','true_magnitude','magnitude','magnitude_error']]

                    master_data = master_data[(master_data['time'] < t0) | (master_data['band'].notna())]
                    master_data['band'] = master_data['band'].fillna(bands)

                    # saving lightcurve to FITS file
                    obj_id = f'{i:05d}'
                    filename = f'{transient}_lightcurve_{obj_id}.fits'
                    filepath = f'TransientContaminantsOutput/{filename}'
                    astro_table = Table.from_pandas(master_data)
                    astro_table.meta = {'Model': model,
                                        'ID': obj_id}
                    astro_table.meta.update(parameters)
                    astro_table.write(filepath, overwrite=True)

                except Exception:
                    print(f'Error producing {transient} lightcurve.')
        
        for transient in transients_GroupB:
            N_trans = transients_GroupB[transient]['Number Function']
            integrand = integrate.quad(N_trans, z_min, z_max)[0]
            N_type = (area/(4*np.pi))*integrand*4.5

            times = np.linspace(0.2,1644,2000)
            t_lisa = np.linspace(0.2,1644,2000)
            bands = 'lssti'

            n_datapoints = 2000
            kwargs = {'output_format':'magnitude', 'bands': bands}
            model = transients_GroupB[transient]['Model']     

            z_pdf = N_trans(z_list)/np.trapezoid(N_trans(z_list), z_list)
            delta_z = z_list[1] - z_list[0]
            z_cdf = np.cumsum(z_pdf) * delta_z
            z_interp = interp1d(z_cdf, z_list, kind='linear')
            if transient == 'SNIa':
                print(f'Producing {int(N_type)} {transient} lightcurves.')
                for i in range(int(N_type)):
                    try:
                        parameters = redback.priors.get_priors(model=model).sample()
                        parameters['redshift'] = z_interp(np.random.rand())
                        sim_obs = SimulateGenericTransient(model=model, parameters=parameters,times=times, data_points=n_datapoints,
                                                            model_kwargs=kwargs, noise_term=0.02)
                        source_df = sim_obs.data

                        sigma_bg = 1e-9
                        randindex = np.random.randint(0,2000)
                        t_start = t_lisa[randindex]

                        background_df = pd.DataFrame()
                        background_df['time'] = t_lisa
                        background_df['true_output'] = 3.6e-8           
                        background_df['output'] = add_noise(background_df['true_output'], sigma_bg)
                        background_df['output_error'] = sigma_bg

                        source_df['time'] = source_df['time'] + t_start - 0.2
                        source_df = source_df[source_df['time'] < 1644]
                        source_df['band'] = bands
                        source_df['true_flux'] = mag_to_flux(source_df['true_output'])
                        source_df['flux'] = mag_to_flux(source_df['output'])

                        t_peak = parameters['peak_time'] + t_start

                        if source_df['time'].max() > t_peak:
                            tail_df = source_df[(source_df['time'] > t_peak) & (source_df['time'] < (t_peak + 40))]

                            t_tail = tail_df['time'].values
                            f_tail = tail_df['true_flux'].values

                            t0 = t_tail[0]
                            coeffs = np.polyfit(t_tail - t0, np.log(f_tail), 1)

                            slope, intercept = coeffs
                            tau = -1./slope
                            A = np.exp(intercept)

                            f_extrap = A*np.exp(-(t_tail - t0)/tau)

                            t_full = source_df['time'].values
                            mask = t_full > t_tail.max()

                            true_flux_tail = A*np.exp(-(t_full[mask] - t0)/tau)
                            flux_tail = add_noise(true_flux_tail, 0.2)                 

                            source_df.loc[mask, 'true_flux'] = true_flux_tail
                            source_df.loc[mask, 'flux'] = flux_tail

                        combined_df = background_df.merge(source_df, on=['time'], how='left', suffixes=('_bg', '_src'))

                        combined_df['output_error_src'] = combined_df['output_error_src'].fillna(0)         

                        combined_df['true_flux'] = combined_df['true_flux'].fillna(0)
                        combined_df['flux'] = combined_df['flux'].fillna(0)

                        combined_df['true_output_sum'] = (combined_df['true_output_bg'] + combined_df['true_flux'])
                        combined_df['output_sum'] = (combined_df['output_bg'] + combined_df['flux'])
                        combined_df['output_error'] = np.sqrt((0.02*combined_df['true_flux'])**2 + combined_df['output_error_bg']**2)

                        combined_df['magnitude'] = flux_to_mag(combined_df['output_sum'])
                        combined_df['true_magnitude'] = flux_to_mag(combined_df['true_output_sum'])
                        combined_df['magnitude_error'] = flux_error_to_mag(combined_df['output_error'], combined_df['output_sum'])

                        master_data = combined_df[['time','band','true_magnitude','magnitude','magnitude_error']]

                        master_data = master_data[(master_data['time'] < t0) | (master_data['band'].notna())]
                        master_data['band'] = master_data['band'].fillna(bands)

                        # saving lightcurve to FITS file
                        obj_id = f'{i:05d}'
                        filename = f'{transient}_lightcurve_{obj_id}.fits'
                        filepath = f'TransientContaminantsOutput/{filename}'
                        astro_table = Table.from_pandas(master_data)
                        astro_table.meta = {'Model': model,
                                            'ID': obj_id}
                        astro_table.meta.update(parameters)
                        astro_table.write(filepath, overwrite=True)
                    except Exception:
                        print(f'Error producing {transient} lightcurve.')
            if transient=='SNII':
                print(f'Producing {int(N_type)} {transient} lightcurves.')
                for i in range(int(N_type)):
                    try:
                        parameters = redback.priors.get_priors(model=model).sample()
                        parameters['redshift'] = z_interp(np.random.rand())

                        source = SimulateGenericTransient(model=model, parameters=parameters, times=times, data_points=n_datapoints,
                                                            model_kwargs=kwargs, noise_term=0.02)
                        source_df = source.data
                        sigma_bg = 2e-9
                        randindex = np.random.randint(0,2000)
                        t0 = t_lisa[randindex]

                        background_df = pd.DataFrame()
                        background_df['time'] = t_lisa
                        background_df['true_output'] = 3.6e-8          
                        background_df['output'] = add_noise(background_df['true_output'], sigma_bg)
                        background_df['output_error'] = sigma_bg

                        source_df['time'] = source_df['time'] + t0 -0.2
                        source_df = source_df[source_df['time'] <= 1644]
                        source_df['band'] = bands

                        source_df['flux'] = mag_to_flux(source_df['output'])
                        source_df['true_flux'] = mag_to_flux(source_df['true_output'])

                        source_df = source_df[['time', 'band', 'true_flux', 'flux', 'output_error']]

                        combined_df = background_df.merge(source_df, on=['time'], how='left', suffixes=('_bg', '_src'))

                        combined_df['true_flux'] = combined_df['true_flux'].fillna(0)
                        combined_df['flux'] = combined_df['flux'].fillna(0)
                        combined_df['output_error_src'] = combined_df['output_error_src'].fillna(0)

                        combined_df['true_output_sum'] = (combined_df['true_output'] + combined_df['true_flux'])
                        combined_df['output_sum'] = (combined_df['output'] + combined_df['flux'])

                        combined_df['magnitude'] = flux_to_mag(combined_df['output_sum'])
                        combined_df['true_magnitude'] = flux_to_mag(combined_df['true_output_sum'])
                        combined_df['magnitude_error'] = combined_df['output_error_src']

                        master_data = combined_df[['time','band','true_magnitude','magnitude','magnitude_error']]

                        master_data = master_data[(master_data['time'] < t0) | (master_data['band'].notna())]
                        master_data['band'] = master_data['band'].fillna(bands)

                        # saving lightcurve to FITS file
                        obj_id = f'{i:05d}'
                        filename = f'{transient}_lightcurve_{obj_id}.fits'
                        filepath = f'TransientContaminantsOutput/{filename}'
                        astro_table = Table.from_pandas(master_data)
                        astro_table.meta = {'Model': model,
                                            'ID': obj_id}
                        astro_table.meta.update(parameters)
                        astro_table.write(filepath, overwrite=True)
                    except Exception:
                        print(f'Error producing {transient} lightcurve.')

        return