import matplotlib.pyplot as plt
from astropy.io import fits
def colors(f, w, s):
    plt.figure(figsize=(6, 3))
    for i in f:
        plt.plot(w,i)
    plt.legend([s])
    
    
    
def plot_sp(filename):
    with fits.open(filename) as hdul:
        fB = hdul['B_FLUX'].data  # Extract flux
        wB = hdul['B_WAVELENGTH'].data
        fR = hdul['R_FLUX'].data  # Extract flux
        wR = hdul['R_WAVELENGTH'].data
        fZ = hdul['Z_FLUX'].data  # Extract flux
        wZ = hdul['Z_WAVELENGTH'].data
        red = hdul['REDSHIFTS'].data['z']
        info = hdul['FIBERMAP'].data
        
        type = hdul['REDSHIFTS'].data['SPECTYPE']
        
    print('RA:', info['TARGET_RA'],'Dec:', info['TARGET_DEC'],'ID:', info['TARGETID'])
    print('type',type)
    print('wavelength number in B R Z:', len(wB), len(wR), len(wZ))
    
    print('Redshift', red)
    colors(fB, wB, 'B')
    colors(fR, wR, 'R')
    colors(fZ, wZ, 'Z')
    plt.xlabel('Wavelength')
    plt.figure(figsize=(6, 3))
    plt.imshow(fZ, origin='lower', cmap='gray', aspect='auto')  # 'origin=lower' to display the image in the correct orientation
    plt.colorbar(label='Flux (10**-17 erg/s/cm^2/Angstrom)')
    plt.title('FLUX Image')
    plt.xlabel('Pixel Axis 1')
    plt.ylabel('Pixel Axis 2')
    plt.tight_layout()
    plt.show()
    
    
    
plot_sp('3-sv1-bright-17681.fits')

