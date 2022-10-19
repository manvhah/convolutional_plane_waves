import Soundfield as sf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# plt.rcParams["grid.color"]='#e5e5e5'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['mathtext.cal'] = 'serif'
mpl.rcParams['font.size'] = 18
# import mha

level_p = lambda p: 20*np.log10(np.abs(p))-20*np.log10(2e-5)
level_u = lambda u: 20*np.log10(np.abs(u))-20*np.log10(5e-8)
level_i = lambda i: 10*np.log10(np.abs(i))-10*np.log10(1e-12)

def test_radiation_filter():
    field_opts = dict({
        "measurement" :'019_lecture_room',
        "frequency"   : np.array([600]),
        "dx"          : .025,
        })
    soundfield_obj = Soundfield(**field_opts)
    fc = soundfield_obj.f[0]*4 # cutoff frequency
    trunc = 2 # at which zero to truncate bessel filter
    pf, hs = apply_radiation_filter(soundfield_obj.p, fc, soundfield_obj.dx, truncation = trunc)

    # plotting
    dims =[[np.min(soundfield_obj.r[:,1]), np.max(soundfield_obj.r[:,1])],
           [np.min(soundfield_obj.r[:,0]), np.max(soundfield_obj.r[:,0])],
           [np.min(soundfield_obj.r[:,2]), np.max(soundfield_obj.r[:,2])]]
    aperture = np.array(dims).ravel()[:4]
    dx_2     = np.diff(aperture[:2])/(soundfield_obj.p.shape[0]-1)/2
    dy_2     = np.diff(aperture[2:])/(soundfield_obj.p.shape[1]-1)/2
    print(dx_2)
    print(aperture)
    extent = tuple((aperture + np.array([-dx_2, dx_2, -dy_2, dy_2]).T)[0])

    hdx = hs.shape[0]*soundfield_obj.dx
    hdy = hs.shape[1]*soundfield_obj.dx
    hsxy = [-hdx/2, hdx/2, -hdy/2, hdy/2]
    print(hsxy)
    hsextent = tuple((hsxy + np.array([-dx_2, dx_2, -dy_2, dy_2]).T)[0])

    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(10,8))
    plt.clf()
    fig.text(.05,.95, "Applying spatial sound field filtering after \"Direct formulation \
of the supersonic acoustic intensity in space domain\" by E. Fernandez-Grande and F. Jacobsen \
JASA 131(1), Jan 2012, pp. 186-193", wrap = True)
    plt.subplot(221)
    plt.imshow(20*np.log10(np.abs(soundfield_obj.p)), extent = extent, origin = 'lower')
    plt.title('$p_{measurement}$' +' at {:.0f} Hz'.format(soundfield_obj.f[0]))
    plt.colorbar()
    plt.ylabel('$y$ [m]')
    plt.xlabel('$x$ [m]')
    plt.grid()
    plt.subplot(222)
    plt.imshow(20*np.log10(np.abs(hs)), extent = hsextent, origin = 'lower')
    plt.title('$h_s$, $f_c$'+" {:.0f} Hz,\n truncation at {}th zero".format(fc,trunc))
    plt.colorbar()
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')
    plt.grid()
    plt.subplot(223)
    plt.imshow(20*np.log10(np.abs(pf)), extent = extent, origin = 'lower')
    plt.title('$p_{filtered}$')
    plt.colorbar()
    plt.ylabel('$y$ [m]')
    plt.xlabel('$x$ [m]')
    plt.grid()
    plt.subplot(224)
    plt.imshow(10*np.log10(np.abs(soundfield_obj.p-pf)**2/np.abs(soundfield_obj.p)**2), extent = extent, origin = 'lower')
    plt.title('MSND $|p_m - p_f| \, / \, |p_m|$')
    plt.colorbar()
    plt.ylabel('$y$ [m]')
    plt.xlabel('$x$ [m]')
    plt.grid()
    plt.draw()
    plt.show()

def test_monoplane():
    config = sf.default_soundfield_config("monoplane")
    sfo = sf.Soundfield(**config)
    print("wavenumber vector", sfo.plane_k)
    plt.figure(1)
    plt.clf()
    opts = dict(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(331)
    ax.imshow(sfo.p.real,origin='lower')
    plt.title('Re p')
    ax.set(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(332)
    # plt.quiver(sfo.u[0].real,sfo.u[1].real)
    X,Y = np.meshgrid(np.arange(0,sfo.shp[0]),np.arange(sfo.shp[1]))
    plt.streamplot(X,Y,sfo.u[0].real,sfo.u[1].real)
    ax.set(**opts)
    plt.title('Im u_xy')
    ax = plt.subplot(333)
    # plt.quiver(sfo.u[0].real,sfo.u[2].real)
    plt.streamplot(X,Y,sfo.u[0].real,sfo.u[2].real)
    ax.set(**opts)
    plt.title('Re u_xz')
    I = sfo.IJ.real
    J = sfo.IJ.imag
    ax = plt.subplot(334)
    # plt.quiver(J[0],J[1])
    plt.streamplot(X,Y,J[0],J[1])
    ax.set(**opts)
    plt.title('J_xy')
    ax = plt.subplot(335)
    # plt.quiver(I[0],I[1])
    plt.streamplot(X,Y,I[0],I[1])
    ax.set(**opts)
    plt.title('I_xy')
    ax = plt.subplot(336)
    # plt.quiver(I[0],I[2])
    plt.streamplot(X,Y,I[0],I[2])
    ax.set(**opts)
    plt.title('I_xz')

    ax = plt.subplot(337)
    im = plt.contour(level_i(np.linalg.norm(sfo.IJ.real,axis=0)),20)
    plt.colorbar(im)
    ax.set(**opts)
    plt.title('L_I')
    ax = plt.subplot(338)
    im = plt.contourf(level_i(np.linalg.norm(sfo.IJ.real,axis=0)),30)
    plt.colorbar(im)
    ax.set(**opts)
    plt.title('L_I')

    plt.draw()
    plt.show()

def test_mono():
    config = sf.default_soundfield_config("mono")
    wl = 343/config['frequency']
    config.update({
        'dx'   : wl/10,
        'rdim' : [[-3*wl, 3*wl],[-3*wl, 3*wl],[wl,wl]],
        })
    sfo = sf.Soundfield(**config)
    plt.figure(2)
    plt.clf()
    ax = plt.subplot(331)
    ax.imshow(sfo.p.real,origin='lower')
    plt.title('Re p')
    ax.set(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(332)
    ax.imshow(np.log10(np.linalg.norm(sfo.u,axis=0)),origin='lower')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('|u|')
    ax = plt.subplot(333)
    ax.imshow(np.log10(np.linalg.norm(sfo.IJ.imag,axis=0)),origin='lower')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('|I|')
    ax = plt.subplot(334)
    plt.quiver(sfo.u[0].real.T,sfo.u[1].real.T)
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('Re u_xy')
    ax = plt.subplot(335)
    ax.imshow(sfo.u[2].real,origin='lower')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('Re u_z')
    ax = plt.subplot(336)
    ax.imshow(np.log10(np.abs(sfo.u[2])),origin='lower')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('|u_z|')
    I = sfo.IJ.real
    J = sfo.IJ.imag
    # ax = plt.subplot(337)
    # ax.imshow(I[0].T,origin='lower')
    # ax.set(xticks=[],yticks=[],aspect='equal')
    # plt.title('I_x')
    # ax = plt.subplot(338)
    # ax.imshow(I[1].T,origin='lower')
    # ax.set(xticks=[],yticks=[],aspect='equal')
    # plt.title('I_y')
    # ax = plt.subplot(339)
    # ax.imshow(I[2].T,origin='lower')
    # ax.set(xticks=[],yticks=[],aspect='equal')
    # plt.title('I_z')
    I /= np.linalg.norm(I,axis=0)[np.newaxis,...]
    J /= np.linalg.norm(J,axis=0)[np.newaxis,...]
    ax = plt.subplot(337)
    plt.quiver(J[0],J[1])
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('J_xy/|J|')
    ax = plt.subplot(338)
    plt.quiver(I[0],I[1])
    plt.title('I_xy/|I|')
    ax.set(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(339)
    plt.quiver(I[0],I[2])
    plt.title('I_xz/|I|')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.draw()
    plt.show()

def test_xyplane():
    config = sf.default_soundfield_config("xyplane")
    sfo = sf.Soundfield(**config)
    print("wavenumber vector", sfo.plane_k)
    plt.figure(1)
    plt.clf()
    ax = plt.subplot(231)
    ax.imshow(sfo.p.real.T,origin='lower')
    plt.title('Re p')
    ax.set(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(232)
    plt.quiver(sfo.u[0].real.T,sfo.u[1].real.T)
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('Re u_xy')
    ax = plt.subplot(233)
    ax.imshow(sfo.u[1].real,origin='lower')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('Re u_y')
    I = sfo.IJ.real
    J = sfo.IJ.imag
    ax = plt.subplot(234)
    plt.quiver(J[0],J[1])
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('J_xy')
    ax = plt.subplot(235)
    plt.quiver(I[0],I[1])
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('I_xy')
    ax = plt.subplot(236)
    plt.quiver(I[0],I[2])
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('I_xz')
    plt.draw()
    plt.show()

def test_scatter_sphere():
    config = sf.default_soundfield_config("scatter_sphere")
    sfo = sf.Soundfield(**config)
    plt.figure(1)
    plt.clf()
    ax = plt.subplot(131)
    ax.imshow(sfo.p.real,origin='lower')
    plt.title('Re p')
    ax.set(xticks=[],yticks=[],aspect='equal')
    ax = plt.subplot(132)
    ax.imshow(np.abs(sfo.p),origin='lower')
    plt.title('|p|')
    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.draw()
    plt.show()


def plot_frequency_response(field,**fieldkwargs):
    mpl.rcParams['font.size'] = 16
    if field == '011':
        field, df = '011', .1 # Hz
        minf,maxf = 20,1000
    else:
        field, df = '019', .5 # Hz
        # minf,maxf = 20,5000
        minf,maxf = 50,2050

    config = sf.default_soundfield_config(field,**fieldkwargs)
    config.update(dict(frequency = np.arange(minf,maxf+1e-4,df)))
    sfo0 = sf.Soundfield(**config)
    markerf = np.array([500, 600, 700, 800, 900, 1000, 1250, 1600, 2000])

    fig = plt.figure(1,figsize=(11,5))
    plt.clf()
    from matplotlib.gridspec import GridSpec
    ax = fig.add_subplot(GridSpec(1,1,bottom=.25)[0])

    # spatial mean(abs())
    pp = np.mean(np.abs(sfo0.fp),axis=0)
    ax.plot(sfo0.f,20*np.log10(pp),'k',lw=2,label='$\overline{\|p\|}$',zorder=2.4)
    # single point
    # pp = np.abs(sfo0.p[0,0].T)
    # ax.plot(sfo0.f,20*np.log10(pp),alpha=.5,label='(0,0) cm')
    # +- spatial std(abs())
    ps = np.std(np.abs(sfo0.fp),axis=0)
    ax.fill_between(sfo0.f,20*np.log10(pp-ps),20*np.log10(pp+ps),
            lw=0,color='#dddddd', label='$\overline{\|p\|} \pm$ stdev($\|p\|$)',zorder=2.2)
    # all responses
    # ax.plot(sfo0.f,20*np.log10(np.abs(sfo0.fp.T)),'k',alpha=.01)

    ax.plot(sfo0.f,20*np.log10(np.abs(sfo0.fp[70].T)),'C0--',
            lw=1,label='$\|p(r_1)\|$',zorder=2.5)
    ax.plot(sfo0.f,20*np.log10(np.abs(sfo0.fp[72].T)),'C1',
            lw=1,label='$\|p(r_2)\|$',zorder=2.45)

    # config.update(dict(frequency = markerf))
    # sfo1 = sf.Soundfield(**config)
    # ax.plot(sfo1.f,20*np.log10(np.abs(sfo1.p[5,5].T)),'rx',alpha=.7)
    axvopts = dict(zorder=2.1,color='k',alpha=.5,linestyle='dashed')

    ## mark test frequencies
    # ax.axvline(markerf[0],**axvopts,label='test freq.')
    # [ax.axvline(ff,**axvopts) for ff in markerf[1:]]

    ax.set(
            # title = 'Frequency responses, room' + field,
            ylabel = r'$20 \log_{10} (\cdot)$ [dB rel a.u.]',
            xlabel='Frequency [Hz]',
            xlim=[minf,maxf],
            ylim=[-3,30],
            )
    # xt = markerf
    xt = [50, 240] + list(markerf)
    xtl = [xx for xx in xt]
    xtl[1]= '240\n'+'$\,(f_S)$'
    ax.set_xticks(xt,labels = xtl, rotation=45)
    plt.grid(True)
    plt.legend(ncol=4,loc='lower center')
    plt.savefig('./figures/thesis_fr_'+field+'.pdf',bbox_inches='tight',pad_inches=0.0)


def test_monoplane():
    plt.rcParams["figure.autolayout"] = True
    config = sf.default_soundfield_config("monoplane")
    sfo = sf.Soundfield(**config)

    fig, ax = plt.subplots(1, 1)

    qr = plt.quiver(sfo.IJ[0].real,sfo.IJ[1].real)

    ax.set(xticks=[],yticks=[],aspect='equal')
    plt.title('I(t)_xy')

    # # import random as rd
    def animate(num, qr, IJ):
        freq = 1/10000
        IJ *= np.exp(2j*np.pi*freq*num)
        qr.set_UVC(IJ[0].real, IJ[1].real)
        # qr.set_color((rd.random(), rd.random(), rd.random(), rd.random()))
        return qr,

    anim = mpl.animation.FuncAnimation(fig, animate, fargs=(qr, sfo.IJ), interval=50, blit=False)

    # anim.save('intensity.mp4')
    plt.show()

if __name__ == "__main__":
    plot_frequency_response('019')

    # test_monoplane()
    # test_mono()
    # test_xyplane()
    # test_scatter_sphere()
