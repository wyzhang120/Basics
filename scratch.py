import numpy as np
import matplotlib.pyplot as plt


class ricker_fft2:
    def __init__(self, N=1024, fc=50., tmax=1., delay=None):
        """
        $\int e^(-iwt) dt$ as Fourier transform (normal convention);
        the numpy function is np.fft.fft and output is scaled by 1/N
        :param N: int, nubmer of samples
        :param fc: float, unit = Hz, central freqency
        :param tmax: float, unit =s, total length of time
        :param zerophase: bool, whether a zero phase wavelet

        """
        fc = float(fc)
        tmax = float(tmax)
        dt = tmax / (N-1)
        df = 1. / tmax
        istart = -N//2 + 1
        nroll = -istart
        t = np.arange(istart, N // 2 + 1) * dt
        f = np.arange(istart, N // 2 + 1) * df
        # arrange f as positve, negative freqs
        f = np.roll(f, -nroll)

        # freq domain ricker
        ricker_f = 2 * f**2 / (np.sqrt(np.pi) * fc**3) * np.exp(-(f/fc)**2)

        if delay is None:
            delay = 1.5 / fc
        ricker_f = ricker_f * np.exp(-1j * 2 * np.pi * f * delay)

        # time domain ricker
        ricker_t = N * np.real(np.fft.ifft(ricker_f))


        amp = np.absolute(ricker_f)
        phase = np.unwrap(np.angle(ricker_f, False))

        # ricker_f[0] contains the zero frequency term,
        # ricker_f[1:N//2] contains the positive-frequency terms,
        # ricker_f[N//2 + 1:] contains the negative-frequency terms,
        #                     in increasing order starting from the most negative frequency
        self.delay = delay
        self.fc = fc
        self.dt = dt
        # arange t and ricker_t in the order of increasing time; the zero phase case contains negative time

        self.ricker_t = np.roll(ricker_t, nroll)
        self.f = f
        self.t = t
        self.df = df
        self.ricker_f = ricker_f
        self.amp = amp
        self.phase = phase
        self.nroll = nroll


    def plot_ricker(self):
        idxNq = len(self.t) // 2
        fig2, ax2 = plt.subplots(1, 3)
        fig2.set_size_inches(18, 6)
        tdelay = self.delay
        ax2[0].plot(self.t, self.ricker_t)
        ax2[0].set_title(r'$t_{{delay}}$ = {:.4f} s'.format(tdelay))
        ax2[0].set_xlabel('t (s)')
        ax2[1].plot(self.f[: idxNq + 1], self.amp[: idxNq + 1])
        ax2[1].set_xlabel('f (Hz)')
        ax2[1].set_ylabel('Amp')
        ax2[1].set_title(r'Ricker wavelet, $f_c$ = {:g} Hz'.format(self.fc))
        slope = (self.phase[idxNq] - self.phase[0]) / (self.f[idxNq] - self.f[0])
        ax2[2].plot(self.f[: idxNq + 1], self.phase[: idxNq + 1])
        ax2[2].set_xlabel('f (Hz)')
        ax2[2].set_ylabel('Phase (radians)')
        ax2[2].set_title(r'$d\phi/d\omega$ = {:.4f}'.format(slope / (2 * np.pi)))
        plt.show()


fc = 50.
rk3 = ricker_fft2(N=256, fc=fc, tmax=1., delay=1.5/fc)
rk3.plot_ricker()
