import numpy as np

class AmmoniumSoilModel:
    def __init__(self):
        self.D = 5.0
        self.v = 2.0
        self.k_a = 0.05
        self.k_d = 0.02
        self.C_max = 50.0
        self.k_nitr = 0.01
        self.V_max = 2.0
        self.K_m = 5.0
        self.z_max = 50.0
        self.t_max = 15.0
        self.nz = 25
        self.nt = 500
        self.C_initial = 20.0
        self.C_ads_initial = 5.0
        self.C = None
        self.C_ads = None
        self.z = None
        self.t = None

    def calculate_R_components(self, C, C_ads):
        C = max(0, C)
        C_ads = max(0, min(C_ads, self.C_max))
        if C_ads >= self.C_max:
            R_adsorb = -self.k_d * C_ads
        else:
            R_adsorb = self.k_a * (self.C_max - C_ads) * C / (C + 1.0) - self.k_d * C_ads
        R_nitr = min(self.k_nitr * C, C * 0.5)
        if C > 0:
            R_plant = self.V_max * C / (self.K_m + C)
            R_plant = min(R_plant, C * 0.3)
        else:
            R_plant = 0
        R = R_nitr + R_plant - R_adsorb
        if not all(np.isfinite([R, R_adsorb, R_nitr, R_plant])):
            R, R_adsorb, R_nitr, R_plant = 0, 0, 0, 0
        return R, R_adsorb, R_nitr, R_plant

    def check_stability(self, dt, dz):
        courant_number = self.v * dt / dz
        diffusion_number = self.D * dt / (dz**2)
        if courant_number > 1.0:
            raise ValueError()
        if diffusion_number > 0.5:
            raise ValueError()
        return courant_number, diffusion_number

    def solve_model(self):
        self.z = np.linspace(0, self.z_max, self.nz)
        self.t = np.linspace(0, self.t_max, self.nt)
        dz = self.z[1] - self.z[0]
        dt = self.t[1] - self.t[0]
        try:
            self.check_stability(dt, dz)
        except ValueError:
            max_dt_courant = 0.9 * dz / self.v if self.v > 0 else float('inf')
            max_dt_diffusion = 0.4 * dz**2 / self.D if self.D > 0 else float('inf')
            max_dt = min(max_dt_courant, max_dt_diffusion, dt)
            if max_dt < dt:
                new_nt = int(self.t_max / max_dt) + 1
                self.nt = min(new_nt, 5000)
                self.t = np.linspace(0, self.t_max, self.nt)
                dt = self.t[1] - self.t[0]
        self.C = np.zeros((self.nt, self.nz))
        self.C_ads = np.zeros((self.nt, self.nz))
        self.C[0, :] = self.C_initial
        self.C_ads[0, :] = self.C_ads_initial
        self.C[:, 0] = self.C_initial
        for n in range(self.nt - 1):
            for i in range(1, self.nz - 1):
                C_curr = max(0, self.C[n, i])
                C_ads_curr = max(0, self.C_ads[n, i])
                if not np.isfinite(C_curr) or not np.isfinite(C_ads_curr):
                    C_curr = self.C_initial * 0.1
                    C_ads_curr = self.C_ads_initial * 0.1
                try:
                    R, R_adsorb, R_nitr, R_plant = self.calculate_R_components(C_curr, C_ads_curr)
                    if not all(np.isfinite([R, R_adsorb, R_nitr, R_plant])):
                        R, R_adsorb, R_nitr, R_plant = 0, 0, 0, 0
                except:
                    R, R_adsorb, R_nitr, R_plant = 0, 0, 0, 0
                d2C_dz2 = (self.C[n, i+1] - 2*self.C[n, i] + self.C[n, i-1]) / (dz**2)
                dC_dz = (self.C[n, i+1] - self.C[n, i-1]) / (2*dz)
                if not np.isfinite(d2C_dz2):
                    d2C_dz2 = 0
                if not np.isfinite(dC_dz):
                    dC_dz = 0
                dC_dt = self.D * d2C_dz2 - self.v * dC_dz - R
                max_change = C_curr * 0.1
                if abs(dC_dt * dt) > max_change and C_curr > 0:
                    dC_dt = np.sign(dC_dt) * max_change / dt
                self.C[n+1, i] = max(0, self.C[n, i] + dt * dC_dt)
                dC_ads_dt = R_adsorb
                max_ads_change = max(C_ads_curr * 0.1, 0.1)
                if abs(dC_ads_dt * dt) > max_ads_change:
                    dC_ads_dt = np.sign(dC_ads_dt) * max_ads_change / dt
                self.C_ads[n+1, i] = max(0, self.C_ads[n, i] + dt * dC_ads_dt)
                if self.C_ads[n+1, i] > self.C_max:
                    self.C_ads[n+1, i] = self.C_max
            self.C[n+1, -1] = max(0, self.C[n+1, -2])
            self.C_ads[n+1, -1] = max(0, self.C_ads[n+1, -2])
            if np.any(self.C[n+1, :] > 1000 * self.C_initial):
                raise RuntimeError()
